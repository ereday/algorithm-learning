using Knet
using ArgParse
using JLD

include("env.jl")
include("model.jl")
include("data.jl")
include("vocab.jl")

const CAPACITY = 20000
const EPS_INIT = 0.9
const EPS_FINAL = 0.005
const EPS_DECAY = 200

function main(args)
    s = ArgParseSettings()
    s.description = ""

    @add_arg_table s begin
        # load/save files
        ("--loadfile"; default=nothing)
        ("--savefile"; default=nothing)
        ("--task"; default="copy")
        ("--seed"; default=-1; arg_type=Int64; help="random seed")
        ("--optim"; default="Rmsprop()")
        ("--units"; default=200)
        ("--controller"; default="feedforward"; help="feedforward or lstm")
        ("--discount"; default=-1.; arg_type=Float64)
        ("--start"; default=6; arg_type=Int64)
        ("--end"; default=50; arg_type=Int64)
        ("--step"; default=4; arg_type=Int64)
        ("--batchsize"; default=20; arg_type=Int64)
        ("--nvalid"; default=60; arg_type=Int64)
        ("--atype"; default=(gpu()>=0 ? "KnetArray{Float32}":"Array{Float32}"))
        ("--period"; default=100; arg_type=Int64)
        ("--dist"; arg_type=String; default="randn"; help="[randn|xavier]")
        ("--supervised"; action=:store_true; help="if not, q-learning")
        ("--capacity"; default=CAPACITY; arg_type=Int64)
        ("--nsteps"; default=20; arg_type=Int64)
        ("--update"; default=5000; arg_type=Int64)
        ("--nepisodes"; default=5000; arg_type=Int64)
        ("--threshold"; default=0.98; arg_type=Float64)
    end

    isa(args, AbstractString) && (args=split(args))
    o = parse_args(args, s; as_symbols=true); println(o); flush(STDOUT)
    sr = o[:seed] > 0 ? srand(o[:seed]) : srand()
    o[:atype] = eval(parse(o[:atype]))
    data_generator = get_data_generator(o[:task])

    # init model, params etc.
    w = opts = s2i = i2s = nothing
    a2i, i2a = initvocab(get_actions(o[:task]))
    s2i, i2s = initvocab(get_symbols(o[:task]))
    if o[:loadfile] == nothing
        w = initweights(
            o[:atype],o[:units],length(s2i),length(a2i),o[:controller],o[:dist])
    else
        o[:task] = load(o[:loadfile], "task")
        w = load(o[:loadfile], "w")
        # opts - not yet!
        # opts = load(o[:loadfile])
        o[:start] = load(o[:loadfile], "complexity")
    end
    opts = initopts(w,o[:optim])

    # C => complexity
    # c => controller cell
    validate = !o[:supervised] ? rlvalidate : slvalidate
    steps_done = 0
    for C = o[:start]:o[:step]:o[:end]
        # prepare validation data
        seqlen = div(C, complexities[o[:task]])
        data = map(xi->data_generator(seqlen), [1:o[:nvalid]...])
        valid = []
        for (input,output,action) in data
            push!(valid, Game(input, output, o[:task]))
        end
        empty!(data)

        iter = 1
        lossval = 0
        while true
            # get examples for training
            trn = map(xi->data_generator(seqlen), 1:o[:nepisodes])

            # build environments
            games = []; actions = []
            for (input,output,action) in trn
                push!(games, Game(input, output, o[:task]))
                push!(actions, [action..., STOP_ACTION])
            end

            # init controller state
            h,c = initstates(
                o[:atype],o[:units],o[:batchsize],o[:controller])

            # train network while running episodes
            this_loss = train!(w,h,c,games,actions,opts,o[:supervised]; o=o)

            # validate network by running episodes
            accuracy = validate(w,h,c,games;o=o)

            lossval = update_lossval(lossval,this_loss,iter)
            println("(iter:$iter,accuracy:$accuracy)")

            if accuracy >= o[:threshold]
                println("$C converged in $iter iterations")
                save(o[:savefile],
                     "w", map(Array, w),
                     # need something like above for opts
                     # "opts", opts,
                     "task", o[:task],
                     "complexity", C)
            end
        end # while true
    end # one complexity step
end

function train!(w,h,c,games,actions,opts,supervised=false; o=Dict())
    train = true
    run_episodes!(w,h,c,games,train,supervised;o=o,opts=opts,actions=actions)
end

function validate(w,h,c,games; o=Dict())
    train = supervised = false
    accuracy = run_episodes!(w,h,c,games,train,supervised; o=o)
    return accuracy
end

function run_episodes!(
    w,games,train,supervised=false; o=Dict(), opts=Dict(), actions=[])

    # init state parameters
    atype = get(o, :atype, typeof(w[:wcont]))
    hidden = size(w[:wcont], 1)
    controller = get(o, :controller, "lstm")

    if train && supervised
        # (1) prepare input data
        inputs, outputs, masks = make_data(games, s2i, a2i)

        # (2) init controller states
        h, c = initstates(atype, hidden, length(games), controller)

        # (3) train network
        batchloss = sltrain!(w,h,c,inputs,outputs,masks,opt; o=o)

        # (4) return batchloss
        return batchloss
    end

    # needed by qwatkins
    histories = []
    for game in games; push!(histories, []); end

    # run episodes
    ncorrect = 0
    done = false
    iter = 1
    for (i,game) in enumerate(games)
        h, c = initstates(atype, hidden, 1, controller)
        while !game.done
            # (1) prepare input data
            input = make_input(game)

            # (2) propage controller
            cout, h, c = propagate(w[:wcont],w[:bcont],input,h,c)

            # (3) take action
            next_action = first(take_action(w[:wact],w[:bact],cout))
            next_action = i2a[next_action]
            move_action, write_action = next_action

            # (4) predict symbol
            symbol = NO_SYMBOL
            if write_action == WRITE
                symbol = predict(w[:wsymb],w[:bsymb],cout)
                symbol = mapslices(indmax, Array(symbol), 1)[1]
                symbol = i2s[symbol]
                move_timestep!(game, symbol, move_action)
            end

            # (5) add transition to history if phase is RL train
            if train
                reward = get_reward(game)
                input = input
                transition = Transition
            end
        end
    end

    # q-watkins training if phase is RL train

    return ncorrect/length(games)
end

function update_lossval(lossval,batchloss,iter)
    if iter < 100
        lossval = (iter-1)*lossval + batchloss
        lossval = lossval / iter
    else
        lossval = 0.01 * batchloss + 0.99 * lossval
    end
    return lossval
end

!isinteractive() && !isdefined(Core.Main, :load_only) && main(ARGS)
