using Knet
using ArgParse
using JLD
using Combinatorics

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
        w = Dict(k=>convert(o[:atype],v) for (k,v) in w)
        # opts - not yet!
        # opts = load(o[:loadfile])
        # o[:start] = load(o[:loadfile], "complexity")
    end
    opts = initopts(w,o[:optim])

    # C => complexity
    # c => controller cell
    steps_done = 0
    for C = o[:start]:o[:step]:o[:end]
        # prepare validation data
        seqlen = div(C, complexities[o[:task]])
        data = map(xi->data_generator(seqlen), [1:o[:nvalid]...])
        valid = []; actions = []
        for (input,output,action) in data
            push!(valid, Game(input, output, o[:task]))
            push!(actions, action)
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

            # train network while running episodes
            this_loss = train!(
                w,games,s2i,i2s,a2i,i2a,actions,opts,o[:supervised]; o=o)
            lossval = update_lossval(lossval, this_loss, iter)

            lossval = update_lossval(lossval,this_loss,iter)
            if iter % o[:period] == 0
                accuracy, average_reward = validate(w,valid,s2i,i2s,a2i,i2a;o=o)
                println("(iter:$iter,loss:$lossval,accuracy:$accuracy,reward:$average_reward)")
                if accuracy >= o[:threshold] && o[:savefile] != nothing
                    save(o[:savefile],
                         "w", Dict(k=>Array(v) for (k,v) in w),
                         # need something like above for opts
                         # "opts", opts,
                         "task", o[:task],
                         "complexity", C)
                end

                if accuracy >= o[:threshold]
                    println("$C converged in $iter iterations")
                    break
                end
            end

            iter += 1
        end # while true
    end # one complexity step
end

# train while running episodes
function train!(w,games,s2i,i2s,a2i,i2a,actions,opts,supervised; o=Dict())
    train = true
    run_episodes!(w,games,s2i,i2s,a2i,i2a,train,supervised;
                  o=o,opts=opts,actions=actions)
end

# validate games
function validate(w,games,s2i,i2s,a2i,i2a; o=Dict())
    train = supervised = false
    accuracy = run_episodes!(w,games,s2i,i2s,a2i,i2a,train,supervised; o=o)
    return accuracy
end

# this is skeleton for all methods
function run_episodes!(w,games,s2i,i2s,a2i,i2a,train,supervised;
                       o=Dict(), opts=Dict(), actions=[])

    # init state parameters
    atype = get(o, :atype, typeof(w[:wcont]))
    hidden = size(w[:wcont], 1)
    controller = get(o, :controller, "lstm")

    # unrolled
    nsteps = get(o, :nsteps, 20)

    # supervised learning
    if train && supervised
        isempty(actions) && error("actions must not be empty for supervision")

        # (1) prepare input data
        inputs, outputs, masks = make_data(games,s2i,a2i,actions)

        # (2) init controller states
        h, c = initstates(atype, hidden, length(games), controller)

        # (3) train network
        batchloss = sltrain!(w,h,c,inputs,outputs,masks,opts; o=o)

        # (4) return batchloss
        return batchloss
    end

    # needed by qwatkins
    histories = []
    for game in games; push!(histories, []); end

    # run episodes - for both validation and q-learning
    ncorrect = 0
    iter = 1

    cumulative_reward = 0
    for (i,game) in enumerate(games)
        episode_reward = 0
        h, c = initstates(atype, hidden, 1, controller)
        while !game.is_done
            # (1) prepare input data
            input_symbol = read_symbol(game)
            input_action = game.prev_actions[end]
            input = zeros(Cuchar, length(s2i)+length(a2i), 1)
            input[s2i[input_symbol],1] = 1
            input[length(s2i)+a2i[input_action],1] = 1

            # (2) propage controller
            prev_h = prev_c = nothing
            if train
                prev_h = Array(h)
                prev_c = Array(c)
            end
            cout, h, c = propagate(w[:wcont],w[:bcont],convert(atype,input),h,c)

            # (3) take action
            next_action = first(take_action(w[:wact],w[:bact],cout))
            next_action = i2a[next_action]
            move_action, write_action = next_action

            # (4) predict symbol
            y = NO_SYMBOL
            if write_action == WRITE
                y = predict(w[:wsymb],w[:bsymb],cout)
                # symbol = mapslices(indmax, Array(symbol), 1)[1]
                y = indmax(Array(y))
                y = i2s[y]
            end

            move_timestep!(game, y, move_action)
            reward = get_reward(game)
            episode_reward += reward

            # (5) add transition to history if phase is RL train
            if train
                remaining_steps = get_remaining_steps(game)

                # new transition
                transition = Transition(
                    reward,
                    input_symbol,
                    input_action,
                    remaining_steps,
                    prev_h, prev_c,
                    next_action,
                    symbol,
                    game.is_done
                )

                # push it do history
                push!(histories[i], transition)
            end
        end

        cumulative_reward += episode_reward
        if game.output_tape == game.gold_tape
            ncorrect += 1
        end
    end

    # q-watkins training if phase is RL train

    for (i,game) in enumerate(games)
        reset!(game)
    end

    return ncorrect/length(games), cumulative_reward/length(games)
end

function sltrain!(w,h,c,inputs,outputs,masks,opt; o=Dict())
    dw = similar(w)
    for k in keys(w); dw[k] = similar(w[k]); fill!(dw[k], 0); end

    total = num_samples = 0
    maxlen = get(o, :maxlen, 50)
    for i = 1:maxlen:length(inputs)
        lower = i; upper = min(i+maxlen-1,length(inputs))
        x = inputs[lower:upper]
        y = outputs[lower:upper]
        m = masks[lower:upper]

        values = []
        gloss = slgrad(w,x,y,m,h,c; values=values)

        # track loss and training sample values
        total += values[1]; num_samples += values[2]

        # accumulate gradients
        for k in keys(dw); dw[k] += gloss[k]; end

        # unbox states
        h = AutoGrad.getval(h)
        c = AutoGrad.getval(c)
    end

    # clip with num_samples
    for k in keys(dw); dw[k] = dw[k]/num_samples; end
    # FIXME: try a different thing in here

    # update weights
    update!(w,dw,opt)

    return total/num_samples
end

function rltrain!(w,batches,opt; o=Dict())
    # dw = similar(w)
    # for k in keys(w); dw[k] = similar(w[k]); fill!(dw, 0); end

    # total = num_samples = 0
    # batches = make_batches(...)
    # for batch in batches

    # end
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
