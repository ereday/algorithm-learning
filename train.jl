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
    end

    isa(args, AbstractString) && (args=split(args))
    o = parse_args(args, s; as_symbols=true); println(o); flush(STDOUT)
    sr = o[:seed] > 0 ? srand(o[:seed]) : srand()
    o[:atype] = eval(parse(o[:atype]))
    data_generator = get_data_generator(o[:task])

    # init model, params etc.
    w = wfix = opts = s2i = i2s = nothing
    a2i, i2a = initvocab(get_actions(o[:task]))
    if o[:loadfile] == nothing
        s2i, i2s = initvocab(get_symbols(o[:task]))
        w = initweights(
            o[:atype],o[:units],length(s2i),length(a2i),o[:controller],o[:dist])
        opts = initopts(w,o[:optim])
    else
        o[:task] = load(o[:loadfile], "task")
        w = load(o[:loadfile], "w")
        # opts - not yet!
        # opts = load(o[:loadfile])
        o[:start] = load(o[:loadfile], "complexity")
        s2i, i2s = initvocab(get_symbols(o[:task]))
    end
    mem = ReplayMemory(o[:capacity])
    if !o[:supervised]
        wfix = Dict(map(k->(k,copy(w[k])), keys(w)))
    end

    # C => complexity
    # c => controller cell
    validate = !o[:supervised] ? rlvalidate : slvalidate
    steps_done = 0
    for C = o[:start]:o[:step]:o[:end]
        seqlen = div(C, complexities[o[:task]])
        val = map(xi->data_generator(seqlen), [1:o[:nvalid]...])
        iter = 0
        lossval = 0

        while true
            trn = map(xi->data_generator(seqlen), 1:o[:nepisodes])
            x = map(xi->xi[1], trn)
            y = map(xi->xi[2], trn)
            actions = map(xi->xi[3], trn)
            game = Game(x,y,actions,o[:task])
            T = length(game.symgold[1])

            if iter == 0

            end

            h,c = initstates(
                o[:atype],o[:units],o[:batchsize],o[:controller])

            # FIXME: sl.iter != rl.iter (but how)
            if o[:supervised]
                inputs = make_inputs(game, s2i, a2i)
                outputs = make_outputs(game, s2i, a2i)
                timesteps = length(inputs)
                batchsize = o[:batchsize]
                batchloss = sltrain!(w,inputs,outputs,h,c,opts)
                batchloss = batchloss / (batchsize * timesteps)
                iter += 1
                lossval = update_lossval(lossval,batchloss,iter)
            else # rl train
                # run new episodes
                steps_done = run_episodes!(
                    game, mem, w, h, c, s2i, i2s, a2i, i2a, steps_done; o=o)


                # train with batches from memory
                for k = 1:o[:period]
                    # info(k)
                    batchsize = o[:batchsize]
                    # make_batch => for targets
                    batch = make_batch(mem, wfix, o[:discount], o[:nsteps],
                                       s2i, a2i, o[:batchsize])
                    batchloss = rltrain!(w,batch...,opts)
                    batchloss = batchloss / batchsize
                    iter += 1
                    lossval = update_lossval(lossval,batchloss,iter)
                end
                println("lagn")
                # empty!(mem)

                # if iter % o[:update] == 0
                #     wfix = Dict(map(k->(k,copy(w[k])), keys(w)))
                #     empty!(mem)
                # end
            end

            # perform the validation
            if iter % o[:period] == 0
                println("lossval:$lossval")
                acc = validate(w,s2i,i2s,a2i,i2a,val,o)
                println("(iter:$iter,acc:$acc)")
                if acc > 0.98
                    println("$C converged in $iter iteration")
                    # Knet.gc(); gc(); Knet.gc()

                    # save model
                    if o[:savefile] != nothing
                        save(o[:savefile],
                             "w", map(Array, w),
                             # need something like above for opts
                             # "opts", opts,
                             "task", o[:task],
                             "complexity", C)
                    end

                    if !o[:supervised]
                        wfix = Dict(map(k->(k,copy(w[k])), keys(w)))
                        empty!(mem)
                    end

                    break
                end
            end # validation
        end # while true
    end # one complexity step
end

function sltrain!(w,x,y,h,c,opts)
    values = []
    gloss = slgradient(w,x,y,h,c; values=values)
    update!(w, gloss, opts)
    return values[1]
end

function rltrain!(w,targets,x,y,a,h,c,vs,opts)
    values = []
    gloss = rlgradient(w,targets,x,y,a,h,c,vs; values=values)
    update!(w, gloss, opts)
    return values[1]
end

function slvalidate(w,s2i,i2s,a2i,i2a,data,o)
    batches = []
    for k = 1:o[:batchsize]:length(data)
        push!(batches, data[k:min(k+o[:batchsize]-1,length(data))])
    end

    ncorrect = 0
    for batch in batches
        x = map(xi->xi[1], batch)
        y = map(xi->xi[2], batch)
        actions = map(xi->xi[3], batch)
        game = Game(x,y,actions,o[:task])
        T = length(game.next_actions[1])
        # info(game.input_tapes[1])

        correctness = trues(length(batch))
        h,c = initstates(o[:atype],o[:units],o[:batchsize],o[:controller])
        for k = 1:T
            x1,x2 = make_input(game, s2i, a2i)
            y1,y2 = make_output(game, s2i, a2i)
            @show i2s[indmax(x1[:,1])],i2a[indmax(x2[:,1])]
            x1 = convert(o[:atype], x1)
            x2 = convert(o[:atype], x2)
            cout, h, c = propagate(w[:wcont],w[:bcont],vcat(x1,x2),h,c)
            y1pred = predict(w[:wsymb],w[:bsymb],cout)
            y2pred = predict(w[:wact],w[:bact],cout)

            y1pred = convert(Array, y1pred)
            y1pred = mapslices(indmax,y1pred,1)
            y1pred = map(yi->i2s[yi], y1pred)

            y2pred = convert(Array, y2pred)
            y2pred = mapslices(indmax,y2pred,1)
            y2pred = map(yi->i2a[yi], y2pred)
            @show y1pred[1],y2pred[1]

            # check correctness
            for i = 1:length(y1pred)
                if y1pred[i] != game.symgold[i][k]
                    correctness[i] = false
                end
            end

            for i = 1:game.ninstances
                game.prev_actions[i][k] = y2pred[i]
            end
            move_timestep!(game,y2pred)
        end
        ncorrect += sum(correctness)
    end
    return ncorrect / length(data)
end

function rlvalidate(w,s2i,i2s,a2i,i2a,data,o)
    ncorrect = 0
    batch = data
    x = map(xi->xi[1], batch)
    y = map(xi->xi[2], batch)
    actions = map(xi->xi[3], batch)
    game = Game(x,y,actions,o[:task])
    correctness = trues(length(batch))
    atype = typeof(w[:wcont])
    T = length(game.next_actions[1])

    for k = 1:game.ninstances
        # warn(k)
        reset!(game)
        h,c = initstates(o[:atype],o[:units],o[:batchsize],o[:controller])
        input_action = "<s>"
        predicted = []
        this_tape = game.gold_tapes[k]
        this_actions = []
        t = 0
        while true
            input_symbol = read_symbol(game.input_tapes[k], game.pointers[k])

            # prepare input array
            x = zeros(length(s2i)+length(a2i),1)
            x[s2i[input_symbol]] = 1
            x[length(s2i)+a2i[input_action]] = 1
            x = convert(atype, x)

            # propagate controller
            cout, h, c = propagate(w[:wcont],w[:bcont],x,h,c)
            y1pred = predict(w[:wsymb],w[:bsymb],cout)
            y2pred = predict(w[:wact],w[:bact],cout)

            # predict symbol action
            output_symbol = i2s[indmax(convert(Array, y1pred))]
            output_action = i2a[indmax(convert(Array, y2pred))]
            push!(this_actions, output_action)
            input_action = output_action

            move_timestep!(game, k, input_action)
            t += 1

            # if k == 1
            #     @show game.input_tapes[k]
            #     @show this_tape
            #     @show output_symbol,output_action
            # end

            correct = false
            if in(game.task,("copy","reverse","walk"))
                push!(predicted, output_symbol)
                outputs = filter(p->p!=NO_SYMBOL, predicted)
                correct = outputs == this_tape[1:length(outputs)]

            else
                unshift!(predicted, output_symbol)
                outputs = filter(p->p!=NO_SYMBOL, predicted)
                correct = outputs == this_tape[end:-1:end-length(outputs)+1]
            end
            finished = length(outputs)==length(this_tape)?correct:false

            if output_action == "<s>" && length(predicted) != length(this_tape)
                correct = false
            end

            correctness[k] = correct
            !correct && break

            if output_action == "<s>"
                correctness[k] = finished
                break
            end

            if t>3*T
                correctness[k] = false
                break
            end
        end

        if k == 1
            @show game.input_tapes[k]
            @show predicted
            @show this_tape
            @show this_actions
        end
    end
    ncorrect += sum(correctness)
    return ncorrect / length(data)
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
