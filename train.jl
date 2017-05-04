using Knet
using ArgParse
using JLD

include("env.jl")
include("model.jl")
include("data.jl")
include("vocab.jl")

function main(args)
    s = ArgParseSettings()
    s.description = ""

    @add_arg_table s begin
        # load/save files
        ("--task"; default="copy")
        ("--seed"; default=-1; arg_type=Int64; help="random seed")
        ("--lr"; default=0.001)
        ("--units"; default=200)
        ("--controller"; default="feedforward"; help="feedforward")
        ("--discount"; default=0.95)
        ("--start"; default=6; arg_type=Int64)
        ("--end"; default=50; arg_type=Int64)
        ("--step"; default=4; arg_type=Int64)
        ("--batchsize"; default=20; arg_type=Int64)
        ("--nvalid"; default=60; arg_type=Int64)
        ("--atype"; default=(gpu()>=0 ? "KnetArray{Float32}" : "Array{Float32}"))
        ("--period"; default=100; arg_type=Int64)
        ("--dist";arg_type=String;default="randn";help="[randn|xavier]")
        # ("--nsymbols"; default=11; arg_type=Int64)
    end

    isa(args, AbstractString) && (args=split(args))
    o = parse_args(args, s; as_symbols=true); println(o); flush(STDOUT)
    s = o[:seed] > 0 ? srand(o[:seed]) : srand()
    o[:atype] = eval(parse(o[:atype]))
    data_generator = get_data_generator(o[:task])

    # init model, params etc.
    s2i, i2s = initvocab(SYMBOLS)
    a2i, i2a = initvocab(ACTIONS)
    w = initweights(
        o[:atype],o[:units],length(s2i),length(a2i),o[:dist])
    opts = initopts(w)

    for c = o[:start]:o[:step]:o[:end]
        seqlen = div(c, complexities[o[:task]])
        val = map(xi->data_generator(seqlen), [1:o[:nvalid]...])
        iter = 1
        lossval = 0

        while true
        # for iter = 1:500 # for debug
            trn = map(xi->data_generator(seqlen), 1:o[:batchsize])
            x = map(xi->xi[1], trn)
            y = map(xi->xi[2], trn)
            actions = map(xi->xi[3], trn)
            game = Game(x,y,actions)
            T = length(game.symgold[1])

            inputs = make_inputs(game, s2i, a2i)
            outputs = make_outputs(game, s2i, a2i)
            timesteps = length(inputs)
            batchsize = o[:batchsize]
            h,c = initstates(o[:atype],o[:units],o[:batchsize],o[:controller])
            batchloss = train!(w,inputs,outputs,h,c)
            batchloss = batchloss / (batchsize * timesteps)

            if iter < 100
                lossval = (iter-1)*lossval + batchloss
                lossval = lossval / iter
            else
                lossval = 0.01 * batchloss + 0.99 * lossval
            end

            # perform the validation
            if iter % o[:period] == 0
                info("batchloss:$batchloss")
                acc = validate(w,s2i,i2s,a2i,i2a,val,o)
                info("(iter:$iter,acc:$acc)")
                if acc > 0.98
                    info("$c converged in $iter iteration")
                    Knet.gc(); gc()
                    break
                end
            end

            iter += 1
        end
    end
end

function train!(w,x,y,h,c,opts)
    values = []
    gloss = lossgradient(w,x,y,h,c; values=values)
    update!(w, gloss, opts)
    return values[1]
end

function validate(w,s2i,i2s,a2i,i2a,data,o)
    batches = map(i->data[i:i+o[:batchsize]-1], [1:o[:batchsize]:length(data)...])
    ncorrect = 0
    for batch in batches
        x = map(xi->xi[1], batch)
        y = map(xi->xi[2], batch)
        actions = map(xi->xi[3], batch)
        game = Game(x,y,actions)
        T = length(game.symgold[1])

        # seqlen = length(y[1])
        correctness = trues(length(batch))
        h,c = initstates(o[:atype],o[:units],o[:batchsize],o[:controller])
        for k = 1:T
            x1,x2 = make_input(game, s2i, a2i)
            y1,y2 = make_output(game, s2i, a2i)
            x1 = convert(o[:atype], x1)
            x2 = convert(o[:atype], x2)
            cout, h, c = propagate(w[:wcont],w[:bcont],vcat(x1,x2),h,c)
            y1pred = predict(w[:wsymb],w[:bsymb],cout)
            y2pred = predict(w[:wact],w[:bact],cout)

            # ypred = predict(w,input) # (K
            y1pred = convert(Array, y1pred)
            y1pred = mapslices(indmax,y1pred,1)
            y1pred = map(yi->i2s[yi], y1pred)

            y2pred = convert(Array, y2pred)
            y2pred = mapslices(indmax,y2pred,1)
            y2pred = map(yi->i2a[yi], y2pred)

            # check correctness
            for i = 1:length(y1pred)
                if y1pred[i] != y[i][k]
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

!isinteractive() && !isdefined(Core.Main, :load_only) && main(ARGS)
