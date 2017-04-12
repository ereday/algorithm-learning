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
        ("--seed"; default=-1; help="random seed")
        ("--lr"; default=0.001)
        ("--units"; default=200)
        ("--discount"; default=0.95)
        ("--start"; default=6; arg_type=Int64)
        ("--end"; default=1000; arg_type=Int64)
        ("--step"; default=4; arg_type=Int64)
        ("--batchsize"; default=20; arg_type=Int64)
        ("--nvalid"; default=60; arg_type=Int64)
        ("--atype"; default=(gpu()>=0 ? "KnetArray{Float32}" : "Array{Float32}"))
        ("--period"; default=50; arg_type=Int64)
        # ("--nsymbols"; default=11; arg_type=Int64)
    end

    isa(args, AbstractString) && (args=split(args))
    o = parse_args(args, s; as_symbols=true); display(o); println(); flush(STDOUT)
    s = o[:seed] > 0 ? srand(o[:seed]) : srand()
    o[:atype] = eval(parse(o[:atype]))
    data_generator = get_data_generator(o[:task])

    # init model, params etc.
    s2i, i2s = initvocab(symbols)
    w = initweights(o[:atype],o[:units],length(symbols))
    opts = initopts(w)

    for c = o[:start]:o[:step]:o[:end]
        seqlen = div(c, complexities[o[:task]])
        val = map(xi->data_generator(seqlen), [1:o[:nvalid]...])
        iter = 1
        lossval = 0

        while true
            trn = map(xi->data_generator(seqlen), 1:o[:batchsize])
            x = map(xi->xi[1], trn)
            y = map(xi->xi[2], trn)
            actions = map(xi->xi[3], trn)
            game = init_game(x,y,actions,o[:task])

            batchloss = 0
            for k = 1:seqlen
                input = make_input(game, s2i)
                output = make_output(game, s2i)
                input = convert(o[:atype], input)
                this_loss = train!(w,input,output,opts)
                batchloss += this_loss
                move_timestep!(game)
            end
            batchloss = batchloss / seqlen
            if iter < 100
                lossval = (iter-1)*lossval + batchloss
                lossval = lossval / iter
            else
                lossval = 0.01 * batchloss + 0.99 * lossval
            end

            # perform the validation
            if iter % o[:period] == 0
                info("batchloss:$batchloss")
                acc = validate(w,s2i,i2s,val,o)
                info("(iter:$iter,acc:$acc)")
                if acc > 0.98
                    info("$c converged in $iter iteration")
                    break
                end
            end

            iter += 1
        end
    end
end

function train!(w,x,y,opts)
    values = []
    gloss = lossgradient(w,x,y; values=values)
    update!(w, gloss, opts)
    return values[1]
end

function validate(w,s2i,i2s,data,o)
    batches = map(i->data[i:i+o[:batchsize]-1], [1:o[:batchsize]:length(data)...])
    ncorrect = 0
    for batch in batches
        x = map(xi->xi[1], batch)
        y = map(xi->xi[2], batch)
        actions = map(xi->xi[3], batch)
        game = init_game(x,y,actions,o[:task])
        seqlen = length(y[1])
        correctness = trues(length(batch))
        for k = 1:seqlen
            input = make_input(game, s2i)
            output = make_output(game, s2i)
            input = convert(o[:atype], input)
            ypred = predict(w,input) # (K,B)
            ypred = convert(Array, ypred)
            ypred = mapslices(indmax,ypred,1)
            ypred = map(y->i2s[y], ypred)
            for i = 1:length(ypred)
                if ypred[i] != y[i][k]
                    correctness[i] = false
                end
            end
        end
        ncorrect += sum(correctness)
    end
    return ncorrect / length(data)
end

!isinteractive() && !isdefined(Core.Main, :load_only) && main(ARGS)
