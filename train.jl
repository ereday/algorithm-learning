using Knet
using ArgParse

include("env.jl")
include("model.jl")
include("data_generator.jl")
include("vocab.jl")

function main(args)
    s = ArgParseSettings()
    s.description = ""

    @add_arg_table s begin
        # load/save files
        ("--datafile"; required=true)
        ("--task"; default="copy")
        ("--seed"; default=-1; help="random seed")
        ("--lr"; default=0.001)
        ("--units"; default=200)
        ("--discount"; default=0.95)
        ("--start"; default=6; arg_type=Int64)
        ("--end"; default=1000; arg_type=Int64)
        ("--step"; default=4; arg_type=Int64)
        ("--batchsize"; default=20; arg_type=Int64)
        ("--nvalid"; default=50; arg_type=Int64)
        ("--atype"; default=(gpu()>=0 ? "KnetArray{Float32}" : "Array{Float32}"))
        # ("--nsymbols"; default=11; arg_type=Int64)
    end

    isa(args, AbstractString) && (args=split(args))
    o = parse_args(args, s; as_symbols=true); display(o); flush(STDOUT)
    s = o[:seed] > 0 ? srand(o[:seed]) : srand()
    atype = eval(parse(o[:atype]))
    data_generator = get_data_generator(o[:task])

    # init model, params etc.
    s2i, i2s = initvocab(symbols)
    w = initweights(o[:atype],o[:units],length(symbols))
    opts = initopts(w)

    for c = o[:start]:o[:step]:o[:end]
        # on the fly data generation
        seqlen = c / complexities[o[:task]]
        val = map(data_generator(seqlen), 1:o[:nvalid])

        while true
            trn = map(data_generator(seqlen), 1:o[:batchsize])
            x = map(xi->xi[1], trn)
            y = map(xi->xi[2], trn)
            actions = map(xi->xi[3], trn)
            game = init_game(x,y,actions,o[:task])

            for k = 1:seqlen
                input = make_input(game)
                output = make_output(game)
                train!(w,input,output,opts)
                move_timestep!(game)
            end

            if acc == 1.0
                break
            end
        end
    end
end

function train!(w,x,y,opts)
    gloss = lossgradient(w,x,y)
    update!(w, gloss, opts)
end

!isinteractive() && !isdefined(Core.Main, :load_only) && main(ARGS)
