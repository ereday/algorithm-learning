using Knet
using ArgParse

include("env.jl")

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
    end

    isa(args, AbstractString) && (args=split(args))
    o = parse_args(args, s; as_symbols=true); display(o); flush(STDOUT)
end

!isinteractive() && !isdefined(Core.Main, :load_only) && main(ARGS)
