using ArgParse, JLD

# data generation file
# action set:
# "mr" -> move right

const complexities = Dict{AbstractString, Int}("copy"=>1,
                                               "reverse"=>2,
                                               "walk"=>1,
                                               "add"=>2,
                                               "radd"=>3,
                                               "mul"=>1)
const goldacts = Dict{Symbol, AbstractString}(:moveright=>"mr",
                                             :moveleft=>"mr",
                                             :up=>"up",
                                              :down=>"down")


function main(args)
    s = ArgParseSettings()
    s.description = ""

    @add_arg_table s begin
        # load/save files
        ("--savefile"; required=true)
        ("--task"; default="copy")
        ("--seed"; default=-1; help="random seed"; arg_type=Int64)
        ("--ninstances"; default=50; arg_type=Int64)
        ("--complexity"; default=1000; arg_type=Int64)
    end

    isa(args, AbstractString) && (args=split(args))
    o = parse_args(args, s; as_symbols=true); display(o); flush(STDOUT)
    s = o[:seed] > 0 ? srand(o[:seed]) : srand()

    data_generator = get_data_generator(o[:task])
    instances = []
    for k = 1:o[:ninstances]
        push!(instances, data_generator(o[:complexity]/complexities[o[:task]]))
    end
    save(o[:savefile], "data", instances)
    println()
    println("Data saved to $(o[:savefile])")
end

"""
experiment is a string given as lowercase name
"""
function get_data_generator(experiment::AbstractString)
    !haskey(complexities, experiment) && error("wrong key type $experiment")

    comp = complexities[experiment]
    fname = string(experiment, "_data")
    f = eval(parse(fname))
    return f
end


function copy_data(seqlen)
    data = Any[ rand(0:9) for i=1:seqlen ]
    actions = [ goldacts[:moveright] for i =1:seqlen ] # to do decide the last item
    ygold = data
    return (data, ygold, actions) # x,y,actions
end


function reverse_data(seqlen::Int)
    data = Any[ rand(0:9) for i=1:seqlen ]
    ygold = reverse(data)
    push!(ygold, -1)
    push!(data, "r")
    actions = [ goldacts[:moveright] for i=1:seqlen ]
    return (data, ygold, actions)
end

!isinteractive() && !isdefined(Core.Main, :load_only) && main(ARGS)
