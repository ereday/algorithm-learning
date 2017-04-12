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
                                             :moveleft=>"ml",
                                             :up=>"up",
                                              :down=>"down")
const no_op = -1

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



function reverse_data(seqlen)
    data = Any[ rand(0:9) for i=1:seqlen ]
    ygold = Any[ no_op for i=1:seqlen+1 ]
    append!(ygold, reverse(data))
    push!(data, "r")
    actions = [ goldacts[:moveright] for i=1:seqlen ]
    actions2 = [ goldacts[:moveleft] for i=1:seqlen ]
    acts = append!(actions, actions2)
    return (data, ygold, actions)
end

!isinteractive() && !isdefined(Core.Main, :load_only) && main(ARGS)
