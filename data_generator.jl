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
