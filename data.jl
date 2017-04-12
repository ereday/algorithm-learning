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


function addition_data(seqlen)
    low = parse("1"*"0"^(seqlen - 1))
    hi = parse("1"*"0"^seqlen) - 1
    n1 = rand(low:hi)
    n2 = rand(0:hi)
    data = (n1, n2)
    ygold = n1 + n2
    compound_move = Any[goldacts[:down], goldacts[:moveleft], goldacts[:up]]

    arrived = 1
    if arrived == seqlen
        actions = compound_move
        return (data, ygold, actions)
    end
    actions = []
    while true
        append!(actions, compound_move); arrived += 1;
        (arrived == seqlen) && break
        append!(actions, [goldacts[:moveleft]]); arrived +=1;
        (arrived == seqlen) && break
    end
    if seqlen % 2 == 1
        append!(actions, compound_move)
    else
        append!(actions, [goldacts[:moveleft]])
        append!(actions, [goldacts[:down]])
    end
    return (data, ygold, actions)
end

!isinteractive() && !isdefined(Core.Main, :load_only) && main(ARGS)
