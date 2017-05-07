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

# walk data
const LEFT = -2
const DOWN = -3
const UP = -4
const ARROWS = Dict{Int64,Symbol}(LEFT => :moveleft, DOWN => :down, UP => :up)

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
    data = Any[rand(0:9) for i=1:seqlen]
    actions = [goldacts[:moveright] for i =1:seqlen]
    ygold = data
    return (data, ygold, actions) # x,y,actions
end


function reverse_data(seqlen)
    data = Any[ rand(0:9) for i=1:seqlen ]
    ygold = reverse(data)
    push!(data, LEFT)
    actions = [goldacts[:moveright] for i=1:seqlen]
    actions2 = [goldacts[:moveleft] for i=1:seqlen+1]
    acts = append!(actions, actions2)
    return (data, ygold, actions)
end


function add_data(seqlen)
    low = parse("1"*"0"^(seqlen - 1))
    hi = parse("1"*"0"^seqlen) - 1
    n1 = rand(low:hi)
    num2len = rand(1:seqlen)
    num2 = parse("1"*"0"^num2len)-1
    n2 = rand(1:num2)
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

function walk_data(seqlen)
    direction = rand(-4:-2) # down, up, left
    width = Int(seqlen/2)
    grid = rand(0:9, width, width)

    mainrow = hcat(rand(0:9, 1, width-1), [direction])
    rowind = -1
    if direction == UP
        rowind = width
    elseif direction == DOWN
        rowind = 1
    else
        rowind = rand(1:width)
    end
    grid[rowind,:] = mainrow
    actions = vcat(
        map(x -> goldacts[:moveright], 1:width-1),
        map(x -> goldacts[ARROWS[direction]], 1:width))

    ygold = nothing
    if direction == UP
        ygold = grid[1:end-1,end]
    elseif direction == DOWN
        ygold = grid[2:end,end]
    else
        ygold = grid[rowind,1:end-1]
    end
    ygold = reverse(ygold)

    return (grid,ygold,actions)
end

function mul_data(seqlen)
    hi = parse("1"*"0"^seqlen) - 1
    low = parse("1"*"0"^(seqlen - 1))
    n = rand(low:hi)
    digit = rand(0:9)
    data = (digit, n)
    ygold = n * digit
    actions = append!([goldacts[:down]], [ goldacts[:moveleft] for i=1:seqlen+1])

    return (data, ygold, actions)
end


function radd_data(seqlen)
    hi = parse("1"*"0"^seqlen) - 1
    low = parse("1"*"0"^(seqlen - 1))
    n = rand(low:hi)
    n2 = rand(1:hi)
    n3 = rand(1:hi)

    data = (n, n2, n3)
    ygold = n + n2 + n3
    compound_action = Any[ goldacts[:down], goldacts[:down], goldacts[:moveleft], goldacts[:up], goldacts[:up]]


    arrived = 1
    if arrived == seqlen
        actions = [ goldacts[:down], goldacts[:down], goldacts[:moveleft], goldacts[:up]]
        return (data, ygold, actions)
    end
    if seqlen == 2
        actions = append!(compound_action, [ goldacts[:moveleft], goldacts[:down]])
        return (data, ygold, actions)
    end

    actions = []
    while true
        append!(actions, compound_action); arrived +=1;
        (arrived == seqlen-1) && break
        append!(actions, [goldacts[:moveleft]]); arrived +=1;
        (arrived == seqlen-1) && break
    end

    fs(num) = length(digits(num))
    second_big = ( fs(n2) > fs(n3) ? fs(n2) : fs(n3) )

    if seqlen % 2 == 1
        append!(actions, [ goldacts[:moveleft] ])
        if second_big == seqlen
            append!(actions, [ goldacts[:down] ])
            if fs(n2) == fs(n3)
                append!(actions, [ goldacts[:down], goldacts[:moveleft], goldacts[:up] ])
            else
                append!(actions, [ goldacts[:down], goldacts[:moveleft] ])
            end
        else
            append!(actions, [ goldacts[:down], goldacts[:down] ])
        end
    else
        append!(actions, compound_action)
        append!(actions, [ goldacts[:moveleft], goldacts[:down] ])
    end

    return (data, ygold, actions)
end
