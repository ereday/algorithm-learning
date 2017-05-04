const ACTIONS = ("mr","ml","up","down", "<s>")
# <s> token stands for start in input, stop in output

type Game
    ninstances
    InputTapes
    OutputTapes
    next_actions
    prev_actions
    task
    pointers
    symgold
    timestep
    # mask

    function Game(x,y,actions,task="copy")
        N = length(x) # ninstances

        # input tapes
        xtapes = []
        if in(task, ("copy","reverse"))
            for xi0 in x
                xi1 = convert(Array{Int64}, xi0)
                xi2 = reshape(xi1, 1, length(xi1))
                push!(xtapes, xi2)
            end
        elseif in(task, ("add","mul","radd"))
            for xi in x
                push!(xtapes, make_grid(xi))
            end
        elseif task == "walk"
            error("walk is not implemented yet")
        end
        # output tapes

        ytapes = map(i->Any[], 1:N)

        # actions
        xactions = map(ai->["<s>", ai...], actions)
        yactions = map(ai->[ai..., "<s>"], actions)

        # pointer <=> head
        lens = map(xi->size(xi,2), x)
        pointers = init_pointers(xtapes,N,task)
        symgold = [y...,-1]
        timestep = 1

        new(
            N,xtapes,ytapes,yactions,xactions,
            task,pointers,symgold,timestep)
    end
end

function init_pointers(grids,ninstances,task)
    map(gi->get_origin(gi,task), grids)
end

function get_origin(grid,task)
    if in(task, ("copy","reverse"))
        return [1,1]
    elseif in(task, ("add","mul","radd"))
        return [1,size(grid,2)]
    elseif task == "walk"
        error("walk is not implemented yet")
    end
    error("invalid task")
end

# now only just for copy and reverse tasks
function move_timestep!(g::Game, actions)
    for k = 1:g.ninstances
        action = actions[k]
        if action == "mr"
            g.pointers[k][2] += 1
        elseif action == "ml"
            g.pointers[k][2] -= 1
        elseif action == "<s>"
            # do nothing
        elseif action == "up"
            g.pointers[k][1] -= 1
        elseif action == "down"
            g.pointers[k][1] += 1
        else
            error("zaa xd $action")
        end
    end
    g.timestep += 1
end

function move_timestep!(g::Game)
    actions = map(ai->ai[g.timestep], g.next_actions)
    move_timestep!(g,actions)
end

function make_input(g::Game, s2i, a2i)
    x1  = zeros(Float32, length(s2i), g.ninstances)

    # x1 => onehots, x11 => values, x12 => decoded (actions)
    x11 = map(i->g.InputTapes[i][g.pointers[i]...], 1:g.ninstances)
    x12 = map(v->s2i[v], x11)
    for k = 1:length(x12); x1[x12[k],k] = 1; end

    # x2 => onehots, x21 => values, x22 => decoded (actions)
    x2  = zeros(Float32, length(a2i), g.ninstances)
    x21 = map(i->g.prev_actions[i][g.timestep], 1:g.ninstances)
    x22 = map(v->a2i[v], x21)
    for k = 1:length(x22); x2[x22[k],k] = 1; end

    return x1,x2
end

function make_inputs(g::Game, s2i, a2i)
    reset!(g)
    inputs = []
    for k = 1:length(g.prev_actions)
        push!(inputs, make_input(g,s2i,a2i))
        move_timestep!(g)
    end
    reset!(g)
    return inputs
end

function make_output(g::Game, s2i, a2i)
    y10 = map(i->g.symgold[i][g.timestep], 1:g.ninstances)
    y11 = map(yi->s2i[yi], y10)

    y20 = map(i->g.next_actions[i][g.timestep], 1:g.ninstances)
    y21 = map(yi->a2i[yi], y20)

    return y11, y21
end

function make_outputs(g, s2i, a2i)
    reset!(g)
    outputs = []
    for k = 1:length(g.next_actions)
        push!(outputs, make_output(g,s2i,a2i))
        move_timestep!(g)
    end
    reset!(g)
    return outputs
end

function make_grid(x)
    x = collect(x)
    x = map(string, x)
    x = sort(x, by=length, rev=true)
    longest = length(x[1]); nelements = length(x)
    grid = zeros(Int64, nelements, longest)
    for (i,xi) in enumerate(x)
        for k = 1:length(xi)
            grid[i,end-length(xi)+k] = parse(Int64, xi[k])
        end
    end
    return grid
end

function reset!(g::Game)
    g.timestep = 1
    g.pointers = init_pointers(g.InputTapes,g.ninstances,g.task)
end
