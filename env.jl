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
        else
            # do nothing for now
        end
        # output tapes

        ytapes = map(i->Any[], 1:N)

        # actions
        xactions = map(ai->["<s>", ai...], actions)
        yactions = map(ai->[ai..., "<s>"], actions)

        # pointer <=> head
        lens = map(xi->size(xi,2), x)
        pointers = init_pointers(lens,N,task)
        symgold = [y...,-1]
        timestep = 1

        new(
            N,xtapes,ytapes,yactions,xactions,
            task,pointers,symgold,timestep)
    end
end

function init_pointers(lens, ninstances,task) # assume task is copy or reverse
    map(i->[1,1], 1:ninstances)
end

# now only just for copy and reverse tasks
function move_timestep!(g,actions)
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

function make_input(g, s2i, a2i)
    x1  = zeros(Float32, length(s2i), g.ninstances)
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

function make_output(g, s2i, a2i)
    y10 = map(i->g.symgold[i][g.timestep], 1:g.ninstances)
    y11 = map(yi->s2i[yi], y10)

    y20 = map(i->g.next_actions[i][g.timestep], 1:g.ninstances)
    y21 = map(yi->a2i[yi], y20)

    return y11, y21
end
