type Game
    ninstances
    InputTapes
    OutputTapes
    actions
    task
    pointers
    symgold
    timestep
    # mask
end

# 0..9
# noop
# reverse
# arrows 4

function init_game(x,y,actions,task)
    ninstances = length(x)
    input_tapes = x
    output_tapes = map(i->Any[], 1:ninstances)
    lens = map(xi->size(xi,2), x)
    pointers = init_pointers(lens,ninstances,task)
    symgold = y
    timestep = 1

    Game(ninstances,input_tapes,output_tapes,actions,task,pointers,symgold,timestep)
end

function init_pointers(lens, ninstances,task) # assume task is copy
    map(i->[1,1], 1:ninstances)
end

function move_timestep!(g)
    for k = 1:g.ninstances
        action = g.actions[k][g.timestep]
        if action == "mr"
            g.pointers[k][1] += 1
            
        elseif action == "ml"
            g.pointers[k][1] -= 1
        else
            error("zaa xd")
        end
    end
    g.timestep += 1
end

function make_input(g, s2i)
    values = map(i->g.InputTapes[i][g.pointers[i]...], 1:g.ninstances)
    decoded = map(v->s2i[v], values)
    input = zeros(length(s2i), g.ninstances)
    for k = 1:length(decoded)
        input[decoded[k],k] = 1
    end
    return input
end

function make_output(g, s2i)
    values = map(i->g.symgold[i][g.timestep], 1:g.ninstances)
    output = map(v->s2i[v], values)
    return output
end
