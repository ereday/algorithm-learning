import Base: push!
import Base: length
import Base: empty!
import Base: pop!
import Base: getindex


# <s> token stands for start in input, stop in output

type Game
    ninstances
    input_tapes
    output_tapes
    gold_tapes
    next_actions
    prev_actions
    task
    pointers
    symgold
    timestep
    # terminated
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
            # error("walk is not implemented yet")
            for xi in x
                push!(xtapes, xi)
            end
        end

        # output tapes
        ytapes = map(i->Any[], 1:N)

        # gold tapes
        gtapes = []
        if in(task, ("copy","reverse","walk"))
            for yi in y
                push!(gtapes, yi)
            end
        else
            for yi in y
                push!(gtapes, digits(yi))
            end
        end

        # actions
        xactions = map(ai->["<s>", ai...], actions)
        yactions = map(ai->[ai..., "<s>"], actions)

        # pointer <=> head
        pointers = init_pointers(xtapes,N,task)
        symgold = map(i->get_symgold(xtapes[i],gtapes[i],actions[i],task), 1:N)
        timestep = 1
        # terminated = falses(N)

        new(
            N,xtapes,ytapes,gtapes,yactions,xactions,
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
        lastcol = grid[:,end]
        i = findfirst(x->x<0, lastcol)
        return [i,1]
    end
    error("invalid task: $task")
end

# now only just for copy and reverse tasks
function move_timestep!(g::Game, actions::Array)
    for k = 1:g.ninstances
        action = actions[k]
        move_timestep!(g, k, action)
    end
    g.timestep += 1
end

function move_timestep!(g::Game)
    actions = map(ai->ai[g.timestep], g.next_actions)
    move_timestep!(g,actions)
end

function move_timestep!(g::Game, instance::Int64, action)
    k = instance
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
        error("invalid action: $action")
    end
end

function make_input(g::Game, s2i, a2i)
    x1  = zeros(Float32, length(s2i), g.ninstances)

    # x1 => onehots, x11 => values, x12 => decoded (actions)
    x11 = map(i->read_symbol(g.input_tapes[i],g.pointers[i]), 1:g.ninstances)
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
    for k = 1:length(g.prev_actions[1])
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
    for k = 1:length(g.next_actions[1])
        push!(outputs, make_output(g,s2i,a2i))
        move_timestep!(g)
    end
    reset!(g)
    return outputs
end

function make_grid(x)
    x = collect(x)
    x = map(string, x)
    longest = length(x[1]); nelements = length(x)
    grid = -ones(Int64, nelements, longest)
    for (i,xi) in enumerate(x)
        for k = 1:length(xi)
            grid[i,end-length(xi)+k] = parse(Int64, xi[k])
        end
    end
    return grid
end

function reset!(g::Game)
    g.timestep = 1
    g.pointers = init_pointers(g.input_tapes,g.ninstances,g.task)
end

# x: input tape, y: output tape, a: actions
function get_symgold(x,y,a,task)
    if task == "copy"
        return [y..., -1]
    elseif task == "reverse"
        return [-1, map(yi->-1, y)..., y..., -1]
    elseif task == "walk"
        return [map(i->-1, 1:size(x,2))..., y..., -1]
    elseif task == "add"
        x1digits = size(x,2)
        y1digits = length(y)
        symgold = mapreduce(i->[-1,y[i]], vcat, 1:x1digits)
        return vcat(symgold, [-1, (y1digits>x1digits?y[end]:-1)])
    else
        error("$task is not implemented yet")
    end
end

function read_symbol(grid, pointer)
    if 0 < pointer[1] <= size(grid,1) && 0 < pointer[2] <= size(grid,2)
        return grid[pointer...]
    end
    return -1
end

# Environment for Reinforcement Learning
type Transition
    # +1 for true symbol output, 0 for otherwise
    reward

    # environment state - POMDP
    input_symbol
    input_action
    nsteps # remaining steps

    # controller state - e.g. RNN hidden/cell
    h
    c

    # next environment state
    output_symbol
    output_action
    done
end

type ReplayMemory
    capacity
    memory

    function ReplayMemory(capacity)
        memory = Transition[]
        new(capacity, memory)
    end
end

function push!(obj::ReplayMemory, t)
    push!(obj.memory, t)
    length(obj.memory) > obj.capacity && shift!(obj.memory)
end

function length(obj::ReplayMemory)
    return length(obj.memory)
end

function empty!(obj::ReplayMemory)
    empty!(obj.memory)
end

function pop!(obj::ReplayMemory)
    pop!(obj.memory)
end

function sample(obj::ReplayMemory, nsamples, nsteps)
    samples = []
    indices = randperm(length(obj))[1:min(nsamples,length(obj))]
    for ind in indices
        sample = []
        for k = ind:min(ind+nsteps-1,length(obj))
            step = obj.memory[k]
            push!(sample, step)
            step.done && break
        end
        push!(samples, sample)
    end
    return samples
end

# currently I don't have an efficient idea to run episodes parallel
# I just leave it in this way for simplicity
function run_episodes!(
    g::Game, mem, w, h, c, s2i, i2s, a2i, i2a, steps_done; o=Dict())
    reset!(g)
    atype = typeof(w[:wcont])

    for k = 1:g.ninstances
        input_action = g.prev_actions[k][1] # no action input
        input_symbol = read_symbol(g.input_tapes[k], g.pointers[k])
        predicted = []

        while true
            # make one-hot input vector
            input = zeros(length(s2i)+length(a2i), 1)
            input[s2i[input_symbol]] = 1
            input[length(s2i)+a2i[input_action]] = 1
            input = convert(atype, input)

            # use controller
            cout, h1, c1 = propagate(w[:wcont], w[:bcont], input, h, c)

            # # predict symbol
            # y1pred = predict(w[:wsymb],w[:bsymb], cout)
            # # y0 = maximum(y1pred,1)
            # # y1 = y1pred .- y0
            # # y2 = exp(y1)
            # # y3 = y2 ./ sum(y2)
            # # y1pred = Array(y3)
            # # y1pred = cumsum(y1pred) .> rand()
            # y1pred = indmax(y1pred)
            # y1pred = rand(1:length(y3))

            # predict symbol
            # symbol = take_action(w[:wsymb],w[:bsymb],cout,steps_done;o=o)
            symbol = take_action(w[:wsymb],w[:bsymb],cout,steps_done;o=o)
            symbol = i2s[symbol]
            push!(predicted, symbol)

            # take action
            action = take_action(w[:wact],w[:bact],cout,steps_done; o=o)
            action = i2a[action]
            steps_done += 1

            # decide reward, termination, remaining steps
            reward, done, nsteps = get_reward(g, k, predicted)

            # very stupid scenario, terminate episode and start a new one
            if action == "<s>" || done
                break
            end

            # transition
            this_transition = Transition(
                reward,
                input_symbol,
                input_action,
                nsteps,
                h != nothing ? Array(h) : nothing,
                c != nothing ? Array(c) : nothing,
                predicted[end], # output_symbol
                action,
                done)
            # @show this_transition

            # push to replay memory
            push!(mem, this_transition)

            # move head
            move_timestep!(g, k, action)

            # change controller state
            h = h1; c = c1

            # change inputs
            input_symbol = predicted[end]
            input_action = action

            # if done, then break
            done && break
        end

        # FIXME: when to do steps_done increament?
        # after each episode or after each step?
        # steps_done += 1
    end

    return steps_done
end

function take_action(w, b, s, steps_done; o=Dict())
    ei = get(o, :epsinit,  EPS_INIT)
    ef = get(o, :epsfinal, EPS_FINAL)
    ed = get(o, :epsdecay, EPS_DECAY)
    et = ef + (ei-ef) * exp(-steps_done/ed)

    if rand() > et
        # @show w,s
        s = reshape(s,length(s),1)
        y = predict(w,b,s)
        y = Array(y)
        return indmax(y)
    else
        return rand(1:size(w,1))
    end
end

function get_reward(g::Game, instance, predictions)
    # symgold = g.symgold[instance]
    # symgold = filter(si->si!=NO_SYMBOL, symgold)
    last_prediction = predictions[end]
    predictions = filter(pi->pi!=NO_SYMBOL, predictions)

    reward = 0
    done = false
    nsteps = length(g.gold_tapes[instance]) - length(predictions)
    desired = g.gold_tapes[instance]
    desired = in(g.task,("copy","reverse","walk")) ? reverse(desired) : desired
    if predictions == desired[1:length(predictions)] && last_prediction != NO_SYMBOL
        reward = 1
    else
        done = true
    end

    return reward, done, nsteps
end
