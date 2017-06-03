import Base: push!
import Base: length
import Base: empty!
import Base: pop!
import Base: getindex

type Game
    input_tape
    output_tape
    gold_tape
    task
    head
    timestep
    prev_actions
    done

    function Game(x,y,actions,task="copy")
        # make input tpae
        input_tape = nothing
        if in(task, ("copy","reverse"))
            x = convert(Array{Int64}, x)
            input_tape = reshape(x, 1, length(x))
        elseif in(task, ("add","mul","radd"))
            input_tape = make_grid(x)
        elseif task == "walk"
            input_tape = x
        end

        # make output tape
        output_tape = Int64[]

        # make gold tape
        gold_tape = nothing
        if in(task, ("copy","reverse","walk"))
            gold_tape = y
        else
            gold_tape = digits(y)
        end

        # previous action
        prev_actions = ["<s>"]

        # head
        head = init_head(input_tape,task)
        timestep = 1

        new(input_tape, output_tape, gold_tape,
            task, head, timestep, prev_actions, done)
    end
end

function init_head(grid,task)
    return get_origin(grid,task)
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

function move_timestep!(g::Game, symbol, action)
    symbol != NO_SYMBOL && write!(g, symbol)
    move!(g, action)
    g.timestep += 1
    push!(g.prev_actions, action)
end

function move!(g::Game, action)
    if action == "mr"
        g.head[2] += 1
    elseif action == "ml"
        g.head[2] -= 1
    elseif action == "<s>"
        g.done = true
    elseif action == "up"
        g.head[1] -= 1
    elseif action == "down"
        g.head[1] += 1
    else
        error("invalid action: $action")
    end
end

function write!(g::Game, symbol)
    (g.task in ("copy","reverse","walk")?push!:unshift!)(g.gold_tape, symbol)
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
    g.prev_actions = ["<s>"]
    g.head = init_head(g.input_tape, g.task)
    g.timestep = 1
end

function read_symbol(grid, head)
    if 0 < head[1] <= size(grid,1) && 0 < head[2] <= size(grid,2)
        return grid[head...]
    end
    return -1
end

# transitions
type Transition
    reward # +1 for true symbol output, 0 for otherwise
    input # controller input (read_symbol+prev_action)
    nsteps # remaining steps
    h; c # controller states
    action # action has been taken
    symbol # symbol written to output tape
    done # last step or not
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

# deprecated old input functions, let me keep them for a while

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
