type Game
    input_tape
    output_tape
    gold_tape
    task
    head
    timestep
    prev_actions
    is_done

    function Game(x,y,task="copy")
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
        prev_actions = [STOP_ACTION]

        # head
        head = init_head(input_tape,task)
        timestep = 1

        new(input_tape, output_tape, gold_tape,
            task, head, timestep, prev_actions, false)
    end
end

function init_head(grid,task)
    return get_origin(grid,task)
end

function init_head(g::Game)
    return get_origin(g.input_tape,g.task)
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

function move_timestep!(g::Game, write_symbol, move_action)
    write_action = write_symbol == NO_SYMBOL ? NOT_WRITE : WRITE
    write_action == WRITE && write!(g, write_symbol)
    move!(g, move_action)
    g.timestep += 1
    push!(g.prev_actions, (move_action, write_action))
    g.is_done = is_done(g)
end

function move!(g::Game, action)
    if action == "mr"
        g.head[2] += 1
    elseif action == "ml"
        g.head[2] -= 1
    elseif action == "<s>"
        g.is_done = true
    elseif action == "up"
        g.head[1] -= 1
    elseif action == "down"
        g.head[1] += 1
    else
        error("invalid action: $action")
    end
end

function write!(g::Game, symbol)
    (g.task in ("copy","reverse","walk")?push!:unshift!)(g.output_tape, symbol)
end

# FIXME: when it is done? right thing is to check the last action is <s> or not
function is_done(g::Game)
    len = length(g.output_tape)
    len >= length(g.gold_tape) && return true
    gold = nothing
    if g.task in ("copy","reverse","walk")
        gold = g.gold_tape[1:len]
    else
        gold = g.gold_tape[end-length(len):end]
    end
    return !(g.output_tape == gold)
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
    g.prev_actions = [STOP_ACTION]
    g.head = init_head(g.input_tape, g.task)
    g.timestep = 1
    empty!(g.output_tape)
    g.is_done = false
end

function read_symbol(grid, head)
    if 0 < head[1] <= size(grid,1) && 0 < head[2] <= size(grid,2)
        return grid[head...]
    end
    return -1
end

function read_symbol(g::Game)
    return read_symbol(g.input_tape, g.head)
end

# transitions
type Transition
    reward # +1 for true symbol output, 0 for otherwise
    input_symbol # read symbol - controller input
    prev_action # previous action - controller input
    nsteps # remaining steps
    h; c # controller states
    action # action has been taken
    output_symbol # symbol written to output tape
    is_done # last step or not
end

function take_action(w, b, s, steps_done; o=Dict())
    ei = get(o, :epsinit,  EPS_INIT)
    ef = get(o, :epsfinal, EPS_FINAL)
    ed = get(o, :epsdecay, EPS_DECAY)
    et = ef + (ei-ef) * exp(-steps_done/ed)

    # I think they did not use GLIE, they just unroll
    if rand() > et
        # @show w,s
        s = ndims(s) == 1 ? reshape(s,length(s),1) : s
        y = predict(w,b,s)
        return mapslices(indmax, Array(y), 1)
    else
        return rand(1:size(w,1), size(s)...)
    end
end

function take_action(w,b,s; eps=0.05)
    if rand() > eps
        s = ndims(s) == 1 ? reshape(s, length(s), 1) : s
        y = predict(w,b,s)
        return mapslices(indmax, Array(y), 1)
    else
        return rand(1:size(w,1), size(s,2))
    end
end

function get_reward(g::Game)
    length(g.output_tape) > length(g.gold_tape) && return 0
    if in(g.task, ("copy","reverse","walk"))
        is_desired = g.output_tape == g.gold_tape[1:length(g.output_tape)]
    else
        is_desired = g.output_tape == g.gold_tape[end-length(g.output_tape):end]
    end
    return Int(is_desired && length(g.output_tape) != 0)
end

function get_remaining_steps(g::Game)
    return length(g.gold_tape) - length(g.output_tape)
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

function make_data(games,s2i,a2i,actions)
    all_done = false
    inputs, outputs, masks = [], [], []

    t = 1
    while !all_done
        input = zeros(Cuchar, length(s2i)+length(a2i), length(games))
        mask = falses(1, length(games))
        symgold = NO_SYMBOL*ones(Int64, length(games))
        actgold = length(a2i)*ones(Int64, length(games))

        for (i,game) in enumerate(games)
            # skip if game is finished
            game.is_done && continue

            input_symbol = read_symbol(game.input_tape, game.head)
            input[s2i[input_symbol],i] = 1
            input[length(s2i)+a2i[game.prev_actions[end]],i] = 1

            move_action, write_action = actions[i][game.timestep]
            y = NO_SYMBOL
            if write_action == WRITE
                mask[1,i] = 1
                if game.task in ("copy","reverse","walk")
                    y = game.gold_tape[length(game.output_tape)+1]
                else
                    y = game.gold_tape[end-length(game.output_tape)]
                end
                symgold[i] = s2i[y]
            end

            actgold[i] = a2i[(move_action,write_action)]
            move_timestep!(game,y,move_action)
        end

        push!(inputs, input)
        push!(outputs, (symgold,actgold))
        push!(masks, mask)

        all_done = true
        for k = 1:length(games)
            if !games[k].is_done
                all_done = false
            end
        end

        # @show t,all_done
        t+=1
    end

    for k = 1:length(games); reset!(games[k]); end
    return inputs,outputs,masks
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
