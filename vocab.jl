const COPY_SYMBOLS = (-1:9...)
const REVERSE_SYMBOLS = (-2:9...)
const WALK_SYMBOLS = (-4:9...)
const SYMBOLS = (0:9...)
const NO_SYMBOL = -1
const WRITE = "write"
const NOT_WRITE = "not-write"

const TAPE_ACTIONS = ("mr","ml")
const GRID_ACTIONS = ("mr","ml")
const WRITE_ACTIONS = (WRITE, NOT_WRITE)
const STOP_ACTION = ("<s>", NOT_WRITE)

function get_symbols(task)
    if in(task,("copy","reverse","walk"))
        return eval(parse(uppercase(string(task,"_symbols"))))
    end
    return SYMBOLS
end

function get_actions(task)
    actions = nothing
    if in(task,("copy","reverse"))
        actions = TAPE_ACTIONS
    else
        actions = GRID_ACTIONS
    end
    actions = mapreduce(wa->map(ai->(ai,wa), actions), vcat, WRITE_ACTIONS)
    return [actions..., STOP_ACTION]
end

function initvocab(symbols)
    symbols = collect(symbols)
    sort!(symbols)
    s2i, i2s = Dict(), Dict()
    c = 1
    for sym in symbols
        s2i[sym] = c
        i2s[c] = sym
        c += 1
    end
    return s2i, i2s
end
