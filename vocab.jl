const COPY_SYMBOLS = (-1:9...)
const REVERSE_SYMBOLS = (-2:9...)
const WALK_SYMBOLS = (-4:9...)
const SYMBOLS = (-1:9...)

function get_symbols(task)
    if in(task,("copy","reverse","walk"))
        return eval(parse(uppercase(string(task,"_symbols"))))
    end
    return SYMBOLS
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
