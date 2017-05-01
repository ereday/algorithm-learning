const SYMBOLS = (-2:9...)

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
