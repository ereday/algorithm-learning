const symbols = [0:9..., :noop]

function initvocab(symbols)
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
