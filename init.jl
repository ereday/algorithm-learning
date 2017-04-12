function initweights(atype,units,nsymbols,winit=0.01)
    w = Dict()
    w[:wfeed] = winit*randn(Float32, units, nsymbols) # (nsymbols,bs)
    w[:bfeed] = zeros(Float32, units, 1)
    w[:wsoft] = winit*randn(Float32, nsymbols, units)
    w[:bsoft] = zeros(Float32, nsymbols, 1)

    for (k,v) in w
        w[k] = convert(atype, v)
    end

    return w
end

function initopts(w,lr=0.001,gclip=5.0)
    opts = Dict()
    for k in keys(w)
        opts[k] = Adam(;lr=lr,gclip=gclip)
    end
    return opts
end
