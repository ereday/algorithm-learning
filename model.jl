function feedforward(w,x)
    tanh(w[:wfeed] * x .+ w[:bfeed])
end

function predict(w,x)
    # x = w[:wemb] * x
    x = feedforward(w,x)
    ypred = w[:wsoft]*x .+ w[:bsoft]
    return ypred
end

function loss(w,x,ygold; values=[])
    batchsize = size(x,2)
    ypred = predict(w,x)
    lossval = logprob(ygold,ypred)
    push!(values, AutoGrad.getval(lossval))
    return -lossval/batchsize
end

lossgradient = grad(loss)

function logprob(output, ypred)
    nrows,ncols = size(ypred)
    index = output + nrows*(0:(length(output)-1))
    o1 = logp(ypred,1)
    o2 = o1[index]
    o3 = sum(o2)
    return o3
end

function initweights(atype,units,nsymbols,winit=0.01)
    w = Dict()
    w[:wfeed] = winit*randn(Float32, units, nsymbols)
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
