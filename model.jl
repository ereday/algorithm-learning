function feedforward(w,x)
    relu(w[:wfeed] * x .+ w[:bfeed])
end

propagate(w,x) = feedforward(w,x)
predict(w,b,x) = w * x .+ b

# x1,y1 => input/output for symbols
# x2,y2 => input/output for actions
# weighted loss for soft symbol/action distributions
function loss(w,x1,y1,x2,y2; values=[])
    batchsize = size(x1,2)

    # controller
    cout = propagate(w,vcat(x1,x2))

    # predictions
    y1pred = predict(w[:wsymb],w[:bsymb],cout)
    y2pred = predict(w[:wact],w[:bact],cout)

    # log probabilities
    lossval1 = logprob(y1,y1pred)
    lossval2 = logprob(y2,y2pred)

    # combined loss
    lossval = 0.5*(lossval1+lossval2)
    push!(values, AutoGrad.getval(-lossval))
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

function initweights(atype,units,nsymbols,nactions,dist="randn", winit=0.01)
    w = Dict()
    winit = (dist=="xavier") ? 1.0 : winit
    _dist = eval(Symbol(dist))

    w[:wfeed] = _dist(Float32, units, nsymbols+nactions)
    w[:bfeed] = zeros(Float32, units, 1)
    w[:wsymb] = _dist(Float32, nsymbols, units)
    w[:bsymb] = zeros(Float32, nsymbols, 1)
    w[:wact]  = _dist(Float32, nactions, units)
    w[:bact]  = zeros(Float32, nactions, 1)

    for (k,v) in w
        w[k] = convert(atype, v*winit)
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
