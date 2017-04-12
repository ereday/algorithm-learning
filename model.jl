function feedforward(w,x)
    tanh(w[:wfeed] * x .+ w[:bfeed])
end

function loss(w,x,ygold)
    x = feedforward(w,x)
    ypred = w[:wsoft]*x .+ w[:bsoft]
    return logprob(ygold, ypred)
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
