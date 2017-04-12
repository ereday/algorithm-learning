function feedforward(w,b,x)
    w * x .+ b
end

function loss(w,x,ygold)
    x = feedforward(w[1],w[2],x)
    ypred = w[3]*x .+ w[4]
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
