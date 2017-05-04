# LSTM model - input * weight, concatenated weights
function lstm(weight, bias, input, hidden, cell)
    gates   = weight * vcat(hidden,input) .+ bias
    hsize   = size(hidden,1)
    forget  = sigm(gates[1:hsize,:])
    ingate  = sigm(gates[1+hsize:2hsize,:])
    outgate = sigm(gates[1+2hsize:3hsize,:])
    change  = tanh(gates[1+3hsize:end,:])
    cell    = cell .* forget + ingate .* change
    hidden  = outgate .* tanh(cell)
    return (hidden,cell)
end

function feedforward(w,b,x)
    relu(w * x .+ w), nothing
end

function propagate(w,b,x,h::Void,c::Void)
    return feedforward(w,b,x), nothing, nothing
end

function propagate(w,b,x,h,c)
    h,c = lstm(w,b,x,h,c)
    return h,h,c
end

predict(w,b,x) = w * x .+ b

# x1,y1 => input/output for symbols
# x2,y2 => input/output for actions
# weighted loss for soft symbol/action distributions
function loss(w,x,y,h,c; values=[])
    batchsize = size(x1,2)
    atype = AutoGrad.getval(typeof(w[:wcont]))

    lossval1 = lossval2 = 0
    for (xi,yi) in zip(x,y)
        # concat previous action and symbol from input tape
        input = convert(atype,vcat(xi...)) # TODO: CPU/GPU comparison

        # use the controller
        cout,h,c = propagate(w[:wcont],w[:bcont],input)

        # make predictions
        y1pred = predict(w[:wsymb],w[:bsymb],cout)
        y2pred = predict(w[:wact],w[:bact],cout)

        # log probabilities
        lossval1 += logprob(yi[1],y1pred)
        lossval2 += logprob(yi[2],y2pred)
    end

    # combined loss
    lossval = 0.5*(lossval1+lossval2)
    push!(values, AutoGrad.getval(-lossval))
    return -lossval/(batchsize*length(x))
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

function initweights(
    atype,units,nsymbols,nactions,controller="feed",dist="randn", winit=0.01)
    unit_dict = Dict("feed"=>1,"lstm"=>4)
    w = Dict()
    winit = (dist=="xavier") ? 1.0 : winit
    _dist = eval(Symbol(dist))

    units = units * get(unit_dict,controller,1)
    w[:wcont] = _dist(Float32, units, nsymbols+nactions)
    w[:bcont] = zeros(Float32, units, 1)
    w[:wsymb] = _dist(Float32, nsymbols, units)
    w[:bsymb] = zeros(Float32, nsymbols, 1)
    w[:wact]  = _dist(Float32, nactions, units)
    w[:bact]  = zeros(Float32, nactions, 1)

    for (k,v) in w
        w[k] = convert(atype, v*winit)
    end

    return w
end

function initstates(atype, hidden, batchsize, controller="lstm")
    if controller == "lstm"
        convert(atype, zeros(hidden, batchsize)),
        convert(atype, zeros(hidden, batchsize))
    elseif controller == "feedforward"
        nothing, nothing
    end
end

function initopts(w,lr=0.001,gclip=5.0)
    opts = Dict()
    for k in keys(w)
        opts[k] = Adam(;lr=lr,gclip=gclip)
    end
    return opts
end
