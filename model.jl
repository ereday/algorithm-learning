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

#w1 -> (2H, X+H) #b1   -> (2H,1)
#w2 -> (2H, X+H) #b2   -> (2H,1)
#reference: https://goo.gl/6uDnrk
function gru(w1,w2,b1,b2,input,hidden)
    gates  = w1 * vcat(hidden,input) .+ b1
    hsize  = size(hidden,1)
    update = sigm(gates[1:hsize,:])
    reset  = sigm(gates[1+hsize:end,:])
    h      = w2 * vcat(hidden .* reset,input) .+ b2
    hidden = (1 .- update) + (update .* hidden)
    return hidden
end

function feedforward(w,b,x)
    return relu(w * x .+ b)
end

function propagate(w,b,x,h::Void,c::Void)
    feedforward(w,b,x), nothing, nothing
end

function propagate(w,b,x,h,c)
    h,c = lstm(w,b,x,h,c)
    return h,h,c
end

function predict(w,b,x)
    return w * x .+ b
end

# x1,y1 => input/output for symbols
# x2,y2 => input/output for actions
# weighted loss for soft symbol/action distributions
function sloss(w,x,y,h,c; values=[])
    batchsize = size(x[1][1],2)
    atype = typeof(AutoGrad.getval(w[:wcont]))

    lossval1 = lossval2 = 0
    for (xi,yi) in zip(x,y)
        # concat previous action and symbol from input tape
        input = convert(atype,vcat(xi...)) # TODO: CPU/GPU comparison

        # use the controller
        cout,h,c = propagate(w[:wcont],w[:bcont],input,h,c)

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

slgradient = grad(sloss)

function initweights(
    atype,units,nsymbols,nactions,
    controller="feedforward",dist="randn", winit=0.01)
    unit_dict = Dict("feed"=>1,"lstm"=>4)
    w = Dict()
    winit = (dist=="xavier") ? 1.0 : winit
    _dist = eval(Symbol(dist))

    params = units * get(unit_dict,controller,1)
    input = nsymbols+nactions+(controller!="feedforward")*units
    w[:wcont] = _dist(Float32, params, input)
    w[:bcont] = zeros(Float32, params, 1)
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


# Reinforcement Learning stuff
# xs => controller inputs (concat prev_action and read_symbol)
# ys => controller symbol outputs written to output tape
# as => actions taken by following behaviour policy
# targets => temporal difference learning targets
# TODO: make it sequential?
function rloss(w, targets, xs, ys, as, h, c; values=[])
    # propagate controller, same with previous
    cout, h, c = propagate(w[:wcont], w[:bcont], xs, h, c)

    # symbol prediction, same
    sympred = predict(w[:wsymb], w[:bsymb], cout)

    # action estimation, same
    qsa = predict(w[:wact], w[:bact], cout)

    # compute indices
    nrows, ncols = size(qsa)
    index = actions + nrows*(0:(length(actions)-1))

    # compute estimate
    qs = qsa[index]
    estimate = reshape(qs, 1, length(qs))

    # hybrid loss calculation
    val = 0
    val += 0.5 * logprob(ys, sympred) # sl loss, output symbols
    val += 0.5 * sumabs2(targets-estimate) # rl loss, actions

    push!(values, val)
    return val
end

rlgradient = grad(rloss)

# compute TD targets for objective
function compute_targets(samples, wfix, discount)
    error("nothing yet")
end
