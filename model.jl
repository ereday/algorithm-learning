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

function logprob(output, ypred, mask=nothing)
    nrows,ncols = size(ypred)
    index = output + nrows*(0:(length(output)-1))
    index = mask == nothing ? index : index[mask]
    o1 = logp(ypred,1)
    # @show index
    o2 = o1[index]
    o3 = sum(o2)
    return o3
end

# loss function for supervised learning
# x: controller input, y: controller output (action+symbol)
# m: masks for loss, h/c: controller states
function sloss(w,x,y,m,h,c; values=[])
    batchsize = size(x[1][1],2)
    atype = typeof(AutoGrad.getval(w[:wcont]))

    lossval1 = lossval2 = 0
    for (i,(xi,yi,mi)) in enumerate(zip(x,y,m))
        # concat previous action and symbol from input tape
        input = convert(atype,xi) # TODO: CPU/GPU comparison

        # use the controller
        cout,h,c = propagate(w[:wcont],w[:bcont],input,h,c)

        # make predictions
        sympred = predict(w[:wsymb],w[:bsymb],cout)
        actpred = predict(w[:wact],w[:bact],cout)

        # log probabilities
        symgold, actgold = yi[1], yi[2]
        lossval1 += logprob(symgold,sympred,mi)
        lossval2 += logprob(actgold,actpred)
    end

    # combined loss
    lossval = -0.5*(lossval1+lossval2)
    push!(values, AutoGrad.getval(lossval))
    push!(values, batchsize*length(x))

    # return -lossval/(batchsize*length(x))
    return lossval
end

slgrad = grad(sloss)

function rloss()
end

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

function initopts(w,optim)
    opts = Dict()
    for k in keys(w)
        opts[k] = eval(parse(optim))
    end
    return opts
end

# Reinforcement Learning stuff
# xs => controller inputs (concat prev_action and read_symbol)
# ys => controller symbol outputs written to output tape
# as => actions taken by following behaviour policy
# targets => temporal difference learning targets
# TODO: make it sequential?
function rloss(w, targets, xs, ys, as, h, c, vs; values=[])
    # propagate controller, same with previous
    cout, h, c = propagate(w[:wcont], w[:bcont], xs, h, c)

    # symbol prediction, same
    sympred = predict(w[:wsymb], w[:bsymb], cout)

    # action estimation, same
    qsa = predict(w[:wact], w[:bact], cout)

    # compute indices
    nrows, ncols = size(qsa)
    index = as + nrows*(0:(length(as)-1))

    # compute estimate
    qs = qsa[index]  # divide by nsteps remaining
    estimate = reshape(qs, 1, length(qs))
    estimate = estimate ./ vs

    # hybrid loss calculation
    val = 0
    val += -0.5*logprob(ys, sympred) # sl loss, output symbols
    val += 0.5 * sumabs2(targets-estimate) # rl loss, actions

    push!(values, val)
    return val / size(targets,2)
end

rlgradient = grad(rloss)

# compute TD targets for objective
function compute_targets(samples, w, discount, nsteps, s2i, a2i)
    # reward calculations
    if discount < 0
        discount = 1
    end
    discounts = map(i->discount^i, 0:nsteps)
    targets = zeros(1, length(samples))
    for k = 1:length(samples)
        sample = samples[k]
        vs, vsp = samples[k][1].nsteps, samples[k][end].nsteps
        reward = mapreduce(
            i->(sample[i].reward)*discounts[i]/vs, +, 1:length(sample))
        targets[k] = reward
    end

    # (1) dynamic discount calculation for max Q(s,a)
    get_steps(s) = (s[1].nsteps, s[end].nsteps)
    gamma = discounts[end]
    discounts = map(i->get_steps(samples[i]), 1:length(samples))
    discounts = map(d->gamma*(d[2]/d[1]), discounts)

    # (2) predict Q(s,a) over all possible actions

    # (2.1) batch controller states
    h = c = nothing
    if samples[1][1].h != nothing
        h = mapreduce(s->s[end].h, hcat, samples)
    end
    if samples[1][1].c != nothing
        c = mapreduce(s->s[end].c, hcat, samples)
    end

    # (2.2) batch environment states - aka controller inputs
    sa = map(s->(s[end].input_symbol, s[end].input_action), samples)
    inputs = zeros(length(s2i)+length(a2i), length(samples))
    for k = 1:length(sa) # symbol-action pairs
        inputs[s2i[sa[k][1]],k] = 1
        inputs[length(s2i)+a2i[sa[k][2]]] = 1
    end

    # (2.3) convert and propagate
    # FIXME: propagate is redundant
    atype = typeof(w[:wcont])
    targets = convert(atype, targets)
    inputs = convert(atype, inputs)
    h = h == nothing ? h : convert(atype,h)
    c = c == nothing ? c : convert(atype,c)
    discounts = reshape(discounts, 1, length(discounts))
    discounts = convert(atype, discounts)
    cout, h, c = propagate(w[:wcont], w[:bcont], inputs, h, c)

    # (2.4) main part - compute Q(s,a') over all possible actions
    # then, find which action maximizes it and select its value
    qsa0 = predict(w[:wact],w[:bact],cout)
    qsa1 = maximum(qsa0,1)
    qsa2 = sum(qsa0 .* (qsa1.==qsa0), 1)
    qsa3 = reshape(qsa2, 1, length(qsa2))

    targets += discounts .* qsa3
    return targets
end

# function make_batch(
#     obj::ReplayMemory, w, discount, nsteps, s2i, a2i, batchsize)
#     samples = sample(obj, batchsize, nsteps)
#     targets = compute_targets(samples, w, discount, nsteps, s2i, a2i)
#     atype = typeof(targets)

#     # xs <-> inputs (read symbol+previous action) - onehots
#     xs = zeros(length(s2i)+length(a2i),length(samples))
#     for (i,sample) in enumerate(samples)
#         xs[s2i[sample[1].input_symbol],i] = 1
#         xs[length(s2i)+a2i[sample[1].input_action],i] = 1
#     end
#     xs = convert(atype, xs)

#     # ys <-> output symbols
#     # as <-> actions
#     ys = map(s->s[1].output_symbol, samples); ys = map(yi->s2i[yi],ys)
#     as = map(s->s[1].output_action, samples); as = map(ai->a2i[ai],as)

#     h = c = nothing
#     if samples[1][1].h != nothing
#         h = mapreduce(s->s[1].h, hcat, samples)
#         h = convert(atype, h)
#     end
#     if samples[1][1].c != nothing
#         c = mapreduce(s->s[1].c, hcat, samples)
#         c = convert(atype, c)
#     end
#     vs = map(si->si[1].nsteps, samples)
#     vs = reshape(vs, 1, length(vs))
#     vs = convert(atype, vs)

#     return targets, xs, ys, as, h, c, vs
# end
