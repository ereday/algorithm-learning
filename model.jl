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
    # FIXME: this is so dirty
    if mask != nothing && length(mask) != 1
        index = index[[mask...]]
    elseif mask != nothing && length(mask) == 1 && !mask[1]
        return 0
    end
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
# ts => temporal difference learning targets
function rloss(w, ts, xs, ys, as, ms, h, c; values=[])
    # propagate controller, same with previous
    cout, h, c = propagate(w[:wcont], w[:bcont], xs, h, c)

    # symbol prediction, same
    sympred = predict(w[:wsymb], w[:bsymb], cout)

    # compute Q estimate
    qsa = predict(w[:wact], w[:bact], cout)
    nrows, ncols = size(qsa)
    index = as + nrows*(0:(length(as)-1))
    qs = qsa[index]  # divide by nsteps remaining
    estimate = reshape(qs, 1, length(qs))
    ts = reshape(ts, 1, length(ts))

    # hybrid loss calculation, supervised (symbols), q-learning (actions)
    val = 0
    val -= 0.5 * logprob(ys, sympred, ms)
    val += 0.5 * sumabs2(ts-estimate)

    push!(values, val, size(ts,2))
    return val
end

rlgrad = grad(rloss)

# FIXME: this is so dirty and inefficient
function make_batches(w,histories,s2i,a2i,discount,nsteps,batchsize; o=Dict())
    atype = get(o, :atype, typeof(w[:wcont]))

    samples = []
    for history in histories
        for k = 1:length(history)-1
            this = history[k]

            # input formation
            x = (this.input_symbol,this.prev_action)
            y = this.output_symbol
            a = this.action
            m = y != NO_SYMBOL && !this.is_done
            ph = this.h
            pc = this.c
            vs = this.nsteps

            # target formation
            T = min(k+nsteps, length(history))
            rs = reduce(+, [0, map(hi->hi.reward, history[k+1:T])...])
            yT = history[T].output_symbol
            target = rs
            if yT != NO_SYMBOL && !history[T].is_done
                # compute target
                xT = (history[T].input_symbol, history[T].prev_action)
                input = zeros(Cuchar, length(s2i)+length(a2i), 1)
                input[s2i[xT[1]]] = 1
                input[length(s2i)+a2i[xT[2]]] = 1
                input = convert(atype, input)

                vT = history[T].nsteps
                phT,pcT = history[T].h, history[T].c
                cout, hT, cT = propagate(w[:wcont],w[:bcont],input,phT,pcT)
                qsa = predict(w[:wact], w[:bact], cout)
                qs = maximum(qsa)
                target += vT * maximum(qs)
            end

            # normalize target
            target = target/vs

            sample = (target,x,y,a,m,ph,pc)
            push!(samples, sample)
        end

        # episode ending
        length(history) >= 1 || continue
        target = history[end].reward / history[end].nsteps
        x = (history[end].input_symbol, history[end].prev_action)
        y = history[end].output_symbol
        a = history[end].action
        ph = history[end].h
        pc = history[end].c
        m = y != NO_SYMBOL && !history[end].is_done
        sample = (target,x,y,a,m,ph,pc)
        push!(samples,sample)
    end

    batches = []
    for k = 1:batchsize:length(samples)
        from = k; to = min(from+k-1,length(samples))
        bsamples = samples[from:to]

        # make target batch
        ts = mapreduce(s->s[1], vcat, bsamples)

        # make input batch
        xs = falses(length(s2i)+length(a2i), to-from+1)
        for j = 1:to-from+1
            xs[s2i[bsamples[j][2][1]],j] = 1
            xs[length(s2i)+a2i[bsamples[j][2][2]]] = 1
        end

        # make output batch
        ys = map(si->s2i[si[3]], bsamples)

        # make action batch
        as = map(si->a2i[si[4]], bsamples)

        # make mask batch
        ms = map(si->si[5], bsamples)

        # make h,c batches
        hs = cs = nothing
        if bsamples[1][end-1] != nothing
            hs = mapreduce(bi->bi[end-1], hcat, bsamples)
            cs = mapreduce(bi->bi[end], hcat, bsamples)
        end

        batch = (ts,xs,ys,as,ms,hs,cs)
        push!(batches, batch)
    end

    return batches
end

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
    sa = map(s->(s[end].input_symbol, s[end].prev_action), samples)
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
