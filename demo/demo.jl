using JSON,Images,ArgParse

const IMGH  = 90
const IMGW  = 60
const RIGHT = "mr"
const LEFT  = "ml"
const UP    = "up"
const DOWN  = "down"
const NOOP  = -78
const EOF   = -2 # eof for reverse data 
const OUTDIR ="outdir/"
function parsedoc(fname)
    inf = JSON.parsefile(fname)
    global taskname = inf["task"]    
    global input    = inf["input"]   
    global actions  = inf["actions"] 
    global output   = inf["output"]  
    global spos     = inf["startpos"]
end

function makegrids(input,output)
    name(x)=string("img/n",x,".png")
    ri = Any[]
    if length(input) > 1
        # input side is 2d grid
        maxlen = maximum(map(y->length(y),input))
        for i = 1:length(input)
            if length(input[i]) != maxlen
                input[i] = [[10 for i=1:maxlen-length(input[i])]...,input[i]...]
            end
        end
    end    
    for i=1:length(input)
        singlerow = load(name(input[i][1]))
        for j=2:length(input[i])
            if input[i][j] == EOF;break; end # reverse task case 
            singlerow = hcat(singlerow,load(name(input[i][j])))
        end
        push!(ri,singlerow)
    end
    ri   = vcat(ri...)    
    outs = [zeros(similar(load(name(input[1][1])))) for i=1:length(output)]
    ro   = hcat(outs...)
    ro = ro .+ RGBA(1,1,1,1)
    return ri,ro 
end 

function makegridadd(input,output)
    maxlen = ndigits(maximum(input))
    minlen = ndigits(input[2])
    ri = Any[]
    push!(ri,reverse(digits(input[1])))
    push!(ri,[[10 for i=1:maxlen-minlen]...,reverse(digits(input[2]))...])
    
    return makegrids(ri,digits(input[1]+input[2]))
end 

# input grid 
function modifygrid(ipos,igrid,frame,counter)
    println("input grid modify",ipos)
    if ipos[1] != -1
        xstart = (ipos[1]-1) * IMGH + 1
    else
        xstart = 1
    end
    ystart = (ipos[2]-1) * IMGW + 1
    igridtmp = convert(Array{Float32},rawview(channelview(igrid)))
    if ystart > size(igridtmp,3)
        ystart = (ipos[2] -2) * IMGW + 1
    end
    igridtmp[1:3,xstart:(xstart+IMGH-1),ystart:(ystart+IMGW-1)] += frame
    # small correction
#    igridtmp[2:3,xstart:(xstart+IMGH-1),ystart:(ystart+IMGW-1)] = 0
    igridtmp[igridtmp .> 255] = 255
    igrid = colorview(RGBA,igridtmp./255)
    save(string(OUTDIR,"itape",counter,".png"),igrid)
end

# output grid 
function modifygrid(img::String,opos,ogrid,counter,frame)
    if img == "NOOP"
        if taskname == "reverse"
            noopimg =  ones(RGBA{N0f8}, 90, 300)
            save(string(OUTDIR,"otape",counter,".png"),noopimg)
            return noopimg,opos
        else
            noopimg = convert(Array{Float32},rawview(channelview(ogrid)))    
            zaa  = colorview(RGBA,noopimg./255)            
            save(string(OUTDIR,"otape",counter,".png"),zaa)
            return noopimg,opos
        end
    end        
    ystart = (opos[2]-1)*IMGW + 1
    ogridtmp = convert(Array{Float32},rawview(channelview(ogrid)))    
    outnum = convert(Array{Float32},rawview(channelview(load(img))))
    ogridtmp[:,:,ystart:(ystart+IMGW-1)] = outnum
    result = copy(ogridtmp)
    ogridtmp[1:3,:,ystart:(ystart+IMGW-1)] += frame 
    ogridtmp[ogridtmp .> 255] = 255
    ogrid = colorview(RGBA,ogridtmp./255)
    save(string(OUTDIR,"otape",counter,".png"),ogrid)
    if taskname == "add"
        return result,[opos[1],opos[2]-1]
    else
        return result,[opos[1],opos[2]+1]
    end
end

function run(igrid,ogrid,actions,outputs,spos,frame)
    name(x)=(x == NOOP ? "NOOP":string("img/n",x,".png"))
    oldi   = igrid
    oldo   = ogrid
    ipos   = spos
    if taskname in ["copy","reverse"]
        opos = (-1,1)
    else
        opos = (-1,length(outputs) - length(find(x->x==NOOP,outputs)))
    end
    counter = 10 
    modifygrid(ipos,igrid,frame,counter)
    for (index,action) in enumerate(actions)       
        # modify output tape        
        ogrid,opos  = modifygrid(name(outputs[index]),opos,ogrid,counter,frame)
        counter +=1
        igrid = oldi        
        ipos = move(action,ipos,Int(size(oldi,1)/IMGH),Int(size(oldi,2)/IMGW))
        modifygrid(ipos,igrid,frame,counter)   
    end    
end

function main()
    opts = parse_commandline()
    parsedoc(opts[:jsonfile])
    println("actions:",actions)
    println("input:",input)
    println("output:",output)
    println("spos:",spos)
    println("taskname:",taskname)
    if taskname == "add"
        igrid,ogrid = makegridadd(input,output)
    else
        igrid,ogrid = makegrids(input,output)
    end

    frame = makeframe()
    # spos     -> copy,reverse (-1,1)
    # addition/mul -> (-1,k)
    run(igrid,ogrid,actions,output,spos,frame)    
end

function makeframe()
    fr = zeros(Float32,3,IMGH,IMGW)
    fr[1,1:3,:]   = 255
    fr[1,IMGH-2:IMGH,:] = 255
    fr[1,:,1:3]   = 255
    fr[1,:,IMGW-2:IMGW] = 255
    return fr
end

# rn : rownumber, cn:columnnumber
function move(a,p,rn,cn)
    if a == RIGHT
        p  = [p[1],min(p[2]+1,cn)]
    elseif a == LEFT
        p = [p[1],max(p[2]-1,1)]
    elseif a == UP
        p = [max(p[1]-1,1),p[2]]
    elseif a == DOWN
        p = [min(p[1]+1,rn),p[2]]
    else
        println(" asdada")
    end
    return p        
end

function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table s begin
        ("--jsonfile";default="testinput1d.json")
        ("--outpath";default="outdir/copy/")
    end 
    return parse_args(s;as_symbols=true)
end

main()
