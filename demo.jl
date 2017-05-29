using JSON,Images,ArgParse

const IMGH  = 90
const IMGW  = 60
const RIGHT = "r"
const LEFT  = "l"
const UP    = "u"
const DOWN  = "d"

function parsedoc(fname)
    inf = JSON.parsefile(fname)
    global taskname = inf["task"]     # "copy"
    global input    = inf["input"]    # " 3 4 2 1 "
    global actions  = inf["actions"]  # "r r r r asd"
    global output   = inf["output"]   # "3 4 2 1"
    global spos     = inf["startpos"] # [1,-1] or [3,4] for 2d 
end

function makegrids(input,output)
    #    name(x)=(x==-1?string("img/emptycell.png"):string("img/n",x,".png"))
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
            singlerow = hcat(singlerow,load(name(input[i][j])))
        end
        push!(ri,singlerow)
    end
    ri   = vcat(ri...)    
    outs = [zeros(similar(load(name(input[1][1])))) for i=1:length(output)]
    ro   = hcat(outs...)
    return ri,ro 
end 

function modifygrid(ipos,igrid,frame,counter)
   # println(ipos)
    if ipos[1] != -1
        xstart = (ipos[1]-1) * IMGH + 1
    else
        xstart = 1
    end
    ystart = (ipos[2]-1) * IMGW + 1
    igridtmp = convert(Array{Float32},rawview(channelview(igrid)))    
    igridtmp[1:3,xstart:(xstart+IMGH-1),ystart:(ystart+IMGW-1)] += frame
    # small correction
#    igridtmp[2:3,xstart:(xstart+IMGH-1),ystart:(ystart+IMGW-1)] = 0
    igridtmp[igridtmp .> 255] = 255
    igrid = colorview(RGBA,igridtmp./255)
    save(string("c",counter,".png"),igrid)
end

function run(igrid,ogrid,actions,spos,frame)
    oldi   = igrid
    oldo   = ogrid
    ipos   = spos
    opos   = (1,-1)
    counter = 0 
    modifygrid(ipos,igrid,frame,counter)
    for action in actions 
        counter +=1
        igrid = oldi
        ipos = move(action,ipos,size(oldi,1),size(oldi,2))
        modifygrid(ipos,igrid,frame,counter)
    end    
end

function main()
    opts = parse_commandline()
    parsedoc(opts[:jsonfile])
    igrid,ogrid = makegrids(input,output)
    frame = makeframe()

    #println(spos)
    run(igrid,ogrid,actions,spos,frame)
    
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
        ("--outpath";default="pathtoasdad")
    end 
    return parse_args(s;as_symbols=true)
end

main()
