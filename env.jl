type Game
    InputTapes
    OutputTapes
    actions
    task
    pointers
    symgold
    timestep
end

0..9
noop
reverse
arrows 4

function init_game(x,y,actions,task)
    ninstances = length(x)
    input_tapes = x
    output_tapes = map(Any[], 1:ninstances)
    lens = map(xi->size(xi,2), x)
    pointers = init_pointers(lens,ninstances,task)
    symgold = y
    timestep = 1
    Game(input_tapes,output_tapes,actions,task,pointers,symgold,timestep)
end

function init_pointers(lens, ninstances,task) # assume task is copy
    map([1,1], 1:ninstances)
end
