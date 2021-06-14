module ImageGather

    using JUDI, DSP, PyCall

    const dbr = PyNULL()

    function __init__()
        pushfirst!(PyVector(pyimport("sys")."path"),dirname(pathof(ImageGather)))
        copy!(dbr, pyimport("double_rtm"))
    end
    # Utility functions
    include("utils.jl")
    # Surface offset gathers
    include("surface_gather.jl")

end # module
