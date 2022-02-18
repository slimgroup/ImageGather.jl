module ImageGather

    using JUDI, DSP, PyCall

    const impl = PyNULL()

    function __init__()
        pushfirst!(PyVector(pyimport("sys")."path"),dirname(pathof(ImageGather)))
        copy!(impl, pyimport("implementation"))
    end
    # Utility functions
    include("utils.jl")
    # Surface offset gathers
    include("surface_gather.jl")
    # Subsurface offset gathers
    include("subsurface_gather.jl")

end # module
