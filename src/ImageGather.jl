module ImageGather

    using JUDI
    using JUDI.DSP, JUDI.PyCall

    import Base: getindex, *
    import JUDI: judiAbstractJacobian, judiMultiSourceVector, judiComposedPropagator, judiJacobian, make_input, propagate
    import JUDI.LinearAlgebra: adjoint

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
