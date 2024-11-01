module ImageGather

    using JUDI
    using JUDI.DSP, JUDI.PythonCall

    import Base: getindex, *
    import JUDI: judiAbstractJacobian, judiMultiSourceVector, judiComposedPropagator, judiJacobian, make_input, propagate
    import JUDI.LinearAlgebra: adjoint

    const impl = PythonCall.pynew()

    IGPath = pathof(ImageGather)

    function __init__()
        pyimport("sys").path.append(dirname(IGPath))
        PythonCall.pycopy!(impl, pyimport("implementation"))
        set_devito_config("autopadding", false)
    end
    # Utility functions
    include("utils.jl")
    # Surface offset gathers
    include("surface_gather.jl")
    # Subsurface offset gathers
    include("subsurface_gather.jl")

end # module
