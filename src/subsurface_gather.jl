import JUDI: judiAbstractJacobian, judiJacobian, task_distributed

export judiExtendedJacobian

struct judiExtendedJacobian{DDT<:Number,RDT<:Number} <: judiAbstractJacobian{DDT,RDT}
    name::String
    m::Integer
    n::Integer
    info::Info
    model::Model
    source::judiVector
    recGeometry::Geometry
    offsets::Vector
    options::Options
    fop::Function              # forward
    fop_T::Union{Function, Nothing}  # transpose
end


function judiExtendedJacobian(F::judiPDEfull, source::judiVector, offsets::AbstractVector; DDT::DataType=Float32, RDT::DataType=DDT, options=nothing)
    # JOLI wrapper for nonlinear forward modeling
        compareGeometry(F.srcGeometry, source.geometry) == true || judiJacobianException("Source geometry mismatch")
        (DDT == Float32 && RDT == Float32) || throw(judiJacobianException("Domain and range types not supported"))
        m = n_samples(F.recGeometry, F.info)
        n = F.info.n
    
        isnothing(options) && (options = F.options)
        return J = judiExtendedJacobian{Float32,Float32}("linearized wave equation", m, n, 
            F.info, F.model, source, F.recGeometry, collect(offsets), options, bornop, adjbornop)
    end
    
    
judiJacobian(J::judiExtendedJacobian{DDT,RDT}; name=J.name, m=J.m, n=J.n, info=J.info, model=J.model, source=J.source,
        geom=J.recGeometry, off=J.offsets, opt=J.options, fop=J.fop, fop_T=J.fop_T) where {DDT, RDT} =
        judiExtendedJacobian{DDT,RDT}(name, m, n, info, model, source, geom, off, opt, fop, fop_T)



############################################################
## Forward/adjoint function to avoid unecessary extra declaration

bornop(J::judiExtendedJacobian, dm_ext) = task_distributed(extended_born, _worker_pool(), J.model, J.source, J.recGeometry, dm_ext, J.offsets; options=J.options)
adjbornop(J::judiExtendedJacobian, w) = task_distributed(cig_sso, _worker_pool(), J.model, J.source, w, J.offsets; options=J.options)

function bornop(model::Model, q::judiVector, recGeometry::Geometry, dm_ext::Array, offsets::Vector, options::Options)
    # Load full geometry for out-of-core geometry containers
    recGeometry = Geometry(recGeometry)
    srcGeometry = Geometry(q.geometry)

    # Avoid useless propage without perturbation
    if norm(dm) == 0
        return judiVector(recGeometry, zeros(Float32, recGeometry.nt[1], length(recGeometry.xloc[1])))
    end

    # Set up Python model structure
    modelPy = devito_model(model, options; dm=dm)
    dtComp = convert(Float32, modelPy."critical_dt")

    # Extrapolate input data to computational grid
    qIn = time_resample(q.data[1], q.geometry, dtComp)[1]

    # Set up coordinates
    src_coords = setup_grid(srcGeometry, model.n)  # shifts source coordinates by origin
    rec_coords = setup_grid(recGeometry, model.n)    # shifts rec coordinates by origin

    # Devito interface
    dD = pycall(impl."cig_lin", PyArray, modelPy, src_coords, qIn, rec_coords,
                dm_ext, offsets, isic=options.isic)

    return argout
end

function cig_sso(model::Model, source::judiVector, res::judiVector, offsets::Vector, options::Options)
    @assert source.nsrc == 1 "Multiple sources are used in a single-source fwi_objective"
    @assert res.nsrc == 1 "Multiple-source data is used in a single-source fwi_objective"    

    # Load full geometry for out-of-core geometry containers
    res.geometry = Geometry(res.geometry)
    source.geometry = Geometry(source.geometry)

    # Set up Python model
    modelPy = devito_model(model, options)
    dtComp = convert(Float32, modelPy."critical_dt")

    # Extrapolate input data to computational grid
    qIn = time_resample(source.data[1], source.geometry, dtComp)[1]
    dObserved = time_resample(convert(Matrix{Float32}, res.data[1]), res.geometry, dtComp)[1]

    # Set up coordinates
    src_coords = setup_grid(source.geometry, model.n)  # shifts source coordinates by origin
    rec_coords = setup_grid(res.geometry, model.n)    # shifts rec coordinates by origin

    # Devito
    g = pycall(impl."cig_grad", PyArray, modelPy, src_coords, qIn, rec_coords, dObserved,
               offsets, isic=options.isic)
    g = remove_padding_cig(g, modelPy.padsizes)
    return g
end

function remove_padding_cig(gradient::AbstractArray{DT}, nb::Array{Tuple{Int64,Int64},1}; true_adjoint::Bool=false) where {DT}
    N = size(gradient)[2:end]
    if true_adjoint
        for (dim, (nbl, nbr)) in enumerate(nb)
            diml = dim+1
            selectdim(gradient, diml, nbl+1) .+= dropdims(sum(selectdim(gradient, diml, 1:nbl), dims=diml), dims=diml)
            selectdim(gradient, diml, N[dim]-nbr) .+= dropdims(sum(selectdim(gradient, diml, N[dim]-nbr+1:N[dim]), dims=diml), dims=diml)
        end
    end
    out = gradient[:, [nbl+1:nn-nbr for ((nbl, nbr), nn) in zip(nb, N)]...]
    return out
end
