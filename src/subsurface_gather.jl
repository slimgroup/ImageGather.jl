export judiExtendedJacobian

struct judiExtendedJacobian{D, O, FT} <: judiAbstractJacobian{D, O, FT}
    m::AbstractSize
    n::AbstractSize
    F::FT
    q::judiMultiSourceVector
    offsets::Vector{D}
    dims::Vector{Symbol}
end

"""
    J = judiExtendedJacobian(F, q, offsets; options::JUDIOptions, omni=false, dims=nothing)

Extended jacobian (extended Born modeling operator) for subsurface horsizontal offsets `offsets`. Its adjoint
comput the subsurface common offset volume. In succint way, the extened born modeling Operator can summarized in a linear algebra frmaework as:

Options structure for seismic modeling.

`omni`: If `true`, the extended jacobian will be computed for all dimensions.
`dims`: If `omni` is `false`, the extended jacobian will be computed for the dimension(s) specified in `dims`.

"""
function judiExtendedJacobian(F::judiComposedPropagator{D, O}, q::judiMultiSourceVector, offsets;
                              options=nothing, omni=false, dims=nothing) where {D, O}
    JUDI.update!(F.options, options)
    offsets = Vector{D}(offsets)
    ndim = length(F.model.n)
    if omni
        dims = [:x, :y, :z][1:ndim]
    else
        if isnothing(dims)
            dims = [:x]
        else
            dims = symvec(dims)
            if ndim == 2
                dims[dims .== :z] .= :y
            end
        end
    end

    return judiExtendedJacobian{D, :born, typeof(F)}(F.m, space(F.model.n), F, q, offsets, dims)
end

symvec(s::Symbol) = [s]
symvec(s::Tuple) = [symvec(ss)[1] for ss in s]::Vector{Symbol}
symvec(s::Vector) = [symvec(ss)[1] for ss in s]::Vector{Symbol}

adjoint(J::judiExtendedJacobian{D, O, FT}) where {D, O, FT} = judiExtendedJacobian{D, adjoint(O), FT}(J.n, J.m, J.F, J.q, J.offsets, J.dims)
getindex(J::judiExtendedJacobian{D, O, FT}, i) where {D, O, FT} = judiExtendedJacobian{D, O, FT}(J.m[i], J.n[i], J.F[i], J.q[i], J.offsets, J.dims)

function make_input(J::judiExtendedJacobian{D, :adjoint_born, FT}, q) where {D, FT}
    srcGeom, srcData = JUDI.make_src(J.q, J.F.qInjection)
    recGeom, recData = JUDI.make_src(q, J.F.rInterpolation)
    return srcGeom, srcData, recGeom, recData, nothing
end

function make_input(J::judiExtendedJacobian{D, :born, FT}, dm) where {D<:Number, FT}
    srcGeom, srcData = JUDI.make_src(J.q, J.F.qInjection)
    return srcGeom, srcData, J.F.rInterpolation.data[1], nothing, dm
end

*(J::judiExtendedJacobian{T, :born, O}, dm::Array{T, 3}) where {T, O} = J*vec(dm)
*(J::judiExtendedJacobian{T, :born, O}, dm::Array{T, 4}) where {T, O} = J*vec(dm)

JUDI.process_input_data(::judiExtendedJacobian{D, :born, FT}, q::Vector{D}) where {D<:Number, FT} = q

############################################################

function propagate(J::judiExtendedJacobian{T, :born, O}, q::AbstractArray{T}, illum::Bool) where {T, O}
    srcGeometry, srcData, recGeometry, _, dm = make_input(J, q)
    # Load full geometry for out-of-core geometry containers
    recGeometry = Geometry(recGeometry)
    srcGeometry = Geometry(srcGeometry)

    # Avoid useless propage without perturbation
    if minimum(dm) == 0 && maximum(dm) == 0
        return judiVector(recGeometry, zeros(Float32, recGeometry.nt[1], length(recGeometry.xloc[1])))
    end

    # Set up Python model structure
    modelPy = devito_model(J.model, J.options)
    nh = [length(J.offsets) for _=1:length(J.dims)]
    dmd = reshape(dm, nh..., J.model.n...)
    dtComp = pyconvert(Float32, modelPy.critical_dt)

    # Extrapolate input data to computational grid
    qIn = time_resample(srcData, srcGeometry, dtComp)

    # Set up coordinates
    src_coords = setup_grid(srcGeometry, J.model.n)  # shifts source coordinates by origin
    rec_coords = setup_grid(recGeometry, J.model.n)    # shifts rec coordinates by origin

    # Devito interface
    dD = impl.cig_lin(modelPy, src_coords, qIn, rec_coords, dmd, J.offsets,
                      ic=J.options.IC, space_order=J.options.space_order, dims=J.dims)
    dD = time_resample(PyArray(dD), dtComp, recGeometry)
    # Output shot record as judiVector
    return judiVector{Float32, Matrix{Float32}}(1, recGeometry, [dD])
end

function propagate(J::judiExtendedJacobian{T, :adjoint_born, O}, q::AbstractArray{T}, illum::Bool) where {T, O}
    srcGeometry, srcData, recGeometry, recData, _ = make_input(J, q)
    # Load full geometry for out-of-core geometry containers
    recGeometry = Geometry(recGeometry)
    srcGeometry = Geometry(srcGeometry)

    # Set up Python model
    modelPy = devito_model(J.model, J.options)
    dtComp = pyconvert(Float32, modelPy.critical_dt)

    # Extrapolate input data to computational grid
    qIn = time_resample(srcData, srcGeometry, dtComp)
    dObserved = time_resample(recData, recGeometry, dtComp)

    # Set up coordinates
    src_coords = setup_grid(srcGeometry, J.model.n)  # shifts source coordinates by origin
    rec_coords = setup_grid(recGeometry, J.model.n)  # shifts rec coordinates by origin

    # Devito
    g = impl.cig_grad(modelPy, src_coords, qIn, rec_coords, dObserved, J.offsets,
                      illum=false, ic=J.options.IC, space_order=J.options.space_order, dims=J.dims,
                      t_sub=J.options.subsampling_factor)
    g = remove_padding_cig(PyArray(g), pyconvert(Tuple, modelPy.padsizes); true_adjoint=J.options.sum_padding)
    return g
end

function remove_padding_cig(gradient::AbstractArray{DT}, nb::NTuple{Nd, NTuple{2, Int64}}; true_adjoint::Bool=false) where {DT, Nd}
    no = ndims(gradient) - length(nb)
    N = size(gradient)[no+1:end]
    hd = tuple([Colon() for _=1:no]...)
    if true_adjoint
        for (dim, (nbl, nbr)) in enumerate(nb)
            diml = dim+no
            selectdim(gradient, diml, nbl+1) .+= dropdims(sum(selectdim(gradient, diml, 1:nbl), dims=diml), dims=diml)
            selectdim(gradient, diml, N[dim]-nbr) .+= dropdims(sum(selectdim(gradient, diml, N[dim]-nbr+1:N[dim]), dims=diml), dims=diml)
        end
    end
    out = gradient[hd..., [nbl+1:nn-nbr for ((nbl, nbr), nn) in zip(nb, N)]...]
    return out
end
