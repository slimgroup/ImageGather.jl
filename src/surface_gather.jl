import JUDI: task_distributed, _worker_pool
export surface_gather, offset_map


"""
    surface_gather(model, q, data; offsets=nothing, options=Options())

Compute the surface offset gathers volume (nx (X ny) X nz X no) for via the double rtm method whith `no` offsets.


Parameters
* `model`: JUDI Model structure.
* `q`: Source, judiVector.
* `data`: Obeserved data, judiVector.
* `offsets`: List of offsets to compute the gather at. Optional (defaults to 0:10*model.d:model.extent)
* `options`: JUDI Options structure.
"""
function surface_gather(model::Model, q::judiVector, data::judiVector; offsets=nothing, options=Options())
    isnothing(offsets) && (offsets = 0f0:10*model.d[1]:(model.n[1]-1)*model.d[1])
    offsets = collect(offsets)
    results = task_distributed(double_rtm_cig, _worker_pool(), model, q, data, offsets; options=options)
    return results
end

"""
    double_rtm_cig(model, q, data, offsets, options)

Compute the single shot contribution to the surface offset gather via double rtm. This single source contribution consists of the following steps:

1. Mute direct arrival in the data.
2. Compute standard RTM ``R``.
3. Compute the offset RTM ``R_o`` for the the weighted data where each trace is weighted by its offset `(rec_x - src_x)`.
4. Compute the envelope ``R_e = \\mathcal{E}(R)`` and ``R_{oe} = \\mathcal{E}(R_o)``.
5. Compute the offset map ``\\frac{R_e \\odot R_{oe}}{R_e \\odot R_e + \\epsilon}``.
6. Apply illumination correction and laplace filter ``R_l = \\mathcal{D} \\Delta R``.
7. Compute each offset contribution ``\\mathcal{I}[:, h] = R_l \\odot \\delta[ha - h]_{tol}`` [`delta_h`](@ref).
8. Return ``\\mathcal{I}``.

"""
function double_rtm_cig(model_full, q::judiVector, data::judiVector, offs, options)
    # Load full geometry for out-of-core geometry containers
    data.geometry = Geometry(data.geometry)
    q.geometry = Geometry(q.geometry)
    ndim = length(model_full.n)

    # Limit model to area with sources/receivers
    if options.limit_m == true
        model = deepcopy(model_full)
        model, _ = limit_model_to_receiver_area(q.geometry, data.geometry, model, options.buffer_size)
    else
        model = model_full
    end

    # Set up Python model
    modelPy = devito_model(model, options)
    dtComp = get_dt(model; dt=options.dt_comp)

    # Extrapolate input data to computational grid
    qIn = time_resample(q.data[1], q.geometry, dtComp)[1]
    obsd = typeof(data.data[1]) == JUDI.SegyIO.SeisCon ? convert(Array{Float32,2}, data.data[1][1].data) : data.data[1]
    res = time_resample(obsd, data.geometry, dtComp)[1]

    # Set up coordinates
    src_coords = setup_grid(q.geometry, model.n)  # shifts source coordinates by origin
    rec_coords = setup_grid(data.geometry, model.n)    # shifts rec coordinates by origin

    # Src-rec offsets
    scale = 5f3
    off_r = abs.(data.geometry.xloc[1] .- q.geometry.xloc[1]) .+ scale

    # mute
    mute!(res, off_r .- scale; dt=dtComp/1f3, t0=.25)
    res_o = res .* off_r'
    # Double rtm
    rtm, rtmo, illum = pycall(impl."double_rtm", Tuple{Array{Float32, modelPy.dim},  Array{Float32, modelPy.dim}, Array{Float32, modelPy.dim}},
                              modelPy, qIn, src_coords, res, res_o, rec_coords, space_order=options.space_order)
    rtm = remove_padding(rtm, modelPy.padsizes)
    rtmo = remove_padding(rtmo, modelPy.padsizes)
    illum = remove_padding(illum, modelPy.padsizes)

    # offset map
    h_map = offset_map(rtm, rtmo; scale=scale)

    rtm = laplacian(rtm)
    rtm[illum .> 0] ./= illum[illum .> 0]

    soffs = zeros(Float32, model.n..., length(offs))

    for (i, h) in enumerate(offs)
        soffs[:, :, i] .+= rtm .* delta_h(h_map, h, 2*diff(offs)[1])
    end
    GC.gc()
    return soffs
end
