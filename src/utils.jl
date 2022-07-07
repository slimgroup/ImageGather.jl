export mv_avg_2d, delta_h, envelope, mute, mute!, laplacian, offset_map

"""
    mv_avg_2d(x; k=5)

2D moving average with a square window of width k
"""
function mv_avg_2d(x::AbstractArray{T, 2}; k=5) where T
    out = 1f0 * x
    nx, ny = size(x)
    kl, kr = k÷2 + 1, k÷2
    for i=kl:nx-kr, j=kl:ny-kr
        out[i, j] = sum(x[i+k, j+l] for k=-kr:kr, l=-kr:kr)/k^2
    end
    out
end

"""
    envelope(x)

Envelope of a vector or a 2D matrix. The envelope over the first dimension is taken for a 2D matrix (see DSP `hilbert`)
"""
envelope(x::AbstractArray{T, 1}) where T = abs.(hilbert(x))

"""
    envelope(x)

Envelope of a vector or a 1D vector (see DSP `hilbert`)

"""
envelope(x::AbstractArray{T, 2}) where T = abs.(hilbert(x))


"""
    delta_h(ha, h, tol)

Compute the binary mask where `ha` is within `tol` of `h`.
"""
delta_h(ha::AbstractArray{T, 2}, h::Number, tol::Number) where T = Float32.(abs.(h .- ha) .<= tol)

"""
    mute!(shot, offsets;vp=1500, t0=1/10, dt=0.004)

In place direct wave muting of a shot record with water sound speed `vp`, time sampling `dt` and firing time `t0`.
"""
function mute!(shot::AbstractArray{Ts, 2}, offsets::Vector{To}; vp=1500, t0=1/10, dt=.004) where {To, Ts}
    length(offsets) == size(shot, 2) || throw(DimensionMismatch("Number of offsets has to match the number of traces"))
    inds = trunc.(Integer, (offsets ./ vp .+ t0) ./ dt)
    inds = min.(max.(1, inds), size(shot, 1))
    for (rx, i) = enumerate(inds)
        shot[1:i, rx] .= 0f0
    end
end

"""
    mute(shot, offsets;vp=1500, t0=1/10, dt=0.004)

Direct wave muting of a shot record with water sound speed `vp`, time sampling `dt` and firing time `t0`.
"""
function mute(shot::AbstractArray{Ts, 2}, offsets::Vector{To}; vp=1500, t0=1/10, dt=.004) where {To, Ts}
    out = Ts(1) .* shot
    mute!(out, offsets; vp=vp, t0=t0, dt=dt)
    out
end


"""
    laplacian(image; hx=1, hy=1)

2D laplacian of an image with grid spacings (hx, hy)
"""
function laplacian(image::AbstractArray{T, 2}; hx=1, hy=1) where T
    scale = 1/(hx*hy)
    out = 1 .* image
    @views begin
        out[2:end-1, 2:end-1] .= -4 .* image[2:end-1, 2:end-1] 
        out[2:end-1, 2:end-1]  .+= image[1:end-2, 2:end-1] + image[2:end-1, 1:end-2]
        out[2:end-1, 2:end-1]  .+= image[3:end, 2:end-1] + image[2:end-1, 3:end] 
    end
    return scale .* out
end

"""
    offset_map(rtm, rtmo; scale=0)

Return the regularized least-square division of rtm and rtmo. The regularization consists of the envelope and moving average
followed by the least-square division [`surface_gather`](@ref)
"""
function offset_map(rtm::AbstractArray{T, 2}, rtmo::AbstractArray{T, 2}; scale=0) where T
    rtmn = mv_avg_2d(envelope(rtm))
    rtmo = mv_avg_2d(envelope(rtmo))

    offset_map = rtmn .* rtmo ./ (rtmn .* rtmn .+ eps(Float32)) .- scale
    return offset_map
end