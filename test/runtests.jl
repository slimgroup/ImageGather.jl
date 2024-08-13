using ImageGather, Test

using JUDI, LinearAlgebra

# Set up model structure
n = (301, 151)   # (x,y,z) or (x,z)
d = (10., 10.)
o = (0., 0.)

# Velocity [km/s]
v =  1.5f0 .* ones(Float32,n)
v[:, 76:end] .= 2.5f0
v0 = 1.5f0 .* ones(Float32,n)
# Slowness squared [s^2/km^2]
m = (1f0 ./ v).^2
m0 = (1f0 ./ v0).^2

# Setup info and model structure
nsrc = 1	# number of sources
model = Model(n, d, o, m; nb=40)
model0 = Model(n, d, o, m0; nb=40)

dm = model.m - model0.m

# Set up receiver geometry
nxrec = 151
xrec = range(0f0, stop=(n[1] -1)*d[1], length=nxrec)
yrec = 0f0
zrec = range(20f0, stop=20f0, length=nxrec)

# receiver sampling and recording time
timeD = 2000f0   # receiver recording time [ms]
dtD = 4f0    # receiver sampling interval [ms]

# Set up receiver structure
recGeometry = Geometry(xrec, yrec, zrec; dt=dtD, t=timeD, nsrc=nsrc)
# Set up source geometry (cell array with source locations for each shot)
xsrc = 1500f0
ysrc = 0f0
zsrc = 20f0

# Set up source structure
srcGeometry = Geometry(xsrc, ysrc, zsrc; dt=dtD, t=timeD)
# setup wavelet
f0 = 0.015f0     # kHz
wavelet = ricker_wavelet(timeD, dtD, f0)
q = diff(judiVector(srcGeometry, wavelet))

###################################################################################################
opt = Options()
# Setup operators
F = judiModeling(model, srcGeometry, recGeometry; options=opt)
J0 = judiJacobian(F(model0), q)
# Nonlinear modeling
dD = J0*dm
rtm = J0'*dD

# Common surface offset image gather
offsets = -40f0:model.d[1]:40f0
nh = length(offsets)

for dims in ((:x, :z), :z, :x)

    J = judiExtendedJacobian(F(model0), q, offsets, dims=dims)

    ssodm = J'*dD
    @show size(ssodm)
    @test size(ssodm, 1) == nh

    ssor = zeros(Float32, size(ssodm)...)
    for h=1:size(ssor, 1)
        if dims == (:x, :z)
            for h2=1:size(ssor, 2)
                ssor[h, h2, :, :] .= dm.data
            end
        else
            ssor[h, :, :] .= dm.data
        end
    end

    dDe = J*ssor
    # @show norm(dDe - dD), norm(ssor[:] - dm[:])
    a, b = dot(dD, dDe), dot(ssodm[:], ssor[:])

    @test (a-b)/(a+b) ≈ 0 atol=1f-3 rtol=0

    # Make sure zero offset is the rtm, remove the sumpadding
    ih = div(nh, 2)+1
    rtmc = dims == (:x, :z) ? ssodm[ih, ih, :, :] : ssodm[ih, :, :]

    @test norm(rtm.data - rtmc, Inf) ≈ 0f0 atol=1f-4 rtol=0
end