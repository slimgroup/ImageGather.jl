# Author: Mathias Louboutin
# Date: June 2021
#

using JUDI, LinearAlgebra, Images, PyPlot, DSP, ImageGather, SlimPlotting

# Set up model structure
n = (601, 333)   # (x,y,z) or (x,z)
d = (15., 15.)
o = (0., 0.)

# Velocity [km/s]
v = ones(Float32,n) .+ 0.5f0
for i=1:12
    v[:,25*i+1:end] .= 1.5f0 + i*.25f0
end
v0 = imfilter(v, Kernel.gaussian(5))

# Slowness squared [s^2/km^2]
m = (1f0 ./ v).^2
m0 = (1f0 ./ v0).^2

# Setup info and model structure
nsrc = 1	# number of sources
model = Model(n, d, o, m; nb=40)
model0 = Model(n, d, o, m0; nb=40)

# Set up receiver geometry
nxrec = 401
xrec = range(0f0, stop=(n[1] -1)*d[1], length=nxrec)
yrec = 0f0
zrec = range(20f0, stop=20f0, length=nxrec)

# receiver sampling and recording time
timeR = 4000f0   # receiver recording time [ms]
dtR = 4f0    # receiver sampling interval [ms]

# Set up receiver structure
recGeometry = Geometry(xrec, yrec, zrec; dt=dtR, t=timeR, nsrc=nsrc)
# Set up source geometry (cell array with source locations for each shot)
xsrc = 4500f0
ysrc = 0f0
zsrc = 20f0

# source sampling and number of time steps
timeS = 4000f0  # ms
dtS = 4f0   # ms

# Set up source structure
srcGeometry = Geometry(xsrc, ysrc, zsrc; dt=dtS, t=timeS)
# setup wavelet
f0 = 0.015f0     # kHz
wavelet = ricker_wavelet(timeS, dtS, f0)
q = judiVector(srcGeometry, wavelet)

###################################################################################################
opt = Options(space_order=16, IC="as")
# Setup operators
F = judiModeling(model, srcGeometry, recGeometry; options=opt)
F0 = judiModeling(model0, srcGeometry, recGeometry; options=opt)
# Nonlinear modeling
dD = F*q

# Make rtms
J = judiJacobian(F0, q)

# Get offsets and mute data
offs  = abs.(xrec .- xsrc)
res = deepcopy(dD)
mute!(res.data[1], offs)
reso = deepcopy(res)

I = inv(judiIllumination(J))

rtm = I*J'*res

omap = Array{Any}(undef, 2)
i = 1

# try a bunch of weighting functions
for (wf, iwf) = zip([x-> x .+ 5f3, x-> log.(x .+ 10)], [x -> x .- 5f3, x-> exp.(x) .- 10])
    reso.data[1] .= res.data[1] .* wf(offs)'
    rtmo = I*J'*reso
    omap[i] = iwf(offset_map(rtm.data, rtmo.data))
    global i+=1
end

figure()
for (i, name)=enumerate(["shift", "log"])
    subplot(1,2,i)
    plot_velocity(omap[i]', (1,1); cmap="jet", aspect="auto", perc=98, new_fig=false, vmax=5000)
    colorbar()
    title(name)
end
tight_layout()
