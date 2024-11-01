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
v0_low = .95f0 .* imfilter(v, Kernel.gaussian(5))
v0_high = 1.05f0 .* imfilter(v, Kernel.gaussian(5))

# Slowness squared [s^2/km^2]
m = (1f0 ./ v).^2
m0 = (1f0 ./ v0).^2
m0_low = (1f0 ./ v0_low).^2
m0_high = (1f0 ./ v0_high).^2
dm = vec(m - m0)

# Setup info and model structure
nsrc = 51	# number of sources
model = Model(n, d, o, m; nb=40)
model0 = Model(n, d, o, m0; nb=40)
model0_low = Model(n, d, o, m0_low; nb=40)
model0_high = Model(n, d, o, m0_high; nb=40)

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
xsrc = convertToCell(range(0f0, stop=(n[1] -1)*d[1], length=nsrc))
ysrc = convertToCell(range(0f0, stop=0f0, length=nsrc))
zsrc = convertToCell(range(20f0, stop=20f0, length=nsrc))

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
opt = Options(space_order=8)
# Setup operators
F = judiModeling(model, srcGeometry, recGeometry; options=opt)
# Nonlinear modeling
dD = F*q

# Common surface offset image gather
offsets = 0f0:150f0:8000f0
CIG = surface_gather(model0, q, dD; offsets=offsets, options=opt)
CIG_low = surface_gather(model0_low, q, dD; offsets=offsets, options=opt);
CIG_high = surface_gather(model0_high, q, dD; offsets=offsets, options=opt);

cc = 1e-1
figure(figsize=(20, 10))
subplot(231)
plot_simage(CIG[301, :, :], d; cmap="Greys", new_fig=false)
title("Good velocity (CDP=4.5km)")
subplot(232)
plot_simage(CIG_low[301, :, :], d; cmap="Greys", new_fig=false)
title("Low velocity (CDP=4.5km)")
subplot(233)
plot_simage(CIG_high[301, :, :], d; cmap="Greys", new_fig=false)
title("High velocity (CDP=4.5km)")

subplot(234)
plot_simage(CIG[101, :, :], d; cmap="Greys", new_fig=false)
subplot(235)
plot_simage(CIG_low[101, :, :], d; cmap="Greys", new_fig=false)
subplot(236)
plot_simage(CIG_high[101, :, :], d; cmap="Greys", new_fig=false)
savefig("./docs/img/cig_cdp.png", bbox_inches="tight")


# Plot gathers as a pseudo rtm image
cig_rtm_good = hcat([CIG[i, :, 1:25] for i=1:20:n[1]]...)
cig_rtm_low = hcat([CIG_low[i, :, 1:25] for i=1:20:n[1]]...)
cig_rtm_high = hcat([CIG_high[i, :, 1:25] for i=1:20:n[1]]...)

figure(figsize=(20, 10))
subplot(131)
plot_simage(cig_rtm_good, d; cmap="Greys", new_fig=false)
xticks([])
title("Good velocity")
subplot(132)
plot_simage(cig_rtm_low, d; cmap="Greys", new_fig=false)
xticks([])
title("Low velocity")
subplot(133)
plot_simage(cig_rtm_high, d; cmap="Greys", new_fig=false)
xticks([])
title("High velocity")
savefig("./docs/img/cig_line.png", bbox_inches="tight")
