# Example for basic 2D modeling:
# The receiver positions and the source wavelets are the same for each of the four experiments.
# Author: Philipp Witte, pwitte@eos.ubc.ca
# Date: January 2017
#

using JUDI, LinearAlgebra, Images, PyPlot, DSP, ImageGather

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
v0b = .95f0 .* imfilter(v, Kernel.gaussian(45))

# Slowness squared [s^2/km^2]
m = (1f0 ./ v).^2
m0 = (1f0 ./ v0).^2
m0b = (1f0 ./ v0b).^2
dm = vec(m - m0)

# Setup info and model structure
nsrc = 51	# number of sources
model = Model(n, d, o, m; nb=40)
model0 = Model(n, d, o, m0; nb=40)
model0b = Model(n, d, o, m0b; nb=40)

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
opt = Options(space_order=16)
# Setup operators
F = judiModeling(model, srcGeometry, recGeometry; options=opt)
F0 = judiModeling(model0, srcGeometry, recGeometry; options=opt)
F0b = judiModeling(model0b, srcGeometry, recGeometry; options=opt)
J = judiJacobian(F0, q)
Jb = judiJacobian(F0b, q)

# Nonlinear modeling
dD = F*q

# include("./src/TimeModeling/Modeling/surface_gather.jl")
soffs = surface_gather(model0, q, dD; offsets=0f0:150f0:8000f0, options=opt)
soffsb = surface_gather(model0b, q, dD; offsets=0f0:150f0:8000f0, options=opt);

cc = 1e-1
figure();
subplot(221)
imshow(soffs[301, :, :], vmin=-cc, vmax=cc, cmap="Greys", aspect="auto", extent=[0, 8, 5, 0])
xlabel("Surface offset (km)")
ylabel("depth (km)")
title("Good velocity (CDP=4.5km)")
subplot(222)
imshow(soffsb[301, :, :], vmin=-cc, vmax=cc, cmap="Greys", aspect="auto", extent=[0, 8, 5, 0])
title("Low velocity (CDP=4.5km)")
xlabel("Surface offset (km)")
ylabel("depth (km)")

subplot(223)
imshow(soffs[101, :, :], vmin=-cc, vmax=cc, cmap="Greys", aspect="auto", extent=[0, 8, 5, 0])
xlabel("Surface offset (km)")
ylabel("depth (km)")
title("Good velocity (CDP=1.5km)")
subplot(224)
imshow(soffsb[101, :, :], vmin=-cc, vmax=cc, cmap="Greys", aspect="auto", extent=[0, 8, 5, 0])
title("Low velocity (CDP=1.5km)")
xlabel("Surface offset (km)")
ylabel("depth (km)")



# Plot gathers as a pseudo rtm image
cig_rtm_good = hcat([soffs[i, :, :] for i=1:20:n[1]]...)
cig_rtm_bad= hcat([soffsb[i, :, :] for i=1:20:n[1]]...)

figure()
subplot(1, 2, 1)
imshow(cig_rtm_good, vmin=-cc, vmax=cc, cmap="Greys", aspect="auto", extent=[0, 8, 5, 0])
xticks([])
ylabel("depth (km)")
title("Good velocity")
subplot(1, 2, 2)
imshow(cig_rtm_bad, vmin=-cc, vmax=cc, cmap="Greys", aspect="auto", extent=[0, 8, 5, 0])
xticks([])
ylabel("depth (km)")
title("Poor velocity")

