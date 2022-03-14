# Author: Mathias Louboutin
# Date: June 2021
#

using JUDI, LinearAlgebra, Images, PyPlot, DSP, ImageGather
import ImageGather: cig_sso
# Set up model structure
n = (601, 333)   # (x,y,z) or (x,z)
d = (30., 30.)
o = (0., 0.)

# Velocity [km/s]
v = ones(Float32,n) .+ 0.5f0
for i=1:12
    v[:,25*i+1:end] .= 1.5f0 + i*.25f0
end
v0 = imfilter(v, Kernel.gaussian(5))
v0_low = .95f0 .* imfilter(v, Kernel.gaussian(45))
v0_high = 1.05f0 .* imfilter(v, Kernel.gaussian(45))

# Slowness squared [s^2/km^2]
m = (1f0 ./ v).^2
m0 = (1f0 ./ v0).^2
m0_low = (1f0 ./ v0_low).^2
m0_high = (1f0 ./ v0_high).^2
dm = vec(m - m0)

# Setup info and model structure
nsrc = 3	# number of sources
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
opt = Options(space_order=12, isic=false)
# Setup operators
F = judiModeling(model, srcGeometry, recGeometry; options=opt)
# Nonlinear modeling
dD = F*q - F(model0)*q

# Common surface offset image gather
offsets = -150f0:model.d[1]:150f0
J = judiExtendedJacobian(F(model0), q, offsets)

cig = cig_sso(model0, q[2], dD[2], collect(offsets), opt)
cig_sso = J'*dD