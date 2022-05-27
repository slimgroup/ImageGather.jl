from devito import Inc, Operator, Function, CustomDimension
from devito.builtins import initialize_function

import numpy as np

from propagators import forward, gradient
from geom_utils import src_rec
from sensitivity import grad_expr, lin_src
from wave_utils import wavefield
from kernels import wave_kernel
from utils import opt_op


def double_rtm(model, wavelet, src_coords, res, res_o, rec_coords, space_order=8):
    """
    """
    _, u, _ = forward(model, src_coords, None, wavelet, space_order=space_order,
                      save=True)

    # Illumination
    illum = Function(name="I", grid=model.grid, space_order=0)
    Operator(Inc(illum, u**2))()

    # RTMs
    rtm, _ = gradient(model, res, rec_coords, u, space_order=space_order)
    rtmo, _ = gradient(model, res_o, rec_coords, u, space_order=space_order)
    return rtm.data, rtmo.data, illum.data


def cig_grad(model, src_coords, wavelet, rec_coords, res, offsets, isic=False, space_order=8):
    """
    """
    _, u, _ = forward(model, src_coords, None, wavelet, space_order=space_order,
                      save=True)
    # Setting adjoint wavefieldgradient
    v = wavefield(model, space_order, fw=False)

    # Set up PDE expression and rearrange
    pde = wave_kernel(model, v, fw=False)

    # Setup source and receiver
    geom_expr, _, _ = src_rec(model, v, src_coords=rec_coords,
                              wavelet=res, fw=False)

    # Setup gradient wrt m with all offsets
    oh = make_offsets(offsets, model)
    x = u.indices[1]

    # Subsurface offsets.
    gradm = Function(name="gradm", grid=model.grid, shape=(oh.shape[0], *u.shape[1:]),
                     dimensions=(oh.indices[0], *model.grid.dimensions), space_order=0)
    g_expr = grad_expr(gradm, u._subs(x, x-oh), v._subs(x, x+oh),
                       model, isic=isic)

    # Create operator and run
    subs = model.spacing_map
    op = Operator(pde + geom_expr + g_expr,subs=subs, name="cig_sso", opt=opt_op(model))
    try:
        op.cfunction
    except:
        op = Operator(pde + geom_expr + g_expr,subs=subs, name="cig_sso", opt='advanced')
        op.cfunction
    # Get bounds from offsets
    xm, xM = -np.min(oh.data), u.shape[1] - np.max(oh.data)
    op(x_m=xm, x_M=xM)

    # Output
    return gradm.data


def cig_lin(model, src_coords, wavelet, rec_coords, dm_ext, offsets, isic=False, space_order=8):
    """
    """
    nt = wavelet.shape[0]
    dt = model.grid.time_dim.spacing
    oh = make_offsets(offsets, model)

    # Setting wavefield
    u = wavefield(model, space_order, nt=nt)
    ul = wavefield(model, space_order, name="l")

    # Set up PDE expression and rearrange
    pde = wave_kernel(model, u)
    qlin = ext_src(model, u, dm_ext, oh, isic=isic)
    pdel = wave_kernel(model, ul) + [Inc(ul.forward, dt**2/(model.irho*model.m) * qlin)]

    # Setup source and receiver
    geom_expr, _, _ = src_rec(model, u, rec_coords=None,
                              src_coords=src_coords, wavelet=wavelet)
    geom_exprl, _, rcvl = src_rec(model, ul, rec_coords=rec_coords, nt=nt)

    # Create operator and run
    subs = model.spacing_map
    op = Operator(pde + geom_expr + pdel + geom_exprl,
                  subs=subs, name="extborn", opt=opt_op(model))
    op.cfunction

    # Remove edge for offsets
    xm, xM = -np.min(oh.data), u.shape[1] - np.max(oh.data)
    op(x_m=xm, x_M=xM)

    # Output
    return rcvl.data


def ext_src(model, u, dm_ext, oh, isic=False):
    # Extended perturbation
    dm = Function(name="gradm", grid=model.grid, shape=(oh.shape[0], *u.shape[1:]),
                  dimensions=(oh.indices[0], *model.grid.dimensions),
                  space_order=u.space_order)
    initialize_function(dm, dm_ext,[(0, 0)] + model.padsizes)

    # extended source
    x = u.indices[1]
    uh = u._subs(x, x-oh)
    ql = model.irho * uh.dt2 * dm._subs(x, x+oh)
    return ql


def make_offsets(offsets, model):
    nh = offsets.shape[0]
    offs = CustomDimension("hdim", 0, nh-1, nh)
    oh = Function(name="hvals", grid=model.grid, space_order=0,
                  dimensions=(offs,), shape=(nh,), dtype=np.int32)
    oh.data[:] = offsets // model.grid.spacing[0]
    return oh