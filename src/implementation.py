from devito import Inc, Operator, Function, CustomDimension
from devito.builtins import initialize_function

import numpy as np

from propagators import forward, gradient
from geom_utils import src_rec, geom_expr
from sensitivity import grad_expr
from fields import wavefield
from kernels import wave_kernel
from utils import opt_op

try:
    from devitopro import *  # noqa
except ImportError:
    pass


def double_rtm(model, wavelet, src_coords, res, res_o, rec_coords, ic="as", space_order=8):
    """
    """
    _, u, illum, _ = forward(model, src_coords, None, wavelet, illum=True,
                             save=True, t_sub=4, space_order=space_order)

    # RTMs
    rtm = gradient(model, res, rec_coords, u, ic=ic)[0]
    rtmo = gradient(model, res_o, rec_coords, u, ic=ic)[0]
    return rtm.data, rtmo.data, illum.data


def cig_grad(model, src_coords, wavelet, rec_coords, res, offsets, ic="as",
             space_order=8, dims=None, illum=False, t_sub=1):
    """
    """
    so = max(space_order, np.max(np.abs(offsets)) // model.grid.spacing[0])
    _, u, _, _ = forward(model, src_coords, None, wavelet, t_sub=t_sub,
                         illum=illum, space_order=(space_order, so, so), save=True)
    # Setting adjoint wavefield
    v = wavefield(model, space_order, fw=False, tfull=True)

    # Set up PDE expression and rearrange
    pde, extra = wave_kernel(model, v, fw=False)

    # Setup source and receiver
    go_expr = geom_expr(model, v, src_coords=rec_coords, wavelet=res, fw=False)
    # Setup gradient wrt m with all offsets
    ohs = make_offsets(offsets, model, dims)

    # Subsurface offsets.
    hs = tuple(h.shape[0] for h in ohs.values())
    hd = tuple(h.indices[0] for h in ohs.values())
    gradm = Function(name="gradm", grid=model.grid, shape=(*hs, *u.shape[1:]),
                     dimensions=(*hd, *model.grid.dimensions))

    uh, vh = shifted_wf(u, v, ohs)
    g_expr = grad_expr(gradm, uh, vh, model, ic=ic)

    # Create operator and run
    subs = model.grid.spacing_map
    op = Operator(pde + go_expr + g_expr, subs=subs, name="cig_sso",
                  opt=opt_op(model))
    op.cfunction

    # Get bounds from offsets
    op(dt=model.critical_dt)

    # Output
    return gradm.data


def cig_lin(model, src_coords, wavelet, rec_coords, dm_ext, offsets,
            ic="as", space_order=8, dims=None):
    """
    """
    so = max(space_order, np.max(np.abs(offsets)) // model.grid.spacing[0])
    nt = wavelet.shape[0]
    dt = model.grid.time_dim.spacing
    oh = make_offsets(offsets, model, dims)

    # Setting wavefield
    u = wavefield(model, (space_order, 2*so, so), nt=nt, tfull=True)
    ul = wavefield(model, (space_order, so, so), name="l")

    # Set up PDE expression and rearrange
    pde, extra = wave_kernel(model, u)
    qlin = ext_src(model, u, dm_ext, oh, ul, ic=ic)
    fact = 1 / (model.damp/dt + (model.m * model.irho)/dt**2)
    pdel, extral = wave_kernel(model, ul)
    pdel += [Inc(ul.forward, fact * qlin)]

    # Setup source and receiver
    go_expr = geom_expr(model, u, src_coords=src_coords, wavelet=wavelet)
    go_exprl = geom_expr(model, ul, rec_coords=rec_coords, nt=nt)
    _, rcvl = src_rec(model, ul, rec_coords=rec_coords, nt=nt)
    # Create operator and run
    subs = model.grid.spacing_map
    mode, opt = opt_op(model)
    opt['min-storage'] = True
    op = Operator(pde + go_expr + extra + pdel + extral + go_exprl,
                  subs=subs, name="extborn", opt=(mode, opt))
    op.cfunction

    # Remove edge for offsets
    op(dt=model.critical_dt, rcvul=rcvl)

    # Output
    return rcvl.data


def ext_src(model, u, dm_ext, oh, ul, ic="as"):
    # Extended perturbation
    hs = (h.shape[0] for h in oh.values())
    hd = (h.indices[0] for h in oh.values())
    dm = Function(name="gradm", grid=model.grid, shape=(*hs, *u.shape[1:]),
                  dimensions=(*hd, *model.grid.dimensions),
                  space_order=u.space_order)
    initialize_function(dm, dm_ext, (*((0, 0) for _ in oh.values()),
                                     *model.padsizes))

    # extended source
    uh = u
    for k, v in oh.items():
        uh = uh._subs(k, k - 2 * v)
        dm = dm._subs(k, k - v)

    ql = -model.irho * uh.dt2 * dm
    return ql


def make_offsets(offsets, model, dims):
    dims = [d for d in model.grid.dimensions if d.name in dims]
    x, z = model.grid.dimensions[0], model.grid.dimensions[-1]
    ohs = dict()
    nh = offsets.shape[0]
    for d in dims:
        parent = x if (len(dims) == 1 and dims[0] == z) else None
        offs = CustomDimension("off%s" % d.name, 0, nh-1, nh, parent=parent)
        oh = Function(name="offv%s" % d.name, space_order=0,
                      dimensions=(offs,), shape=(nh,), dtype=np.int32)
        oh.data[:] = offsets // model.grid.spacing[0]
        ohs[d] = oh
    return ohs


def shifted_wf(u, v, ohs):
    uh = u
    vh = v
    for k, oh in ohs.items():
        if v.shape[0] == 1:
            continue
        uh = uh._subs(k, k - oh)
        vh = vh._subs(k, k + oh)
    return uh, vh
