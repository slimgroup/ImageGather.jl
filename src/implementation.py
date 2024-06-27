from devito import Inc, Operator, Function, CustomDimension, norm
from devito.finite_differences.differentiable import Add
from devito.builtins import initialize_function
from devito.types.utils import DimensionTuple

import numpy as np

from propagators import forward, gradient
from geom_utils import src_rec, geom_expr
from sensitivity import grad_expr, lin_src
from fields import wavefield
from kernels import wave_kernel
from utils import opt_op


def double_rtm(model, wavelet, src_coords, res, res_o, rec_coords, ic="as"):
    """
    """
    _, u, illum, _ = forward(model, src_coords, None, wavelet, illum=True, save=True, t_sub=4)

    # RTMs
    rtm = gradient(model, res, rec_coords, u, ic=ic)[0]
    rtmo = gradient(model, res_o, rec_coords, u, ic=ic)[0]
    return rtm.data, rtmo.data, illum.data


def cig_grad(model, src_coords, wavelet, rec_coords, res, offsets, ic="as", space_order=8, omni=False, illum=False):
    """
    """
    u = forward(model, src_coords, None, wavelet, space_order=space_order, save=True)[1]
    # Setting adjoint wavefield
    v = wavefield(model, space_order, fw=False)

    # Set up PDE expression and rearrange
    pde = wave_kernel(model, v, fw=False)

    # Setup source and receiver
    go_expr = geom_expr(model, v, src_coords=rec_coords,
                        wavelet=res, fw=False)
    # Setup gradient wrt m with all offsets
    ohs = make_offsets(offsets, model, omni)

    # Subsurface offsets.
    hs = tuple(h.shape[0] for h in ohs.values())
    hd = tuple(h.indices[0] for h in ohs.values())
    gradm = Function(name="gradm", grid=model.grid, shape=(*hs, *u.shape[1:]),
                     dimensions=(*hd, *model.grid.dimensions))

    uh, vh = shifted_wf(u, v, ohs)
    g_expr = grad_expr(gradm, uh, vh, model, ic=ic)

    # Create operator and run
    subs = model.grid.spacing_map
    op = Operator(pde + go_expr + g_expr, subs=subs, name="cig_sso", opt=opt_op(model))

    try:
        op.cfunction
    except:
        op = Operator(pde + go_expr + g_expr, subs=subs, name="cig_sso", opt='advanced')
        op.cfunction

    # Get bounds from offsets
    dim_kw = make_kw(ohs, DimensionTuple(*u.shape[1:], getters=model.grid.dimensions))
    op(dt=model.critical_dt, **dim_kw)

    # Output
    return gradm.data


def cig_lin(model, src_coords, wavelet, rec_coords, dm_ext, offsets, ic="as", space_order=8, omni=False):
    """
    """
    nt = wavelet.shape[0]
    dt = model.grid.time_dim.spacing
    oh = make_offsets(offsets, model, omni)

    # Setting wavefield
    u = wavefield(model, space_order, nt=nt)
    ul = wavefield(model, space_order, name="l")

    # Set up PDE expression and rearrange
    pde = wave_kernel(model, u)
    qlin = ext_src(model, u, dm_ext, oh, ic=ic)
    fact = 1 / (model.damp/dt + (model.m * model.irho)/dt**2)
    pdel = wave_kernel(model, ul) + [Inc(ul.forward, fact * qlin)]

    # Setup source and receiver
    go_expr = geom_expr(model, u, src_coords=src_coords, wavelet=wavelet)
    go_exprl = geom_expr(model, ul, rec_coords=rec_coords, nt=nt)
    _, rcvl = src_rec(model, ul, rec_coords=rec_coords, nt=nt)
    # Create operator and run
    subs = model.grid.spacing_map
    op = Operator(pde + go_expr + pdel + go_exprl,
                  subs=subs, name="extborn", opt=opt_op(model))
    op.cfunction

    # Remove edge for offsets
    dim_kw = make_kw(oh, DimensionTuple(*u.shape[1:], getters=model.grid.dimensions),
                     born=True)
    op(dt=model.critical_dt, rcvul=rcvl, **dim_kw)

    # Output
    return rcvl.data


def ext_src(model, u, dm_ext, oh, ic="as"):
    # Extended perturbation
    hs = (h.shape[0] for h in oh.values())
    hd = (h.indices[0] for h in oh.values())
    dm = Function(name="gradm", grid=model.grid, shape=(*hs, *u.shape[1:]),
                  dimensions=(*hd, *model.grid.dimensions),
                  space_order=u.space_order)
    initialize_function(dm, dm_ext, (*((0, 0) for _ in oh.values()), *model.padsizes))

    # extended source
    uh = _shift(u, oh)
    ql = -model.irho * _shift(uh.dt2 * dm, oh)
    return ql


def make_offsets(offsets, model, omni):
    dims = model.grid.dimensions
    dims = dims if omni else (dims[0],)
    ohs = dict()
    for d in dims:
        nh = offsets.shape[0]
        offs = CustomDimension("off%s" % d.name, 0, nh-1, nh)
        oh = Function(name="offv%s" % d.name, grid=model.grid, space_order=0,
                      dimensions=(offs,), shape=(nh,), dtype=np.int32)
        oh.data[:] = offsets // model.grid.spacing[0]
        ohs[d] = oh
    return ohs


def shifted_wf(u, v, ohs):
    uh = u
    vh = v
    for k, v in ohs.items():
        if v.shape[0] == 1:
            continue
        uh = uh._subs(k, k-v)
        vh = vh._subs(k, k+v)
    return uh, vh


def _shift(u, oh):
    uh = u 
    for k, v in oh.items():
        uh = uh._subs(k, k-v)
    return uh


def make_kw(ohs, shape, born=False):
    kw = dict()
    scale = 2 if born else 1
    for d, v in ohs.items():
        kw['%s_m' % d.name] = -scale*np.min(v.data.view(np.ndarray))
        kw['%s_M' % d.name] = shape[d] - scale*np.max(v.data.view(np.ndarray))
    return kw
