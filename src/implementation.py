from devito import Inc, Operator, Function, CustomDimension

import numpy as np

from propagators import forward, gradient
from geom_utils import src_rec
from sensitivity import grad_expr
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
    dims = model.grid.dimensions
    N = model.grid.shape
    nh = offsets.shape[0]
    offs = CustomDimension("hdim", 0, nh-1, nh)
    oh = Function(name="hvals", grid=model.grid, space_order=0, dimensions=(offs,), shape=(nh,), dtype=np.int32)
    oh.data[:] = offsets // model.grid.spacing[0]
    gradm = Function(name="gradm", grid=model.grid, dimensions=(offs, *dims), shape=(len(offsets), *N), space_order=0)
    g_expr = grad_expr(gradm, u._subs(dims[0], dims[0]+oh), v._subs(dims[0], dims[0]-oh), model, isic=isic)

    # Create operator and run
    subs = model.spacing_map
    from IPython import embed; embed()
    op = Operator(pde + geom_expr + g_expr,subs=subs, name="cig_sso", opt=opt_op(model))
    try:
        op.cfunction
    except:
        op = Operator(pde + geom_expr + g_expr,subs=subs, name="cig_sso", opt='advanced')
        op.cfunction
    
    # Output
    return gradm.data