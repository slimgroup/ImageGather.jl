from devito import Inc, Operator, Function
from propagators import forward, gradient


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


def shifted_ic(g_expr, offsets):
    """
    """
    return