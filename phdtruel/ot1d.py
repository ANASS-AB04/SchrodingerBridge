import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import PchipInterpolator
from scipy.signal import savgol_filter


def mass(x, a):
    return np.trapezoid(a, x)


def to_gradient_of_a(x: np.ndarray, a_raw: np.ndarray, const: float):
    a = np.sqrt(np.gradient(a_raw, x) ** 2)
    a = savgol_filter(a, 10, 2)

    artificial_mass = mass(x, np.full_like(x, const))
    initial_a_mass = mass(x, a)
    a /= initial_a_mass
    a += const
    a /= 1.0 + artificial_mass
    return a


def to_density(x: np.ndarray, a_raw: np.ndarray, const: float):
    artificial_mass = mass(x, np.full_like(x, const))
    a_shift = a_raw.min()
    a = a_raw - a_shift
    initial_a_mass = mass(x, a)
    a /= initial_a_mass
    a += const
    a /= 1.0 + artificial_mass
    return a


def to_local_support_density(x: np.ndarray, a_raw: np.ndarray, const: float):
    artificial_mass = mass(x, np.full_like(x, const))

    a_shift = np.median(a_raw)
    a = a_raw - a_shift
    a = a * a
    if mass(x, a) < 1e-8:
        a += np.ones_like(a)
    a /= mass(x, a)
    a += const
    a /= 1.0 + artificial_mass
    return a


def to_hess_reg(x: np.ndarray, a_raw: np.ndarray, const: float):
    artificial_mass = mass(x, np.full_like(x, const))

    da = np.gradient(a_raw, x)
    dda = np.gradient(da, x)

    dda = dda * dda
    a = dda / mass(x, dda)
    a += const
    a /= 1.0 + artificial_mass

    return a


def ot1d_unbalanced_via_gradient_of_cumulative(
    x: np.ndarray,
    y: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    s: float,
):
    c_a = cumulative_trapezoid(a, x, initial=0)
    c_b = cumulative_trapezoid(b, y, initial=0)
    assert np.isclose(c_a[-1], 1.0, rtol=0.001)
    assert np.isclose(c_b[-1], 1.0, rtol=0.001)

    vals = np.linspace(0, 1, 10_000)
    c_a_inv = np.interp(vals, c_a, x)
    c_b_inv = np.interp(vals, c_b, y)
    c_i_inv = c_a_inv * (1 - s) + c_b_inv * s
    c_i = np.interp(x, c_i_inv, vals)
    interpolated_values = np.gradient(c_i, x)
    return interpolated_values


def ot1d_unbalanced_via_mappings(
    x: np.ndarray,
    y: np.ndarray,
    a_raw: np.ndarray,
    b_raw: np.ndarray,
    s: float,
    const: float = 1e-4,
):
    a = to_density(x, a_raw, const)
    b = to_density(y, b_raw, const)

    T, T_inv = ot1d_mappings(x, y, a, b)

    vals = cdi_1d(x, y, a, b, T, T_inv, s)
    return vals


def cdi_1d(x, y, a_raw, b_raw, T, T_inv, s):
    Ts = s * x + (1 - s) * T
    Ts_inv = (1 - s) * y + s * T_inv
    vals = (1 - s) * np.interp(Ts_inv, x, a_raw) + s * np.interp(Ts, y, b_raw)
    return vals


def direct_interpolation_1d(x, a_raw, T_inv, s):
    Ts_inv = (1 - s) * x + s * T_inv
    vals = np.interp(Ts_inv, x, a_raw)
    return vals


def ot1d_mappings(
    x: np.ndarray, y: np.ndarray, a: np.ndarray, b: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    Ca = cumulative_trapezoid(a, x=x, initial=0)
    Cb = cumulative_trapezoid(b, x=y, initial=0)
    assert np.isclose(Ca[-1], Cb[-1], rtol=0.001)
    assert np.isclose(Ca[-1], 1.0, rtol=0.001)
    vals = np.linspace(0.0, 1.0, 10_000)
    fa_inv = np.interp(vals, Ca, x)
    fb_inv = np.interp(vals, Cb, y)
    T = PchipInterpolator(vals, fb_inv)(PchipInterpolator(x, Ca)(x))
    W = PchipInterpolator(vals, fa_inv)(PchipInterpolator(y, Cb)(y))
    return T, W


def ot1d_cdi_unbalanced(
    x: np.ndarray,
    y: np.ndarray,
    a_raw: np.ndarray,
    b_raw: np.ndarray,
    s: float,
    const: float = 1e-4,
):
    a = to_local_support_density(x, a_raw, const)
    b = to_local_support_density(y, b_raw, const)

    T, T_inv = ot1d_mappings(x, y, a, b)

    vals = cdi_1d(x, y, a_raw, b_raw, T, T_inv, s)
    return vals


def w2_distance_1d(
    x: np.ndarray,
    y: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
) -> float:
    """Compute the squared W2 distance between two 1D densities.

    Reuses ``ot1d_mappings`` to obtain the quantile functions (inverse CDFs)
    and integrates their squared difference.

    Args:
        x: Grid points for density ``a``.
        y: Grid points for density ``b``.
        a: Probability density on ``x`` (must integrate to 1).
        b: Probability density on ``y`` (must integrate to 1).

    Returns:
        Squared W2 distance (take ``np.sqrt()`` for the actual distance).
    """
    T, T_inv = ot1d_mappings(x, y, a, b)

    # T is the forward transport map: T(x) = F_b^{-1}(F_a(x))
    # T_inv is the inverse transport map: T_inv(y) = F_a^{-1}(F_b(y))
    # Quantile functions on the original grids:
    # fa_inv = T_inv  # F_a^{-1} evaluated at y (via Cb interpolation)
    # fb_inv = T  # F_b^{-1} evaluated at x (via Ca interpolation)

    # We need both quantiles on a common probability grid.
    vals = np.linspace(0.0, 1.0, 10_000)
    # Re-interpolate the quantiles onto the common grid
    Ca = cumulative_trapezoid(a, x=x, initial=0)
    Cb = cumulative_trapezoid(b, x=y, initial=0)
    fa_inv_common = np.interp(vals, Cb, y)
    fb_inv_common = np.interp(vals, Ca, x)

    squared_diff = (fa_inv_common - fb_inv_common) ** 2
    w2_squared = np.trapezoid(squared_diff, vals)
    return float(w2_squared)
