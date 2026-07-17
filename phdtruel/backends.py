import numpy as np
from jax import numpy as jnp


def get_backend(*arrays):
    """Get appropriate backend (np or jnp) based on input arrays.

    Args:
        *arrays: Variable number of arrays to check

    Returns:
        Backend module (np or jnp), with priority to np if multiple types exist
    """
    if not arrays:
        return np  # Default to numpy if no arrays provided

    has_numpy = False
    has_jax = False

    for arr in arrays:
        if arr is None:
            continue

        arr_type = str(type(arr))
        if "numpy" in arr_type or "ndarray" in arr_type:
            has_numpy = True
        elif "jax" in arr_type or "Array" in arr_type:
            has_jax = True

    # Priority to numpy if both are present
    if has_numpy:
        return np
    elif has_jax:
        return jnp
    else:
        return np  # Default to numpy
