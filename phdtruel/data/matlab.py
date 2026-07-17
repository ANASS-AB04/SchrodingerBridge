from pathlib import Path

import numpy as np
import scipy.io


def export_matlab(path: Path | str, **variables: np.ndarray) -> None:
    scipy.io.savemat(path, variables)
