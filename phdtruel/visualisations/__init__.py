import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

import phdtruel
from phdtruel import printable_path
from phdtruel.fields.parameters import ParameterSet
from phdtruel.visualisations.contours import contour, contourf, create_norm, pcolormesh
from phdtruel.visualisations.figures import savefig, subplots
from phdtruel.visualisations.mappings import (
    get_jacobian_cmap_and_norm,
    overlay_constraint_status,
    overlay_control_points,
    overlay_sync_points,
    plot_deformed_checkerboard,
    plot_mapping_with_cp_overlay,
)
from phdtruel.visualisations.minimisation import (
    plot_solver_trace_panels,
)

__all__ = [
    "contourf",
    "contour",
    "pcolormesh",
    "create_norm",
    "get_jacobian_cmap_and_norm",
    "overlay_constraint_status",
    "overlay_control_points",
    "overlay_sync_points",
    "plot_deformed_checkerboard",
    "plot_mapping_with_cp_overlay",
    "plot_solver_trace_panels",
    "savefig",
    "subplots",
    "set_subfolder_name",
    "get_subfolder_name",
    "get_plot_subfolder",
    "name_generator",
]

my_rcParams: dict[str, Any] = {
    "figure.facecolor": "white",
    "figure.dpi": 100,
    "grid.alpha": 0.2,
    "font.size": 12,
    "text.usetex": True,
    "font.family": "serif",
    "grid.linestyle": "--",
    "grid.linewidth": 0.7,
    "grid.color": "black",
    "image.cmap": "YlOrRd",
    "animation.embed_limit": 128,
    "image.interpolation": "none",
}


def _latex_dependencies_available() -> bool:
    """Return whether Matplotlib usetex prerequisites are available.

    We require executables used by Agg + LaTeX workflow and key style files
    expected by Matplotlib's default TeX preamble.
    """
    required_bins = ("latex", "dvipng", "kpsewhich")
    if any(shutil.which(bin_name) is None for bin_name in required_bins):
        return False

    required_styles = ("type1cm.sty", "type1ec.sty")
    for style in required_styles:
        try:
            result = subprocess.run(
                ["kpsewhich", style],
                check=False,
                capture_output=True,
                text=True,
                timeout=2,
            )
        except (OSError, subprocess.SubprocessError):
            return False

        if result.returncode != 0 or not result.stdout.strip():
            return False

    return True


def _resolve_usetex_setting(default: bool = True) -> bool:
    """Resolve usetex from env override, otherwise auto-detect dependencies.

    Env var ``PHDTRUEL_MPL_USETEX`` accepts: ``1/true/yes/on``,
    ``0/false/no/off``, or ``auto``.
    """
    raw = os.environ.get("PHDTRUEL_MPL_USETEX", "auto").strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    if raw != "auto":
        return default
    return _latex_dependencies_available()


_usetex_enabled = _resolve_usetex_setting(default=True)
my_rcParams["text.usetex"] = _usetex_enabled

# Allways use the same rcParams
for key, value in my_rcParams.items():
    plt.rcParams[key] = value

logger = logging.getLogger("phdtruel.visualisations")
if not _usetex_enabled:
    logger.warning(
        "Matplotlib usetex disabled (missing LaTeX dependencies or "
        "PHDTRUEL_MPL_USETEX override). Falling back to mathtext."
    )


class VisualizationConfig:
    subfolder_name: str | None = None


def _register_slurm_plot_folder(plot_folder: Path) -> None:
    """Write plot folder path for the Slurm job wrapper to copy logs after exit."""
    job_id = os.environ.get("SLURM_JOB_ID")
    if not job_id:
        return
    repo_root = os.environ.get("PHDTRUEL_REPO_ROOT")
    if not repo_root:
        logger.debug(
            "SLURM_JOB_ID is set but PHDTRUEL_REPO_ROOT is missing; "
            "skipping Slurm log folder registration."
        )
        return
    marker_dir = Path(repo_root) / "artifacts" / "jobs"
    marker_dir.mkdir(parents=True, exist_ok=True)
    marker = marker_dir / f".plot_folder_{job_id}"
    marker.write_text(str(plot_folder.resolve()), encoding="utf-8")
    logger.debug("Registered Slurm plot folder marker: %s", printable_path(marker))


def set_subfolder_name(name: str) -> None:
    """
    Sets the global subfolder name to be used by get_plot_folder when no subfolder is specified.
    """
    VisualizationConfig.subfolder_name = name
    logger.info(f"Setting subfolder name to {name}")
    plot_folder = get_plot_subfolder(name)
    logger.info(f"New folder is {printable_path(plot_folder)}")
    _register_slurm_plot_folder(plot_folder)


def get_subfolder_name() -> str | None:
    """
    Returns the global subfolder name to be used by get_plot_folder when no subfolder is specified.
    """
    return VisualizationConfig.subfolder_name


def _get_notebook_path_vscode() -> Path:
    from IPython.core.getipython import get_ipython

    ip = get_ipython()
    if ip is None:
        raise RuntimeError("No IPython instance found")

    path = None
    if "__vsc_ipynb_file__" in ip.user_ns:
        path = ip.user_ns["__vsc_ipynb_file__"]
    if path is None:
        raise FileNotFoundError("Could not determine notebook path in VS Code")
    return Path(path).expanduser().resolve()


def _get_notebook_path() -> Path:
    import ipynbname

    try:
        return Path(ipynbname.path()).expanduser().resolve()
    except FileNotFoundError:
        return _get_notebook_path_vscode()


def get_plot_subfolder(subfolder_name: Path | str | None = None) -> Path:
    """
    Determine the folder where plots should be saved.

    Priority order for determining the subfolder name:
    1. Explicitly provided subfolder_name parameter
    2. Global VisualizationConfig.subfolder_name (set via set_subfolder_name())
    3. Notebook name (if executed in Jupyter/notebook environment)
    4. Script name (derived from sys.argv[0] stem, without extension)

    The base figures directory is determined from phdtruel.config["figures_dir"].

    Paths are normalized and validated to prevent directory traversal attacks.
    The target directory will be created if it does not already exist.

    Args:
        subfolder_name: Optional subfolder name or relative path. If contains path separators,
                       will be treated as a relative path and normalized.

    Returns:
        Path: The fully resolved plot folder path.

    Raises:
        ValueError: If "figures_dir" is not configured or if normalized subfolder escapes base directory.
    """
    # Determine the base figures directory from configuration.
    config_figures_dir = phdtruel.config.get("figures_dir")
    if config_figures_dir is not None:
        base_figures_dir = Path(config_figures_dir).expanduser().resolve()
    else:
        raise ValueError(
            "No 'figures_dir' key found in the configuration. "
            "Please set it in phdtruel.config or provide a subfolder_name."
        )

    # Determine the subfolder name if not explicitly provided
    if subfolder_name is None:
        if VisualizationConfig.subfolder_name is not None:
            subfolder_name = VisualizationConfig.subfolder_name
            logger.debug(f"Using globally configured subfolder name: {subfolder_name}")
        else:
            # Try to detect notebook execution
            try:
                notebook_path = _get_notebook_path()
                derived_notebook_name = notebook_path.stem
                subfolder_name = f"{derived_notebook_name}_figures"
                logger.debug(
                    f"Detected notebook execution; using subfolder: {subfolder_name}"
                )
            except Exception as e:
                # Fall back to script name derivation
                derived_script_name = Path(sys.argv[0]).stem
                subfolder_name = derived_script_name
                logger.debug(
                    f"Notebook detection failed ({type(e).__name__}); "
                    f"using script name derivation: {subfolder_name}"
                )

    # Normalize the subfolder_name to a relative path and validate it doesn't escape base directory
    subfolder_path = Path(subfolder_name)
    # Construct the final plot folder path and resolve it to detect any path traversal attempts
    plot_folder = (base_figures_dir / subfolder_path).resolve()

    # Validate that the resolved plot folder is still within the base directory
    try:
        plot_folder.relative_to(base_figures_dir)
    except ValueError:
        raise ValueError(
            f"Subfolder path '{subfolder_name}' would escape base figures directory "
            f"'{base_figures_dir}'. Please use a valid relative path."
        )

    # Create the directory if it does not exist.
    plot_folder.mkdir(parents=True, exist_ok=True)
    logger.debug(f"Plot folder: {printable_path(plot_folder)}")

    return plot_folder


def name_generator(
    *,
    method: str,
    test_case: str,
    parameters: list[ParameterSet],
    target_param: ParameterSet,
):
    active_params_str = ""
    for p_name in target_param.parameters:
        values = [v.as_dict()[p_name] for v in parameters]
        active_params_str += f"{p_name}{min(values)}_{max(values)}__"

    target_param_str = ""
    for v in target_param.values:
        target_param_str += f"{v}_"
    name = f"{method}__{test_case}__{active_params_str}{target_param_str}"
    name = name.rstrip("_")
    return name
