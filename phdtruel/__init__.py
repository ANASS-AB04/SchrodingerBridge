import logging
import os
import subprocess
from datetime import datetime
from pathlib import Path

# os.environ["JAX_PLATFORM_NAME"] = "cpu"  # Must happen before any `import jax`
import jax
import numpy
import yaml

jax.config.update("jax_enable_x64", True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info(f"JAX is using device: {jax.devices()[0]}")
logger.info(
    f"Jax precision: {'float64' if jax.config.read('jax_enable_x64') else 'float32'}"
)

PLOT_ENABLED_IN_TESTS = False

SEED = 535
# logger.info(f"Using seed {SEED}")
numpy.random.seed(SEED)

CONFIG_FILE = "phdtruel_config.yaml"

home_dir = Path.home()

yaml_file = home_dir / CONFIG_FILE

default_config: dict[str, Path | str | None] = {
    "figures_dir": None,
    "symlink_figures_dir": None,
    "errors_db_path": None,
    "cache_dir": None,
    "data_dir": None,
    "nascar_mean_fields_path": None,
    "nascar_fields_path": None,
    "dafoam_fields_path": None,
    "bump_data_path": None,
    "generative_pde_raw_path": None,
}

config = default_config.copy()
if yaml_file.exists():
    with open(yaml_file, "r") as file:
        loaded_config = yaml.safe_load(file)
    if isinstance(loaded_config, dict):
        config.update(loaded_config)

for entry, value in config.items():
    if "path" in entry or entry.endswith("_dir"):
        if isinstance(value, str | os.PathLike):
            config[entry] = Path(value).expanduser()


def require_config_path(key: str) -> Path:
    """Return a configured path or raise if the key is missing from ~/phdtruel_config.yaml."""
    value = config.get(key)
    if value is None:
        raise ValueError(f"{key!r} is not set in ~/phdtruel_config.yaml")
    return Path(value)


def printable_path(string: str | Path) -> str:
    string = Path(string)
    return string.as_uri()


def get_code_identifier() -> str:
    """For reproducibility, return a string identifying the current code version.

    Returns:
        A string containing commit revision, branch name, date/time, and script name.
    """
    import sys

    # Get commit revision
    try:
        commit = (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=Path(__file__).parent,
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit = "unknown"

    # Get branch name
    try:
        branch = (
            subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=Path(__file__).parent,
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        branch = "unknown"

    # Get current date and time
    timestamp = datetime.now().isoformat()

    # Get script name - detect interactive environments
    script_name = "unknown"
    if sys.argv:
        argv0_name = Path(sys.argv[0]).name
        # Check for Jupyter/IPython interactive environments
        if argv0_name == "ipykernel_launcher.py":
            script_name = "jupyter-interactive"
        elif argv0_name.endswith(".py"):
            script_name = argv0_name
        else:
            script_name = argv0_name

    return (
        f"commit={commit}\nbranch={branch}\ntimestamp={timestamp}\nscript={script_name}"
    )


def parse_code_identifier(raw: str) -> dict[str, str]:
    """Parse the ``key=value`` lines produced by `get_code_identifier` into a dict.

    Args:
        raw: The multi-line string returned by `get_code_identifier`.

    Returns:
        A dict mapping each ``key`` to its (stripped) ``value``.
    """
    parsed: dict[str, str] = {}
    for line in raw.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", maxsplit=1)
        parsed[key.strip()] = value.strip()
    return parsed


logger.info(f"Code identifier:\n{get_code_identifier()}")
