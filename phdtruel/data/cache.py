from pathlib import Path

import phdtruel


def get_cache_folder() -> Path:
    """
    Get the cache folder from the config file.
    """
    cache_dir = phdtruel.config["cache_dir"]
    if cache_dir is None:
        raise ValueError("cache_dir is not set in the config")
    return Path(str(cache_dir))


def set_cache_folder(folder: Path | str) -> None:
    """
    Set the cache folder in the config file.
    """
    phdtruel.config["cache_dir"] = Path(folder).expanduser().resolve()
