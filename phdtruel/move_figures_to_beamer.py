import shutil
from pathlib import Path

LOCAL_COPY_BEAMER_PATH = Path(__file__).parent.parent / "beamer_figures"
TRUE_BEAMER_PATH = Path("~/Documents/beamer").expanduser() / "ressources"

# Check if folder exists
if not LOCAL_COPY_BEAMER_PATH.exists() or not LOCAL_COPY_BEAMER_PATH.is_dir():
    raise FileNotFoundError(f"Le dossier {LOCAL_COPY_BEAMER_PATH} n'exsite pas.")


# Check if folder exists
if not TRUE_BEAMER_PATH.exists() or not TRUE_BEAMER_PATH.is_dir():
    raise FileNotFoundError(f"Le dossier {TRUE_BEAMER_PATH} n'exsite pas.")

# Check for lock file
if TRUE_BEAMER_PATH.joinpath("folder.lock").exists():
    raise FileExistsError(
        "The beamer folder is locked. Remove 'folder.lock' to allow the copy."
    )

LOCAL_BEAMER_PATH = Path(__file__).parent.parent / "notebooks" / "figures"


def main():
    shutil.copytree(LOCAL_BEAMER_PATH, LOCAL_COPY_BEAMER_PATH, dirs_exist_ok=True)
    shutil.copytree(LOCAL_COPY_BEAMER_PATH, TRUE_BEAMER_PATH, dirs_exist_ok=True)


if __name__ == "__main__":
    main()
