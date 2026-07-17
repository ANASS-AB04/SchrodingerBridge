import shutil
from pathlib import Path

import click

from phdtruel.move_figures_to_beamer import (
    LOCAL_BEAMER_PATH,
    LOCAL_COPY_BEAMER_PATH,
    TRUE_BEAMER_PATH,
)


@click.group()
def cli():
    pass


@cli.command()
def copy_plots():
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
    click.echo("Copie des figures", nl=False)
    shutil.copytree(LOCAL_BEAMER_PATH, LOCAL_COPY_BEAMER_PATH, dirs_exist_ok=True)
    shutil.copytree(LOCAL_COPY_BEAMER_PATH, TRUE_BEAMER_PATH, dirs_exist_ok=True)
    click.echo(" effectuée.")


@cli.command()
def generate_symlinks():
    symlinks_to_create = []
    import phdtruel

    figures_dir = phdtruel.config["figures_dir"]
    if figures_dir is None:
        raise click.BadParameter("figures_dir is not set in the config file.")

    symlink_figures_directories = phdtruel.config["symlink_figures_dirs"]
    if symlink_figures_directories is None:
        raise click.BadParameter("symlink_figures_dir is not set in the config file.")

    figures_dir = Path(figures_dir).expanduser().resolve()

    if not isinstance(symlink_figures_directories, list):
        symlink_figures_directories = [symlink_figures_directories]

    if not figures_dir.exists() or not figures_dir.is_dir():
        raise click.BadParameter(f"Le dossier {figures_dir} n'exsite pas.")

    for dir in symlink_figures_directories:
        dir = Path(dir).expanduser()

        if dir.exists():
            if dir.is_symlink():
                click.echo(f"Le lien symbolique {dir} existe déjà et est recrée.")
                dir.unlink()
                symlinks_to_create.append(dir)
            else:
                raise click.BadParameter(f"Le dossier {dir} existe déjà.")
        else:
            symlinks_to_create.append(dir)
    for dir in symlinks_to_create:
        if not dir.parent.exists():
            raise click.BadParameter(f"Le dossier parent {dir.parent} n'exsite pas.")

    for dir in symlinks_to_create:
        dir.symlink_to(figures_dir)
        click.echo(f"Le dossier {dir} a été créé.")
