# Accès aux données du stage — pour Mathias Truel

## Où c'est

Sur CALI3 :

```
/scratch/aaboufadel/backup-stage/
├── outputs/     34 Go — résultats des interpolations
├── meshes/      260 Mo — maillages (dièdre, bosse, NACA0012, RAE2822, OA209)
├── results/     34 Mo — bundles Euler (snapshots de référence)
└── rapport/     583 Ko — rapport de stage LaTeX
```

Les droits sont déjà posés (ACL), rien à demander : `ssh` sur CALI3 et le
chemin est lisible directement.

## Ce qu'il y a dans `outputs/`

| Contenu | Volume | Régénérable ? |
|---|---|---|
| 1433 `fields.npz` — champs reconstruits par run | 5,5 Go | **Non**, sans relancer les jobs GPU |
| 1744 `metrics.json` — erreurs, coefficients, timings, diagnostics IPFP | 0,2 Go | Non |
| CSV de synthèse (`_sweep/sweep_summary_eig.csv`) | — | Oui, depuis les `metrics.json` |
| 89 994 figures PNG | 28,8 Go | Oui, depuis les `fields.npz` |

Arborescence : `<axe>/<cas>/iso/hessian/<mode>/<h>/[aoa]/[bande]/drift_<dérive>/gmin<γ>/`

## Le récupérer

**Consulter sur place** — rien à copier si c'est pour analyser sur le cluster.

**Copier dans son propre espace** (34 Go) :

```bash
rsync -a --partial --progress \
  /scratch/aaboufadel/backup-stage/ /scratch/mtruel/backup-anass/
```

**Rapatrier sur sa machine** :

```bash
rsync -a --partial --progress \
  mtruel@cali3.unilim.fr:/scratch/aaboufadel/backup-stage/ ./backup-anass/
```

Ne prendre que l'essentiel (6,1 Go, sans les figures) :

```bash
rsync -a --partial --progress \
  --include='*/' --include='*.npz' --include='*.json' --include='*.csv' \
  --exclude='*.png' --exclude='*' \
  mtruel@cali3.unilim.fr:/scratch/aaboufadel/backup-stage/outputs/ ./outputs/
```

## Le code

`github.com/ANASS-AB04/SchrodingerBridge`, branche `euler-update`. Contient le
solveur, le module SB, les scripts SLURM, le rapport, et les `metrics.json`
(les PNG et `.npz` en sont exclus — ils sont sur le scratch).

## Échéance

`/scratch` **n'est pas sauvegardé**, et le compte `aaboufadel` ferme dans un
mois environ. Passé ce délai ces données disparaissent. Si l'équipe veut les
conserver, il faut les déplacer vers un espace qui survive au compte — un
espace iRODS au nom de l'équipe, par exemple.
