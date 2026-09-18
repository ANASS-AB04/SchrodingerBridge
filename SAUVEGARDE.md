# Sauvegarde de fin de stage — 18 septembre 2026

Procédure pour préserver l'intégralité du travail avant restitution du PC pro.

## 1. Déjà sauvegardé — rien à faire

Le commit `3c12ca6` sur la branche `euler-update` est poussé sur
`git@github.com:ANASS-AB04/SchrodingerBridge.git` (compte GitHub personnel).

Il contient **tout ce qui est nécessaire pour finir le rapport** :

| Contenu | Détail |
|---|---|
| Code | `Euler/`, `SB/`, `slurm/`, tous les `.sbatch` |
| Rapport | `rapport/` — LaTeX, `refs.bib`, logos |
| Métriques | les 1744 `metrics.json` et les CSV `_sweep` (211 Mo) |
| Documentation | `CLAUDE.md` |

Les PNG et `.npz` d'`outputs/` sont volontairement ignorés par `.gitignore` :
ils se régénèrent depuis les bundles Euler.

## 2. Ce qui n'est PAS sauvegardé

| Contenu | Taille | Régénérable ? |
|---|---|---|
| `outputs/**/fields.npz` | 5,5 Go | Oui, mais seulement en relançant les jobs GPU |
| `outputs/**/*.png` | 26,8 Go (90 000 fichiers) | Oui, depuis `fields.npz` via `SB/plot.py` |
| `outputs/**/ffd_beta_*.npz` | 1,2 Go | Oui — simples caches de registration |
| `meshes/` | 249 Mo | Oui, via `Euler/generate_mesh.py` |
| `results/` (bundles Euler) | 33 Mo local, 158 Mo sur le cluster | Oui, en relançant les solveurs |
| Contexte Claude Code | 55 Mo | Non |

**Priorité** : les `fields.npz`. Tout le reste se reconstruit.

## 3. Transfert vers CALI3

Quota `/scratch` : **1 To**, dont 6,3 Go utilisés — la place ne manque pas.
Le home cluster (60 Go) est **plein**, ne rien y mettre.

### 3a. Les données essentielles d'abord (~6 Go)

```bash
ssh aaboufadel@cali3.unilim.fr 'mkdir -p /scratch/aaboufadel/backup-stage'

rsync -avz --partial --progress \
  --include='*/' --include='*.npz' --include='*.json' --include='*.csv' \
  --exclude='ffd_beta_*.npz' --exclude='*' \
  ~/GenerativePDE/outputs/ \
  aaboufadel@cali3.unilim.fr:/scratch/aaboufadel/backup-stage/outputs/

rsync -avz --partial --progress \
  ~/GenerativePDE/results/ ~/GenerativePDE/meshes/ \
  aaboufadel@cali3.unilim.fr:/scratch/aaboufadel/backup-stage/
```

### 3b. Les figures ensuite, si le temps le permet (~27 Go)

```bash
rsync -avz --partial --progress \
  --include='*/' --include='*.png' --exclude='*' \
  ~/GenerativePDE/outputs/ \
  aaboufadel@cali3.unilim.fr:/scratch/aaboufadel/backup-stage/outputs/
```

### 3c. Le contexte Claude Code (55 Mo)

**Révoquer d'abord la clé API OpenRouter** présente dans ton historique shell : elle apparaît en
clair dans `~/.bash_history` et donc dans les transcripts ci-dessous.

```bash
tar -C ~/.claude -czf /tmp/contexte-claude.tar.gz \
    projects/-home-anass-GenerativePDE plans
rsync -avz --progress /tmp/contexte-claude.tar.gz \
    aaboufadel@cali3.unilim.fr:/scratch/aaboufadel/backup-stage/
```

Contenu : 11 transcripts de conversation, le dossier `memory/` (7 notes de
contexte sur le projet) et les 3 fichiers de plan.

### 3d. Vérifier avant de rendre le PC

```bash
ssh aaboufadel@cali3.unilim.fr 'du -sh /scratch/aaboufadel/backup-stage/*'
```

## 4. Depuis le PC perso

```bash
git clone git@github.com:ANASS-AB04/SchrodingerBridge.git
cd SchrodingerBridge && git checkout euler-update
uv sync

rsync -avz --partial --progress \
  aaboufadel@cali3.unilim.fr:/scratch/aaboufadel/backup-stage/ ./backup-stage/
```

## 5. Avertissements

1. **Le compte CALI3 sera probablement désactivé** à la fin du stage. C'est le
   risque principal : personne ne récupère de données sur un compte fermé.
   **À demander à Mathias Truel aujourd'hui.** Si la réponse est « bientôt »,
   il faut transférer directement sur un disque externe plutôt que par le
   cluster.
2. **`/scratch` n'est pas sauvegardé** et les scratch se purgent après une
   période d'inactivité. Ce n'est un relais, jamais une archive.
3. **Ne jamais laisser une seule copie.** Tant que les données ne sont pas sur
   le PC perso, le cluster est un point de défaillance unique.
4. Les transcripts contiennent une clé API : dépôt privé uniquement, et
   révoquer la clé.
