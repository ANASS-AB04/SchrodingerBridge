# Archivage sur iRODS (MCIA)

Troisième copie, pérenne : iRODS réplique sur deux sites en RAID6, quota 10 To
par défaut. Wiki : https://redmine.mcia.fr/projects/irods-v2/wiki

**Tout se lance depuis CALI3**, où les iCommands sont déjà installées et où les
34 Go sont déjà présents dans `/scratch/aaboufadel/backup-stage/`. Le PC pro
n'intervient pas.

## 1. Authentification (une seule fois)

```bash
ssh aaboufadel@cali3.unilim.fr
iinit
```

`iinit` demande hôte, port (1247), utilisateur, zone et mot de passe. Les
valeurs exactes sont sur les pages « Comptes Utilisateurs » et « ICommands » du
wiki. Vérifier ensuite :

```bash
ils          # doit lister ta collection racine
iquota       # doit montrer les 10 To
```

## 2. Préparer l'archive des figures

90 000 PNG poussés un par un seraient très lents, et iRODS est fait pour peu
d'objets volumineux plutôt que l'inverse. On en fait une archive unique ; les
données, elles, restent en arborescence pour rester consultables fichier par
fichier.

```bash
cd /scratch/aaboufadel/backup-stage
find outputs -name '*.png' > /tmp/liste-png.txt
wc -l /tmp/liste-png.txt          # doit donner 89994
tar -cf outputs-figures.tar -T /tmp/liste-png.txt
ls -lh outputs-figures.tar        # ~28,8 Go
```

## 3. Envoyer

```bash
imkdir -p stage2026/outputs
```

Les données, en arborescence (3198 fichiers, 6,1 Go). `-b` active le mode
bulk, nettement plus rapide sur beaucoup de petits fichiers ; `-K` calcule et
vérifie une somme de contrôle.

```bash
cd /scratch/aaboufadel/backup-stage
iput -b -r -K -P outputs/Mach_interpolation stage2026/outputs/
iput -b -r -K -P outputs/AoA_interpolation  stage2026/outputs/
iput -b -r -K -P outputs/1d_test_case outputs/2d_test_case outputs/deck \
                 outputs/warmstart stage2026/outputs/
iput -b -r -K -P meshes results rapport stage2026/
```

L'archive des figures, en un seul objet de 28,8 Go. `-X` crée un fichier de
reprise : si la connexion tombe, relancer la même commande reprend où ça s'est
arrêté.

```bash
iput -K -P -T -X /tmp/irods-restart.txt outputs-figures.tar stage2026/
```

## 4. Vérifier

```bash
ils -lr stage2026 | tail -20
ichksum -r stage2026            # recalcule et compare toutes les sommes
iquota
```

Contrôles chiffrés, à comparer avec la source :

```bash
ils -lr stage2026/outputs | grep -c 'fields.npz'    # doit donner 1433
ils -l  stage2026/outputs-figures.tar               # doit donner ~28,8 Go
```

## 5. Récupérer plus tard, depuis le poste personnel

Installer les iCommands, `iinit`, puis :

```bash
iget -r -K stage2026 ./stage2026
tar -xf stage2026/outputs-figures.tar
```

## Ordre de priorité

Si le temps manque, pousser dans cet ordre :

1. `outputs/**/fields.npz` et les `metrics.json` — 6,1 Go, **seule partie qui
   ne se régénère qu'en relançant les jobs GPU** ;
2. `meshes`, `results`, `rapport` — 294 Mo ;
3. `outputs-figures.tar` — 28,8 Go, entièrement régénérable depuis les
   `fields.npz`.

Ne rien supprimer de `/scratch` tant que `ichksum -r` n'est pas passé sans
erreur.
