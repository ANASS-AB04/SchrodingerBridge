# Contexte Claude Code — sauvegarde de fin de stage

Ce dossier préserve le contexte accumulé pendant le stage, pour pouvoir
reprendre le travail sur un autre poste sans repartir de zéro.

## `memory/`

Sept notes de contexte écrites au fil du stage, chargées automatiquement au
démarrage d'une session Claude Code. Ce sont les conclusions non déductibles du
code seul :

| Note | Contenu |
|---|---|
| `sb-does-not-beat-ffd-baseline.md` | Le pont perd contre la registration FFD à tout γ et tout maillage, et pourquoi |
| `aoa-study-findings.md` | L'axe d'incidence est dominé par l'amplitude, SB y perd contre l'interpolation linéaire |
| `mesh-resolution-study-diamond.md` | L'échelle de raffinement h, le piège géométrique 5°/10° |
| `naca0012-bow-shock-underresolved.md` | Le sawtooth des C_D/C_L de référence, standoff à 1,3–5 cellules |
| `mach-range-physics-limits.md` | Les trois régimes physiques et leurs bornes |
| `cali3-storage.md` | Quotas du cluster |
| `MEMORY.md` | Index chargé à chaque session |

Pour les réutiliser sur un autre poste, copier `memory/` vers
`~/.claude/projects/<chemin-du-projet-encodé>/memory/`.

## Fichiers de plan

Trois plans de travail détaillés, dont `ok-je-veux-faire-glistening-shore.md`
qui contient **le plan complet du rapport de stage** : structure en 8 parties
avec budget de pages, annexes, placement de chaque référence bibliographique, et
la méthode de rédaction partie par partie.

## Transcripts — non inclus

Les 11 transcripts de conversation (55 Mo) ne sont pas dans ce dépôt. Ils
contiennent une clé API apparue dans un affichage de `~/.bash_history`, et plus
généralement l'intégralité des échanges. Pour les récupérer en les nettoyant :

```bash
mkdir -p transcripts
for f in ~/.claude/projects/-home-anass-GenerativePDE/*.jsonl; do
  sed -E 's/sk-(or|ant)-[A-Za-z0-9_-]{16,}/[CLE-SUPPRIMEE]/g' \
      "$f" > "transcripts/$(basename "$f")"
done
grep -rcE 'sk-(or|ant)-[A-Za-z0-9_-]{16,}' transcripts/   # doit donner 0 partout
```

### Conversion en Markdown lisible

Le script `jsonl_to_md.py` fait mieux que le `sed` ci-dessus : il produit un
`.md` par conversation, scrube les clés, et tronque les sorties d'outils (qui
représentent l'essentiel des 55 Mo et n'ont aucune valeur pour reprendre le
travail), tout en conservant le texte des échanges intégralement. Il refuse
d'écrire un fichier où une clé subsisterait.

```bash
python3 contexte-claude/jsonl_to_md.py
# option : --max-tool-chars 5000 pour garder plus de sortie d'outils
```

Les `.md` atterrissent dans `contexte-claude/transcripts/`. Relire avant de
committer, puis `git add contexte-claude/transcripts`.

Révoquer la clé OpenRouter dans tous les cas : elle reste dans l'historique
shell du PC rendu.
