---
name: cali3-storage
description: CALI3 cluster (Univ. Limoges) storage — home 60 GB (was full 2026-09-11), run from /scratch/aaboufadel
metadata:
  type: reference
---

Cluster: `ssh aaboufadel@cali3.unilim.fr` (CALI3, Université de Limoges; help
svp-cali@unilim.fr; docs https://redmine.mcia.fr/projects/cluster-cali3/wiki/Stockage).
This workstation only has the SLURM client — `sinfo`/`squeue` cannot reach the cluster,
and it has no GPU.

- home `/home/aaboufadel`: 60 GB, backed up. **Hit the quota on 2026-09-11** (repo
  outputs ~35 GB, mostly ~50 PNGs per SB run, plus .venv and the uv cache).
- scratch `/scratch/aaboufadel` (also `~/scratch`): 1 TB, auto-created, NOT backed up,
  planned purge after 6 months without access.
- Advised 2026-09-11: move the repo to scratch and always `sbatch` from there (the
  scripts write relative to $SLURM_SUBMIT_DIR), set UV_CACHE_DIR on scratch, rebuild
  .venv after the move. Keep results/ (Euler bundles) archived locally — scratch has
  no backup.
