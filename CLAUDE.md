# Notes for Claude Code sessions in this repo

Sessions run on a remote HPC inside Docker containers that get replaced. Only what is
written under `/home/iqbal` (NFS: this repo) or `/netscratch/naeem` survives a container.

## In a new container, first

`/root` is a tmpfs: the `pose` conda env, Claude Code's memory folder and the session
scratchpad from the last container are gone.

1. Restore the memory folder (it lives on NFS; `/root/.claude/...` only links to it):

   ```bash
   mkdir -p /root/.claude/projects/-home-iqbal-blender-leaf-generator
   [ -L /root/.claude/projects/-home-iqbal-blender-leaf-generator/memory ] || \
     { rmdir /root/.claude/projects/-home-iqbal-blender-leaf-generator/memory 2>/dev/null; \
       ln -s /home/iqbal/.claude/projects/-home-iqbal-blender-leaf-generator/memory \
             /root/.claude/projects/-home-iqbal-blender-leaf-generator/memory; }
   ```

   Until then, read `/home/iqbal/.claude/projects/-home-iqbal-blender-leaf-generator/memory/MEMORY.md` directly.
2. `ls /root/miniconda3/envs` -- if `pose` is missing, run `./setup_env.sh` (in the background:
   ~15 min; gsplat then compiles its CUDA kernels on the first render, ~5 min). Its final
   `[FAIL] sam3 clean-up` line only matters when SAM3 runs (`./setup_env.sh --with-sam3`).
3. Read `docs/apex-plan.md` section 0: the current state of the work, numbers, next steps,
   how to run everything.

## Running the whole pipeline (since 2026-10-13)

`./run_pipeline.sh <dataset>` runs P1 -> P6, then `scale p4m p4g p5x s2d apex`, ending in a
Blender bundle at `<dataset>/plant/apex/bundle/`. Every phase writes a stamp
(`<workdir>/stamps/`, `scripts/phase_stamp.py`): a phase whose stamp matches this checkout's
code, its inputs and its settings is kept, anything else runs -- `--dry-run` prints which and
why, `--rerun-from <phase>` forces, `./scripts/phase_stamp.py show <workdir>` lists them.
Many plants: `./batch_run_plants.sh <specimen>... --log-dir <dir>` (one by one by default).
Configs: the repo's `./pipeline.conf` (token), `/netscratch/naeem/blender_assets/pipeline.conf`,
the session's and the specimen's (`architecture = caulescent | rosette` per specimen).
Do not edit `run_pipeline.sh` while a run is going: bash reads it as it runs.
A P4m (COLMAP PatchMatch) killed mid-kernel can hang in uninterruptible exit for a long time;
`ps`/`pgrep` then block on it -- read `/proc/<pid>/status` instead.

## Standing facts

- Python: `/root/miniconda3/envs/pose/bin/python`, with `PYTHONPATH=src` from the repo root.
- Test plant: `P=/netscratch/naeem/blender_assets/06-10-2026-Naeem/vogelmeere/plant`, `D=$P/debug`.
- Experiments and outputs go to `$D/<experiment>/`, never into the repo (disk quota).
  Diagnostic scripts go into `scripts/` or next to their output in `$D/<experiment>/` --
  never only in the scratchpad, which dies with the container.
- No tuned thresholds: structural rules only; a needed number is measured and its source said.
- The user judges results in Blender on their own machine: every deliverable ends with a
  fresh dated bundle (`$D/blender_bundle/make_bundle.sh`) and the command to view it.
- `rm` on a shell-variable path is refused by Claude Code's safety check: use literal paths.
- Keep `docs/apex-plan.md` section 0 current at the end of a session, in plain words.

## To keep more across containers (the user's side)

Bind-mount `/root/.claude` from NFS when starting the container (e.g.
`-v /home/iqbal/.claude_root:/root/.claude`), which keeps memory *and* session transcripts;
Claude Code also reads its config directory from `CLAUDE_CONFIG_DIR` (check the Claude Code
docs before relying on it: it moves the login too).
