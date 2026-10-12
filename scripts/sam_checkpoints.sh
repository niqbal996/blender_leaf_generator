# Where the SAM model weights live. Sourced by run_pipeline.sh and
# setup_env.sh, which must agree: when they did not, setup_env.sh downloaded
# into the repo and run_pipeline.sh reported a checkpoint it could not find
# in a directory the user had already moved the weights out of.
#
# A 900 MB checkpoint does not fit in a repo checkout under a disk quota, so
# the weights have to be able to live somewhere else entirely. Two knobs, in
# priority order:
#
#   SAM2_CHECKPOINT     the SAM2 weights: a file, OR a directory to look in
#   SAM_CHECKPOINT_DIR  one directory holding every checkpoint this repo uses
#
# SAM2_CHECKPOINT accepting a directory is the fix for a real failure: the
# search used to test each candidate with `-f`, which a directory fails, so
# `export SAM2_CHECKPOINT=/netscratch/.../checkpoints/` was skipped without
# comment and the run died naming the repo path it had fallen back to.
#
# Requires REPO_ROOT to be set by the sourcing script.

SAM2_CHECKPOINT_NAME="sam2.1_hiera_large.pt"
SAM2_CHECKPOINT_URL="https://dl.fbaipublicfiles.com/segment_anything_2/092824/$SAM2_CHECKPOINT_NAME"

# Every path that could hold <name>, most specific first, one per line.
sam_checkpoint_candidates() {   # <name>
    local name="$1"
    if [[ -n "${SAM2_CHECKPOINT:-}" && "$name" == "$SAM2_CHECKPOINT_NAME" ]]; then
        # A path that does not exist yet is read as a file when it ends in
        # .pt and as a directory otherwise, so a mistyped directory is
        # reported as the directory it was meant to be.
        if [[ -d "$SAM2_CHECKPOINT" || "$SAM2_CHECKPOINT" != *.pt ]]; then
            echo "${SAM2_CHECKPOINT%/}/$name"
        else
            echo "$SAM2_CHECKPOINT"
        fi
    fi
    [[ -n "${SAM_CHECKPOINT_DIR:-}" ]] && echo "${SAM_CHECKPOINT_DIR%/}/$name"
    echo "$REPO_ROOT/checkpoints/$name"
    echo "$REPO_ROOT/third_party/sam2/checkpoints/$name"
    echo "$HOME/.cache/sam2/$name"
}

# Prints the first candidate that exists; returns 1 if none do.
find_sam_checkpoint() {   # [name]
    local name="${1:-$SAM2_CHECKPOINT_NAME}" c
    while IFS= read -r c; do
        [[ -n "$c" && -f "$c" ]] && { echo "$c"; return 0; }
    done < <(sam_checkpoint_candidates "$name")
    return 1
}

# Where a fresh download belongs: the same place the search looks first, so
# fetching and finding can never disagree again.
sam_checkpoint_dir() {
    if [[ -n "${SAM_CHECKPOINT_DIR:-}" ]]; then
        echo "${SAM_CHECKPOINT_DIR%/}"
    elif [[ -n "${SAM2_CHECKPOINT:-}" ]]; then
        # A path that does not exist yet is read as a file when it ends in
        # .pt and as a directory otherwise -- the download has to pick one,
        # and the extension is the only signal available before it lands.
        if [[ -d "$SAM2_CHECKPOINT" || "$SAM2_CHECKPOINT" != *.pt ]]; then
            echo "${SAM2_CHECKPOINT%/}"
        else
            dirname "$SAM2_CHECKPOINT"
        fi
    else
        echo "$REPO_ROOT/checkpoints"
    fi
}

# The paths actually tried, so the error names them rather than a fixed list
# that drifts from the search above.
sam_checkpoint_error() {   # <name>
    local name="$1" c
    echo "ERROR: SAM2 checkpoint not found. Looked for $name in:" >&2
    while IFS= read -r c; do
        [[ -n "$c" ]] || continue
        if [[ -d "$c" ]]; then
            echo "    $c   <-- is a directory, not the .pt file" >&2
        else
            echo "    $c" >&2
        fi
    done < <(sam_checkpoint_candidates "$name")
    echo "" >&2
    if [[ -n "${SAM2_CHECKPOINT:-}" ]]; then
        echo "  \$SAM2_CHECKPOINT is set to '$SAM2_CHECKPOINT' but no $name is there." >&2
    else
        echo "  To keep the weights off this disk:  export SAM_CHECKPOINT_DIR=/big/disk/checkpoints" >&2
    fi
    echo "  Fetch them with:  ./setup_env.sh --checkpoint-only" >&2
    echo "  (which downloads into \$SAM_CHECKPOINT_DIR / \$SAM2_CHECKPOINT when either is set)" >&2
}
