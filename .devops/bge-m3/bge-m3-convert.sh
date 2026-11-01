#!/bin/sh
# Converts a BAAI/bge-m3 download (`hf download`) to the hybrid (dense + sparse) GGUF that the server image loads.
#
# input  /hf      read-only, one of:
#                 - the Hugging Face hub cache (~/.cache/huggingface/hub, or $HF_HOME/hub)
#                 - a single repo cache dir (.../hub/models--BAAI--bge-m3)
#                 - a plain directory from `hf download BAAI/bge-m3 --local-dir DIR`
# output /models  writable, gets bge-m3-hybrid-$BGE_OUTTYPE.gguf (skipped when it is already there)
#
# env  BGE_REPO (BAAI/bge-m3), BGE_REVISION (main: a branch, tag or commit in the cache),
#      BGE_OUTTYPE (f32: fastest on GPUs without fast fp16, e.g. Pascal; f16 halves the VRAM),
#      BGE_OUTFILE (bge-m3-hybrid-$BGE_OUTTYPE.gguf), BGE_FORCE=1 to convert again
set -eu

IN=${BGE_INPUT:-/hf}
OUT=${BGE_OUTPUT:-/models}
REPO=${BGE_REPO:-BAAI/bge-m3}
REV=${BGE_REVISION:-main}
OUTTYPE=${BGE_OUTTYPE:-f32}
OUTFILE=${BGE_OUTFILE:-bge-m3-hybrid-$OUTTYPE.gguf}

die() { echo "bge-m3-convert: $*" >&2; exit 1; }

case "$OUTTYPE" in f32|f16|bf16|q8_0) ;; *) die "BGE_OUTTYPE must be f32, f16, bf16 or q8_0" ;; esac
case "$OUTFILE" in */*|.*) die "BGE_OUTFILE must be a plain file name" ;; esac

# find the snapshot directory
if [ -f "$IN/config.json" ]; then
    SRC=$IN
else
    CACHE=$IN
    REPO_DIR="models--$(echo "$REPO" | sed 's|/|--|g')"
    [ -d "$CACHE/$REPO_DIR" ] && CACHE=$CACHE/$REPO_DIR
    [ -d "$CACHE/hub/$REPO_DIR" ] && CACHE=$CACHE/hub/$REPO_DIR
    [ -d "$CACHE/snapshots" ] || die "no $REPO download under $IN (run: hf download $REPO)"
    if [ -f "$CACHE/refs/$REV" ]; then
        SHA=$(cat "$CACHE/refs/$REV")
    else
        SHA=$REV
    fi
    case "$SHA" in *[!0-9a-f]*|"") die "bad revision '$SHA'" ;; esac
    SRC=$CACHE/snapshots/$SHA
fi

[ -f "$SRC/config.json" ] || die "$SRC: no config.json"
[ -f "$SRC/sparse_linear.pt" ] || die "$SRC: no sparse_linear.pt (the sparse head; download the whole repo)"
[ -f "$SRC/tokenizer.json" ] || [ -f "$SRC/sentencepiece.bpe.model" ] || die "$SRC: no tokenizer"
[ -f "$SRC/model.safetensors" ] || [ -f "$SRC/pytorch_model.bin" ] || die "$SRC: no model weights"

if [ -s "$OUT/$OUTFILE" ] && [ "${BGE_FORCE:-0}" != 1 ]; then
    echo "bge-m3-convert: $OUT/$OUTFILE already exists, nothing to do (BGE_FORCE=1 converts again)"
    exit 0
fi

# the cache snapshot is a tree of symlinks into ../../blobs: mirror it without the parts the converter does not read
# (the sentence-transformers files matter: 1_Pooling/ and modules.json give the CLS pooling, README.md the metadata)
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
for f in "$SRC"/* "$SRC"/.[!.]*; do
    [ -e "$f" ] || continue
    case "$(basename "$f")" in onnx|imgs|*.jpg|*.png|colbert_linear.pt|.git*|.cache) continue ;; esac
    ln -s "$(readlink -f "$f")" "$WORK/$(basename "$f")"
done
# a cache can hold both weight files for a revision only if it was mixed by hand, prefer safetensors (no pickle)
if [ -e "$WORK/model.safetensors" ]; then rm -f "$WORK/pytorch_model.bin"; fi

echo "bge-m3-convert: $SRC -> $OUT/$OUTFILE ($OUTTYPE)"
python3 /conv/convert_hf_to_gguf.py "$WORK" --outtype "$OUTTYPE" --outfile "$OUT/.$OUTFILE.tmp"
mv -f "$OUT/.$OUTFILE.tmp" "$OUT/$OUTFILE"
echo "bge-m3-convert: done"
