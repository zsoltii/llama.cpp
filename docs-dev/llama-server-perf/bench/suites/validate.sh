#!/bin/bash
# Validation after a speedup (Vulkan, $VK_DEV): test-backend-ops, wiki QA (hallucination), speed with the
# Qwen sampling, story loop test, attachment test (the user's command, 24 seeds).  validate.sh [TAG]
# Long (~3 h with the attachment test); SKIP_ATTACH=1 skips it.  Accuracy levels 1-2: eval/kld.sh, eval/mc.sh.
source "$(dirname "$0")/../config.sh"
TAG=${1:-$(git -C "$REPO" rev-parse --short HEAD)}
R=$BENCH_OUT/validate-$TAG; mkdir -p $R
echo "=== $(date +%T) test-backend-ops (Vulkan$VK_IDX, full)"
GGML_VK_VISIBLE_DEVICES=$VK_IDX $VK_BIN/test-backend-ops -b Vulkan0 > $R/test-backend-ops.log 2>&1
grep -E "tests passed|backends passed|FAIL" $R/test-backend-ops.log | head -5
echo "=== $(date +%T) wiki QA"
LLAMA_MTP_DEFER_CATCHUP=1 $VK_BIN/llama-server -hf $MODEL_HF -c 128000 -fa on --parallel 1 -t 16 -tb 16 \
  -dev $VK_DEV --port 8080 -ctk q8_0 -ctv q8_0 $MTP -bs > $R/wikiqa-srv.log 2>&1 &
SRV=$!
until curl -sf localhost:8080/health >/dev/null; do sleep 2; kill -0 $SRV 2>/dev/null || { echo "server died"; break; }; done
python3 $BENCH/eval/wiki_qa.py --name $TAG --port 8080 2>&1 | tail -1
kill $SRV; wait $SRV 2>/dev/null
echo "=== $(date +%T) speed (Qwen sampling, MTP and no MTP)"
T=$TAG- $BENCH/suites/q-bench.sh ownv ownv-nomtp
echo "=== $(date +%T) story loop test (p1, 6 seeds)"
ONLY_P=1 $BENCH/loop-test/loop-run.sh vk-full vk-qwen > /dev/null 2>&1
python3 $BENCH/loop-test/loop-score.py $BENCH_OUT/loop-test/loop-L-vk-full.txt | tail -1
python3 $BENCH/loop-test/loop-score.py $BENCH_OUT/loop-test/loop-L-vk-qwen.txt | tail -1
if [ -z "$SKIP_ATTACH" ]; then
  echo "=== $(date +%T) attachment test (user's command, 24 seeds)"
  python3 $BENCH/../tests/attach-test.py --name $TAG-vk-user --bin $VK_BIN/llama-server --dev $VK_DEV --sampling qwen \
    --seeds $(seq -s, 7 30) --max-tokens 32000 --env LLAMA_MTP_DEFER_CATCHUP=1 | tail -1
fi
echo "=== $(date +%T) done"
