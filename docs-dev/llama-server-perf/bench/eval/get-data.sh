#!/bin/bash
# Downloads the accuracy test data into $BENCH_OUT/eval/data and checks it against the files used on the B50.
#   wiki.test.raw (wikitext-2, KLD), mc/ (HellaSwag, Winogrande, TruthfulQA, ARC-Challenge, MMLU)
set -e
source "$(dirname "$0")/../config.sh"
D=$BENCH_OUT/eval/data
mkdir -p $D/mc
get() { [ -s "$2" ] || curl -fL "$1" -o "$2"; }

get https://raw.githubusercontent.com/klosax/hellaswag_text_data/main/hellaswag_val_full.txt $D/mc/hellaswag_val_full.txt
get https://huggingface.co/datasets/ikawrakow/winogrande-eval-for-llama.cpp/raw/main/winogrande-debiased-eval.csv $D/mc/winogrande-debiased-eval.csv
for f in truthful-qa-validation.bin arc-challenge-validation.bin mmlu-validation.bin; do
    get https://huggingface.co/datasets/ikawrakow/validation-datasets-for-llama.cpp/resolve/main/$f $D/mc/$f
done
if [ ! -s $D/wiki.test.raw ]; then
    curl -fL https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip -o $D/wikitext-2-raw-v1.zip
    unzip -o -j $D/wikitext-2-raw-v1.zip 'wikitext-2-raw/wiki.test.raw' -d $D
    rm $D/wikitext-2-raw-v1.zip
fi

cd $D && sha1sum -c <<'SUM'
ab84f5544b1224a8615f38509c71ff6b894b9959  mc/arc-challenge-validation.bin
ea3841618209fa89afd983d7eb2bf6f475071e74  mc/hellaswag_val_full.txt
8df13d9d6a267b2db4fa5969e1beaba3205214e2  mc/mmlu-validation.bin
3810cbd76574ed499cf901eca2fc518528124d24  mc/truthful-qa-validation.bin
9942c501df92d9dfb68bbefccec7079d5dad5c94  mc/winogrande-debiased-eval.csv
745366a9eea1154d35a90324e04aa4b02b7764e8  wiki.test.raw
SUM
