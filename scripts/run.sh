#!/usr/bin/env bash
# run.sh — every analysis reported in the paper, both ion modes.
# Inputs go in raw/, see README. Roughly 25 minutes.

set -euo pipefail
PY=${PYTHON:-python3}
step () { printf '\n== %s\n' "$1"; }
for f in raw/ST002773_AN004513_Results.txt raw/ST002773_AN004514_Results.txt \
         raw/meta_AN004513.txt raw/meta_AN004514.txt raw/refmet.csv \
         raw/ME16_features.csv; do
  [ -f "$f" ] || { echo "missing input: $f"; exit 1; }
done
mkdir -p data out disc

# positive mode is HILIC, negative mode is C18. They are analyzed separately
# throughout, since a mass implies a different neutral molecule in each.
for spec in "hilic_pos AN004513 positive rp" "c18_neg AN004514 negative rp"; do
  set -- $spec; TAG=$1; AN=$2; MODE=$3

  step "$TAG  features and matched pairs"
  $PY parse_mwtab.py raw/meta_$AN.txt --outdir data/$TAG || true
  $PY prep_results.py raw/ST002773_${AN}_Results.txt \
      --metadata data/$TAG/metadata.csv --outdir data/$TAG --drop-unlabeled
  $PY reconstruct_pairs.py data/$TAG/metadata_aligned.csv \
      --out data/$TAG/metadata_paired.csv

  step "$TAG  graph"
  $PY annotate_to_graph.py --features data/$TAG/features.csv \
      --db raw/refmet.csv --mode $MODE --rt-tol 10 --min-presence 0.5 \
      --outdir out/$TAG

  step "$TAG  edge false discovery rate"
  $PY edge_fdr.py --data out/$TAG --src . --out edge_fdr_$TAG.csv

  step "$TAG  untargeted analyses"
  $PY discover.py --data out/$TAG --intensities data/$TAG/features.csv \
      --metadata data/$TAG/metadata_paired.csv --outdir disc/$TAG

  step "$TAG  published module ME16"
  $PY match_module.py --module raw/ME16_features.csv \
      --disc disc/$TAG/features.csv --mode $MODE --ppm 10 --rt-tol 30 \
      --out me16_$TAG.csv
  $PY module_connectivity.py --data out/$TAG --matched me16_$TAG.csv \
      --label "ME16 $TAG" --out conn_me16_$TAG.csv

  step "$TAG  node sets defined without the outcome"
  RPMODE=rp; [ "$TAG" = "hilic_pos" ] && RPMODE=hilic
  $PY find_series.py --data out/$TAG --mode $RPMODE --src . \
      --edge-fdr edge_fdr_$TAG.csv --out series_$TAG.csv
  $PY test_series.py --series series_$TAG.csv --data out/$TAG \
      --intensities data/$TAG/features.csv \
      --metadata data/$TAG/metadata_paired.csv --out series_tests_$TAG.csv
  $PY propagate.py --data out/$TAG --intensities data/$TAG/features.csv \
      --metadata data/$TAG/metadata_paired.csv --edge-fdr edge_fdr_$TAG.csv \
      --out prop_$TAG.csv
done

if [ -f raw/ME8_features.csv ]; then
  step "second published module ME8"
  $PY run_me8.py --csv raw/ME8_features.csv --disc-pos disc/hilic_pos \
      --disc-neg disc/c18_neg --outdir me8
  for TAG in hilic_pos c18_neg; do
    M=me8/me8_$TAG.csv
    [ -f "$M" ] && $PY module_connectivity.py --data out/$TAG --matched $M \
        --label "ME8 $TAG" --out conn_me8_$TAG.csv
  done
fi

printf '\ndone. Expected values are listed in the README.\n'
