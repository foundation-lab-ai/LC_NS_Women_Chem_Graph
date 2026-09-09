# Mass-difference graph analysis

Code accompanying *A mass-difference graph resolves a pre-diagnostic lung cancer
lipid signature into a desaturation and chain-elongation series in never-smoking
women*.

The method connects mass-spectral features whose exact-mass difference matches a
known biochemical transformation, so an edge means one reaction step and no
compound identity is required. Every analysis in the paper is reproduced by the
scripts below.

## Requirements

```
python >= 3.10
numpy pandas scipy networkx openpyxl
```

```bash
pip install numpy pandas scipy networkx openpyxl
```

## Data

Nothing is bundled. The study is public.

| File | Source |
|---|---|
| `ST002773_AN004513_Results.txt`, `ST002773_AN004514_Results.txt` | Metabolomics Workbench FTP, study ST002773 |
| `meta_AN004513.txt`, `meta_AN004514.txt` | Workbench mwTab, same study |
| `refmet.csv` | Workbench RefMet download, used for labeling only |
| `ME16_features.csv`, `ME8_features.csv` | Supplemental Tables 1 and 2 of Rahman et al., doi:10.1002/ijc.34929 |

Place them in `raw/`. The published tables carry footnotes inside the `m.z`
column, so rows are kept only when `m.z` parses as a number.

## Reproduce

```bash
bash run.sh
```

Roughly 25 minutes. The steps are below if you prefer to run them separately.
Positive and negative ion modes are analyzed independently throughout, since a
feature's mass implies a different neutral molecule in each and intensities are
not comparable across columns.

### 1. Features and matched pairs

```bash
python parse_mwtab.py raw/meta_AN004514.txt --outdir data/c18_neg
python prep_results.py raw/ST002773_AN004514_Results.txt \
    --metadata data/c18_neg/metadata.csv --outdir data/c18_neg --drop-unlabeled
python reconstruct_pairs.py data/c18_neg/metadata_aligned.csv \
    --out data/c18_neg/metadata_paired.csv
```

The deposit carries no matched-set identifiers. They are recovered by optimal
assignment within batch on age and injection order, giving 397 pairs.

### 2. Graph

```bash
python annotate_to_graph.py --features data/c18_neg/features.csv \
    --db raw/refmet.csv --mode negative --rt-tol 10 --min-presence 0.5 \
    --outdir out/c18_neg
```

Ionization mode is not cosmetic. Sodium and potassium adduct rules apply in
positive mode only, and using them in negative mode deletes real metabolites.

### 3. Edge reliability

```bash
python edge_fdr.py --data out/c18_neg --src . --out edge_fdr_c18.csv
```

Each transformation mass is displaced by an offset no chemistry produces, so
every edge the decoy draws is coincidental and the ratio to observed edges
estimates the false discovery rate.

### 4. Untargeted analyses

```bash
python discover.py --data out/c18_neg --intensities data/c18_neg/features.csv \
    --metadata data/c18_neg/metadata_paired.csv --outdir disc/c18_neg
```

Features, Louvain modules, connected feature pairs and reaction log-ratios, each
against the exact sign-flip null. Writes `null_max_t.npy` for the null of the
largest statistic.

### 5. Published modules

```bash
python match_module.py --module raw/ME16_features.csv \
    --disc disc/c18_neg/features.csv --mode negative --out me16_c18.csv

python module_connectivity.py --data out/c18_neg --matched me16_c18.csv \
    --label "ME16 C18" --out conn_me16_c18.csv

python run_me8.py --csv raw/ME8_features.csv --disc-pos disc/hilic_pos \
    --disc-neg disc/c18_neg --outdir me8
```

`match_module.py` tests a published feature list, which requires no multiplicity
correction because the list is fixed in advance. `module_connectivity.py`
compares the module with random feature sets of the same size drawn within mass
deciles. `run_me8.py` handles a module whose odds ratios run in both directions
and therefore uses direction concordance in place of a sign test.

### 6. Node sets defined without the outcome

```bash
python find_series.py --data out/c18_neg --mode rp --src . \
    --edge-fdr edge_fdr_c18.csv --out series_c18.csv

python test_series.py --series series_c18.csv --data out/c18_neg \
    --intensities data/c18_neg/features.csv \
    --metadata data/c18_neg/metadata_paired.csv --out series_tests_c18.csv

python propagate.py --data out/c18_neg --intensities data/c18_neg/features.csv \
    --metadata data/c18_neg/metadata_paired.csv --edge-fdr edge_fdr_c18.csv \
    --out prop_c18.csv
```

Homologous series are connected components of the graph restricted to one
transformation, retained when retention time follows position along the chain in
the direction the chemistry predicts. Propagation diffuses the statistic over
the graph, with each null diffused through the same operator.

### 7. Unannotated neighbors

```bash
python explore_unknowns.py --data out/c18_neg --disc disc/c18_neg/features.csv \
    --mode rp --seed-list seeds_c18.csv --max-delta-rt 20 --out unknowns_c18.csv
```

Proposes an identity for each unannotated feature adjacent to a curated seed and
checks the retention shift against the direction the transformation predicts.

## Files

| File | Purpose |
|---|---|
| `common.py` | Normalization, within-pair differences, sign-flip null, Benjamini-Hochberg |
| `parse_mwtab.py` | Read sample metadata from a Workbench mwTab file |
| `prep_results.py` | Split `mz_rt`, drop QC injections, filter on detection rate |
| `reconstruct_pairs.py` | Recover matched sets from age, batch and injection order |
| `annotate_to_graph.py` | Build the graph, flag redundant ions, attach node attributes |
| `edge_fdr.py` | False discovery rate of edges, by decoy transformations |
| `discover.py` | Feature, module, feature-pair and reaction-state tests |
| `match_module.py` | Test a published feature list against this analysis |
| `module_connectivity.py` | Module connectivity against mass-matched random sets |
| `run_me8.py` | A second published module, with direction concordance |
| `find_series.py` | Enumerate homologous series and validate by retention order |
| `test_series.py` | Test series as pre-specified node sets |
| `propagate.py` | Diffuse the statistic over the graph |
| `explore_unknowns.py` | Propose identities for unannotated neighbors |

Every script takes `--help`.

## Expected values

| Quantity | Value |
|---|---|
| Features after filtering | 6,618 positive and 8,674 negative |
| Matched pairs | 397, covering 794 of 838 samples |
| Mass-difference edges | 30,205 positive and 35,950 negative |
| Edge false discovery rate | 6% positive and 15% negative |
| ME16 recovery | 121 of 121 matched, 119 inversely associated |
| Untargeted analyses | none below FDR 0.20 |

## License

MIT.
