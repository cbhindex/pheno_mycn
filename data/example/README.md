# Synthetic example data

The files here are **synthetic** and exist only to illustrate the expected
formats — they contain **no patient data** and no real slide identifiers.

- `cv_splits/fold0.csv` — a minimal cross-validation split in the format read by
  `pheno_mycn/datasets/camel_data.py` (6 train, 2 val, 2 test slides with fake
  `SAMPLE####` ids and 0/1 MYCN labels).

To actually train or evaluate, point the config's `label_dir` at a directory of
real `fold0.csv … fold9.csv` files and `data_dir` at the matching
`<slide_id>.pt` tile-embedding tensors. See `../README.md` for the schema.
