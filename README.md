# ADIA - Structural Breakpoint Detection Package

A modular package for detecting structural breakpoints in time series.

## Install

```bash
pip install -r requirements.txt
```

## Quick start

```python
from adia_refactored import process_time_series
predictors_df, metadata_df = process_time_series(
    input_path='data.parquet',
    output_pred_path='predictors.parquet',
    output_meta_path='metadata.parquet'
)
```

## CLI

```bash
python main.py --input data.parquet --output-pred predictors.parquet --output-meta metadata.parquet
```
