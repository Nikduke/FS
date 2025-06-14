# Frequency Sweep Worst-Case Selection

This repository contains the Python scripts for analyzing impedance sweeps stored in an Excel workbook and identifying worst cases according to absolute and relative rules.

`FS_rules_single.py` is the maintained version of the tool. Previous scripts have been moved to the `archive/` directory for reference.
## Requirements

- Python 3.8+
- `numpy`, `pandas`, `matplotlib`, `scipy`

## Input Workbook

Provide `FS_sweep.xlsx` with sheets `R1`, `X1`, `R0`, and `X0`. Each sheet should use frequency as the index and cases as columns.

## Running

```bash
python FS_rules_single.py
```


Adjust configuration constants at the top of the script to change behaviour. In particular, `MAX_REL_CASES` controls how many cases are kept for each relative rule. `ENV_Z_SHIFT` sets the envelope-rule difference threshold used when absolute rules are enabled. You may override it by defining the environment variable before running, e.g. `ENV_Z_SHIFT=0.1 python FS_rules_single.py`. (The parameter has no effect unless `ENABLE_ABSOLUTE` is `True`.)

## Outputs

- `absolute_worst_cases.txt`
- `relative_worst_cases.txt`
- `worst_case_report.txt`
- `absolute_cases.png` – plots of absolute-rule winners (only when absolute rules run)
- `positive_sequence.png`
- `zero_sequence.png`
## Archive

Older scripts have been moved to the [`archive/`](archive/) directory to keep the repository tidy. They remain available for historical reference.
