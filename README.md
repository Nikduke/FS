# Frequency Sweep Worst-Case Selection

This repository contains a single script `FS_rules_Hzband_textreport_final.py` that analyzes impedance sweeps stored in an Excel workbook and identifies worst cases according to absolute and relative rules.

## Requirements

- Python 3.8+
- `numpy`, `pandas`, `matplotlib`, `scipy`

## Input Workbook

Provide `fs_seet.xlsx` with sheets `R1`, `X1`, `R0`, and `X0`. Each sheet should use frequency as the index and cases as columns.

## Running

```bash
python FS_rules_Hzband_textreport_final.py
```

Adjust configuration constants at the top of the script to change behaviour. In particular, `MAX_REL_CASES` controls how many cases are kept for each relative rule.

## Outputs

- `absolute_worst_cases.txt`
- `relative_worst_cases.txt`
- `worst_case_report.txt`
- `absolute_cases.png` – plots of absolute-rule winners (only when absolute rules run)
- `positive_sequence.png`
- `zero_sequence.png`
