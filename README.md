# Diverse 0-Shot Tracking (D0T)

This repository contains the code for the paper [Diverse and Effective Synthetic Data Generation for Adaptable Zero-Shot Dialogue State Tracking](https://aclanthology.org/2024.findings-emnlp.731/).

The domain-diverse **D0T** dataset can be found under `data/dsg5k/train` and is formatted as 3 CSV-of-JSON tables containing the dialogue turns (`turn.csv`), slot definitions (`slot.csv`), and slot-value pairs (`slot_value.csv`).

The main experiment script is `dextrous/experiment.py` (experiments presented in the paper launched this script as slurm jobs using launcher scripts under `dextrous/launch`).

