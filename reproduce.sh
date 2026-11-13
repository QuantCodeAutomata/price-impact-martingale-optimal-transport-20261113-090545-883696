#!/usr/bin/env bash
# Runs the unit test suite and all three experiments of the PIMOT replication.
set -uo pipefail

python -m pytest -q
python -m exp.exp1
python -m exp.exp2
python -m exp.exp3
