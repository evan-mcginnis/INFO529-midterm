#!/bin/bash --login
# The --login ensures the bash configuration is loaded,
# enabling Conda.

# Enable strict mode.
set -euo pipefail
# ... Run whatever commands ...

# Temporarily disable strict mode and activate conda:
set +euo pipefail
conda activate midterm 

# Re-enable strict mode:
set -euo pipefail

# exec the final command:
exec python predict-yield.py -d data/train.npz -i 350000  -p ./model -m midterm
#exec python predict-yield.py --help 
 
