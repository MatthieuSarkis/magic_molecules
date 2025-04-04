# Magic of Molecules

This repository contains the code to reproduce the results of 'Are molecules magical? Non-Stabilizerness in Molecular Bonding'.

## Installation

To set up the project in a virtual environment and install the package, run the following commands to:

- Create a virtual environment
```bash
python3 -m venv .env
```

- Activate it
```bash
source .env/bin/activate
```

- Install the required dependencies
```bash
pip install -e .
```

## Run the script

To run the script for the main results, use the following command:
```bash
python src/main_results.py
```

If you also want to reproduce the results for the larger basis set:
```bash
python src/larger_basis_set.py simulate
python src/larger_basis_set.py plot
```

The scripts reproduces exactly the results presented in the paper.