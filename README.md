# Re-uploading quantum data

Implementation of the numerical experiments in **“Re-uploading quantum data: a universal function approximator for quantum inputs”**. A single signal qubit sequentially interacts with fresh copies of an input quantum state.

## Experiments

- **Polynomial fitting:** fit functions of a one-parameter qubit state with a restricted SU(2) + CNOT model.
- **Purity classification:** classify single-qubit mixed states by purity.
- **Entanglement-entropy classification:** classify two-qubit pure states by the entropy of a reduced state.
- **Bloch-sphere classification:** learn binary decision rules on pure single-qubit states.

## Setup and usage

Run individual experiments:

```bash
python main.py polynomial
python main.py purity --epochs 100
python main.py entanglement --device cuda
python main.py bloch --no-plots
```

## Reference

*Re-uploading quantum data: a universal function approximator for quantum inputs*, arXiv:2509.18530.
