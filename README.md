# SMART

Core implementation for **Sample Margin-Aware Recalibration of Temperature Scaling**.

This repository is intentionally minimal. It contains the SMART calibrator, the
Charbonnier-SoftECE objective, small calibration metrics, and a runnable example
for applying SMART to existing logits.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```python
from smart import SMART
from metrics import expected_calibration_error

# val_logits: shape [n_val, n_classes]
# val_labels: shape [n_val]
# test_logits: shape [n_test, n_classes]
calibrator = SMART(
    hidden_dim=16,
    lr=5e-3,
    epochs=2000,
    loss="smooth_soft_ece",
    n_bins=15,
    seed=1,
)

calibrator.fit(val_logits, val_labels)
calibrated_probs = calibrator.calibrate(test_logits)
calibrated_logits = calibrator.calibrate(test_logits, return_logits=True)

ece = expected_calibration_error(probs=calibrated_probs, labels=test_labels, n_bins=15)
```

`val_logits` and `test_logits` can be either `numpy.ndarray` or `torch.Tensor`.
SMART returns the same broad type: NumPy inputs produce NumPy outputs, and tensor
inputs produce tensor outputs on the original device.

## Example

```bash
python example.py
```

The example creates synthetic over-confident logits, fits SMART on validation
logits, and reports ECE/NLL before and after calibration.

## Naming

We use **Charbonnier-SoftECE** and **SmoothSoftECE** interchangeably in this
codebase. `smooth_soft_ece`, `smoothsoftece`, `charbonnier_softece`, and
`charbonnier_soft_ece` are accepted aliases for the same objective.

## Citation

If you find this code useful, please cite our paper:

```bibtex
@inproceedings{guo2026smart,
  title={Sample Margin-Aware Recalibration of Temperature Scaling},
  author={Guo, Haolan and Tao, Linwei and Luo, Haoyang and Dong, Minjing and Xu, Chang},
  booktitle={International Conference on Machine Learning},
  year={2026}
}
```

## License

This project is released under the MIT License.
