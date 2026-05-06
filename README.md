# SMART

PyTorch implementation of **Sample Margin-Aware Recalibration of Temperature Scaling**.

## Usage

```python
from smart import SMART

calibrator = SMART()
calibrator.fit(val_logits, val_labels)
calibrated_probs = calibrator.calibrate(test_logits)
calibrated_logits = calibrator.calibrate(test_logits, return_logits=True)
```

`val_logits` and `test_logits` can be NumPy arrays or PyTorch tensors with shape `[N, C]`.

## Example

```bash
python example.py
```

## Note

`Charbonnier-SoftECE` and `SmoothSoftECE` are used interchangeably in this repository.

## Citation

```bibtex
@inproceedings{guo2026smart,
  title={Sample Margin-Aware Recalibration of Temperature Scaling},
  author={Guo, Haolan and Tao, Linwei and Luo, Haoyang and Dong, Minjing and Xu, Chang},
  booktitle={International Conference on Machine Learning},
  year={2026}
}
```
