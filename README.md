# BirdCLEF+ 2026 Experiments

PyTorch/EfficientNet experiments for BirdCLEF+ 2026 audio classification.

## Current best

- EfficientNet-B0 clip model trained on train_audio + labeled train_soundscapes
- epoch05/epoch06 ensemble
- temporal smoothing
- Kaggle public score: ~0.820

## Main experiments

- exp006: full train_audio + labeled soundscapes
- exp007: rare-class multicrop focal audio
- exp008: mined background/no-call windows
- exp009: EfficientNet SED-style temporal head

## Notes

Large files are intentionally excluded from git:

- competition data
- precomputed mels
- model checkpoints
- Kaggle tokens
- outputs