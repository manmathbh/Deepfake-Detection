# Attribution Notice

This repository is an engineered adaptation of the OpenAVFF implementation:

https://github.com/JoeLeelyf/OpenAVFF

The original project implements the architecture described in:

AVFF: Audio-Visual Feature Fusion for Video Deepfake Detection

https://arxiv.org/abs/2406.02951

The original repository and its authors retain credit for the underlying
deep learning architecture and original source files (such as those in
`src/utilities/`).

## Engineering Modifications in this Repository

This repository contains engineering modifications and experiments maintained
by Manmath Hatte ([@manmathbh](https://github.com/manmathbh)), focusing on:

- **Data Pipeline Engineering:** Reworked data generation scripts
  (`generate_csv.py`) to implement majority-class undersampling for the
  DFDC dataset.

- **Hardware Optimization:** Adjusted batch sizes and dataloaders to work
  within CUDA memory constraints on available GPU hardware.

- **Dynamic Augmentation:** Added video frame augmentation during training
  to improve robustness and reduce minority-class overfitting.
