# One task to rule them all: A closer look at traffic classification generalizability

**Intended for reasearch purposes only.**

We find that existing website fingerprinting and traffic classification methods break down when applied outside their original evaluation settings, as they rely heavily on context-specific assumptions. To explore this, we take three prior solutions from related tasks and apply each model to the others’ datasets. This reveals both dataset-specific and model-specific factors that cause each method to overperform in its own context.

To address the need for realistic evaluation, we build a framework using two recent TLS traffic datasets from large-scale networks. Our setup simulates a future scenario where SNIs are hidden in some networks but visible in others. The goal is to predict destination services in one network using a model trained on labeled data from another. This introduces a real-world distribution shift without concept drift. We show that even with plenty of labeled data, top-performing models only achieve 30–40% accuracy under this shift, with a simple 1-NN classifier performing surprisingly close.

This repository contains the deep learning models we used to classify network traffic. You can use it to test the models on your own data.

## Table of Contents
- [Installation](#installation)
- [Models](#models)
- [Usage](#usage)
- [Model Outputs](#model-outputs)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/one-task-traffic/traffic-generalization-study.git
cd traffic-generalization-study
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Set up the model output directories:
```bash
mkdir -p model_outputs/model_weights
mkdir -p model_outputs/full_model
```

## Models

### Model_First.py
A minimal Transformer with 1 attention head of size 3, 2 encoder blocks, and no embedding layer.

### Model_Full.py  
The original UWTransformer; includes 8 attention heads of size 256, 2 encoder blocks, and a dense embedding layer.

### Model_MoreBlocks.py
Similar to the First Model, but with 4 encoder blocks instead of 2.

### Model_NewRun.py
A compact model with 8 small heads of size 3, 2 encoder blocks, and no embedding. It also has small dense units (90), making it very lightweight (~12K parameters).

### Model_NoMB.py
A variant with 8 attention heads of size 256, like the Full Model, but without an embedding layer.

## Usage
1. **Data Preparation**: Prepare your traffic data in the required format with the following features:
   - **Packet sizes**: Numerical values representing the size of each packet.
   - **Directions**: Indicate the direction of each packet using the format `-1`, `0`, or `1` (e.g., `-1` for outgoing, `1` for incoming, `0` for unknown or N/A).
   - **Inter-arrival times**: Time intervals between consecutive packets.

   Ensure your dataset is cleaned and organized before proceeding to training.

2. **Training**: Open and run `model_implementation.ipynb` to:
   - Load the data
   - Configure model parameters
   - Train your chosen model architecture

## Model Outputs

Trained models are saved in the following structure:

```
model_outputs/
├── model_weights/              # Model weights only (.h5 files)
└── full_model/                 # Complete model architecture + weights
```

- `model_weights/`: Contains only the learned parameters for faster loading
- `full_model/`: Contains both architecture and weights for complete model restoration

Make sure both subdirectories exist before training begins.

## Citation

If you use this work in your research, please cite:

Elham Akbari, Zihao Zhou, Mohammad Ali Salahuddin, Noura Limam, Raouf Boutaba, Bertrand Mathieu, Stephanie Moteau, and Stephane Tuffin.  
**One task to rule them all: A closer look at traffic classification generalizability**  
*arXiv preprint* arXiv:2507.06430, 2025.  
https://arxiv.org/abs/2507.06430

```bibtex
@misc{akbari2025taskruleallcloser,
      title={One task to rule them all: A closer look at traffic classification generalizability}, 
      author={Elham Akbari and Zihao Zhou and Mohammad Ali Salahuddin and Noura Limam and Raouf Boutaba and Bertrand Mathieu and Stephanie Moteau and Stephane Tuffin},
      year={2025},
      eprint={2507.06430},
      archivePrefix={arXiv},
      primaryClass={cs.NI},
      url={https://arxiv.org/abs/2507.06430}, 
}
```
