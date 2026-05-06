# Entity-Centric World Models: Interaction-Aware Masking for Causal Video Prediction

Official implementation of **Interaction-Aware JEPA (IA-JEPA)**, a self-supervised video representation learning framework designed to induce emergent physical intuition by targeting interaction interfaces.

## 🚀 Key Results

| Model Variant | Inductive Bias | CLEVRER Causal (MC) | CLEVRER Physical IQ | SSv2 (18k subset) |
| :--- | :--- | :---: | :---: | :---: |
| Baseline | Random Patch | 3.22% | 51.4% | 34.4% |
| **IA-JEPA (Ours)** | **Interaction-Aware** | **14.26%** | **82.1%** | **40.6%** |

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/santoshpaidi/IA-JEPA.git
cd IA-JEPA

# Install dependencies
pip install -r requirements.txt
```

## 📦 Data Preparation

This implementation supports the **CLEVRER** benchmark and **Something-Something V2**. 
1. Download the CLEVRER dataset.
2. Place video tensors and QA annotations in the `data/` directory.

## 🏋️ Training

To train IA-JEPA with the Interaction-Aware masking distribution:
```bash
python train.py --config configs/interaction_variant.yaml
```

## 📊 Evaluation

To evaluate a trained backbone using the Multimodal Reasoner probe:
```bash
python src/eval/evaluate_jepa.py \
    --variant interaction \
    --checkpoint path/to/checkpoint.pth \
    --batch_size 256
```

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
