# MaVEn: Multi-Granularity Hybrid Visual Encoding Framework

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![Pytorch](https://img.shields.io/badge/pytorch-1.11%2B-orange.svg)](https://pytorch.org/)

## Introduction

**MaVEn** (Multi-granularity Hybrid Visual Encoding) is a novel visual encoding framework designed for **Multimodal Large Language Models (MLLMs)** to enhance their performance in **multi-image reasoning** and **single-image visual comprehension** tasks. By combining **discrete visual tokens** for high-level semantic abstraction and **continuous representations** for fine-grained details, MaVEn achieves state-of-the-art performance on multiple benchmarks, bridging the gap between visual encoding and language understanding.

---

## Key Features

- **Hybrid Encoding Framework**: Combines discrete and continuous representations for comprehensive visual understanding.
- **Efficient Patch Reduction**: Reduces computational overhead while maintaining high performance using a dynamic patch reduction mechanism.
- **Versatile Applicability**: Excels in both **multi-image reasoning tasks** (e.g., DemonBench, SEED-Bench) and **single-image tasks** (e.g., VQA, MMBench).
- **Zero-Shot Capability**: Demonstrates strong zero-shot performance on multimodal benchmarks.

---

## Architecture Overview

MaVEn employs a **multi-granularity hybrid encoding** strategy:
1. **Discrete Encoding**: Captures high-dimensional, abstract semantics.
2. **Continuous Encoding**: Preserves fine-grained, low-level details.
3. **Patch Selector**: Dynamically selects relevant visual tokens based on task requirements.

![Architecture Overview](assets/architecture.png)

---

## Benchmarks and Results

MaVEn achieves state-of-the-art performance across various benchmarks:
- **DemonBench**: Superior multi-image reasoning, visual relation inference, and multi-modal cloze tasks.
- **SEED-Bench**: Exceptional video understanding in action recognition and procedure comprehension.
- **VQA**: Outperforms existing MLLMs like LLaVA-1.5, BLIP2, and Qwen-VL-Chat in single-image visual question answering.
- **MMBench**: Demonstrates significant gains in multimodal benchmarks.

For detailed results, refer to our [paper](https://arxiv.org/pdf/2408.12321).

---

## Installation

### Prerequisites
- Python 3.8+
- PyTorch 1.11+
- CUDA 11.3+ (optional for GPU acceleration)

### Steps

1. Clone the repository:
    ```bash
    git clone https://github.com/your-repo/MaVEn.git
    cd MaVEn
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Download pre-trained weights and place them in the `checkpoints/` directory:
    - [MaVEn Pretrained Weights](https://example.com/maven-weights)

---

## Usage

### Inference
Use the following script for inference on multi-image tasks:
```python
from maven import MaVEn

model = MaVEn.load_pretrained('checkpoints/maven_model.pth')
result = model.infer(images=['image1.jpg', 'image2.jpg'], question='What is common between these images?')
print(result)
```

### Training
Train MaVEn on a custom dataset:
```bash
python train.py --config configs/maven_config.yaml --data_path /path/to/data
```

### Evaluation
Evaluate MaVEn on benchmarks:
```bash
python evaluate.py --config configs/maven_config.yaml --checkpoint checkpoints/maven_model.pth
```

---

## Repository Structure

```plaintext
MaVEn/
├── configs/            # Configuration files
├── data/               # Data loaders and preprocessing
├── models/             # Model architecture
├── checkpoints/        # Pre-trained model weights
├── scripts/            # Helper scripts
├── train.py            # Training script
├── evaluate.py         # Evaluation script
└── README.md           # Project documentation
```

---

## Results Visualization

### Multi-Image Tasks
![Multi-Image Results](assets/multi_image_results.png)

### Single-Image Tasks
![Single-Image Results](assets/single_image_results.png)

---

## Citation

If you find MaVEn helpful in your research, please cite our paper:
```bibtex
@article{maven2024,
  title={MaVEn: Multi-Granularity Hybrid Visual Encoding Framework for Multimodal Large Language Models},
  author={Author Name and Others},
  journal={Arxiv preprint},
  year={2024}
}
```


## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---



**Happy coding with MaVEn! 🚀**
```
