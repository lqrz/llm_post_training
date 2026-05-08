# LLM Post-Training

Notebook-based walkthroughs for three common LLM post-training techniques:

- **Supervised fine-tuning (SFT)** on instruction-style chat data.
- **Direct preference optimization (DPO)** with chosen/rejected preference pairs.
- **Online reinforcement learning with GRPO** on math reasoning examples.

The repository is organized as a set of lesson folders. Each folder contains a Jupyter notebook, a matching `helper.py`, and a pinned `requirements.txt` for that lesson.

## Repository Structure

```text
.
|-- supervised_fine_tuning/
|   |-- supervised_fine_tuning.ipynb
|   |-- helper.py
|   `-- requirements.txt
|-- direct_preference_optimisation/
|   |-- Lesson_5.ipynb
|   |-- helper.py
|   `-- requirements.txt
|-- online_reinforcement_learning/
|   |-- grpo.ipynb
|   |-- helper.py
|   `-- requirements.txt
`-- README.md
```

## Lessons

### 1. Supervised Fine-Tuning

Path: `supervised_fine_tuning/supervised_fine_tuning.ipynb`

This notebook compares a base chat model before and after SFT, then walks through a small local SFT run using:

- `trl.SFTTrainer`
- `HuggingFaceTB/SmolLM2-135M`
- `banghua/DL-SFT-Dataset`

The notebook also references pre-trained comparison models such as `Qwen/Qwen3-0.6B-Base` and `banghua/Qwen3-0.6B-SFT`.

### 2. Direct Preference Optimization

Path: `direct_preference_optimisation/Lesson_5.ipynb`

This notebook shows how preference pairs can steer an instruction model, using:

- `trl.DPOTrainer`
- `HuggingFaceTB/SmolLM2-135M-Instruct` for a lightweight local run
- `mrfakename/identity` and `banghua/DL-DPO-Dataset`

It also compares against DPO-trained Qwen checkpoints such as `banghua/Qwen2.5-0.5B-DPO`.

### 3. Online Reinforcement Learning with GRPO

Path: `online_reinforcement_learning/grpo.ipynb`

This notebook evaluates and trains a model on GSM8K-style math reasoning with a reward function that extracts final numeric answers, using:

- `trl.GRPOTrainer`
- `openai/gsm8k`
- `HuggingFaceTB/SmolLM2-135M-Instruct` for a lightweight local run

It also includes evaluation flow for a fully trained `banghua/Qwen2.5-0.5B-GRPO` checkpoint.

## Setup

Create a virtual environment from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install the requirements for the lesson you want to run:

```bash
pip install -r supervised_fine_tuning/requirements.txt
```

The three `requirements.txt` files currently pin the same core stack:

- `torch`
- `transformers`
- `datasets`
- `trl`
- `huggingface-hub`
- `pandas`
- `numpy`

If you want to run notebooks from Jupyter:

```bash
pip install jupyter
jupyter notebook
```

## Model and Dataset Access

The notebooks use Hugging Face datasets and model checkpoints. Large model files are not committed to this repository. The placeholder `models/` directories indicate where local snapshots can be placed.

Most notebooks reference models with paths like:

```python
model_name = "./models/HuggingFaceTB/SmolLM2-135M"
```

For those paths to resolve without edits, either:

- run the notebook from its lesson directory and place model snapshots under that lesson's `models/` folder, or
- update `model_name` in the notebook to a Hugging Face model ID, such as `"HuggingFaceTB/SmolLM2-135M"`, so `transformers` downloads it through the Hugging Face cache.

Some datasets or model checkpoints may require network access or Hugging Face authentication.

## Running on CPU vs GPU

Each notebook defines:

```python
USE_GPU = False
```

Keep this as `False` for CPU-only runs. Set it to `True` only when CUDA is available and the selected model fits in GPU memory.

The notebooks intentionally use small models and reduced dataset slices for local training. Larger Qwen checkpoints are used mainly to demonstrate the expected behavior after full training.

## Shared Helpers

Each lesson folder includes a `helper.py` with utilities for:

- loading a model and tokenizer
- applying a simple chat template when needed
- generating deterministic responses
- testing a model on example questions
- displaying small dataset samples

The helper files are currently duplicated across lesson folders so each notebook can be run independently.

## Suggested Order

1. Start with `supervised_fine_tuning/supervised_fine_tuning.ipynb`.
2. Continue with `direct_preference_optimisation/Lesson_5.ipynb`.
3. Finish with `online_reinforcement_learning/grpo.ipynb`.

This order follows the usual post-training progression: teach the model the desired response format with SFT, improve preferences with DPO, then optimize task-specific behavior with reinforcement learning.
