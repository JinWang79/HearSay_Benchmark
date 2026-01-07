# HearSay Benchmark: Do Audio LLMs Leak What They Hear?

## 📖 Introduction

While Audio Large Language Models (ALLMs) have achieved remarkable progress in understanding and generation, their potential privacy implications remain largely unexplored. **HearSay** is the first comprehensive benchmark designed to investigate whether ALLMs inadvertently leak user privacy solely through acoustic voiceprints.

Constructed from over 22,000 real-world audio clips, *HearSay* covers **eight sensitive attributes** including age, gender, health status, and income. Our extensive evaluation reveals critical vulnerabilities: models can infer private information from non-semantic audio with alarming accuracy (e.g., 92.89% for Gender), and advanced reasoning mechanisms further amplify these risks.

![Main Figure](./assets/main.png)
*(Figure 1: Overview of the HearSay Benchmark framework and key finding.)*

## 🛠️ Installation

### Basic Setup
To set up the environment for running the benchmark evaluation scripts, follow these steps:

```bash
# Create a new conda environment
conda create -n hearsay python=3.10

# Activate the environment
source activate hearsay  # Or: conda activate hearsay

# Install dependencies
pip install -r requirements.txt
```

### ⚠️ Note on Model Environments
The `requirements.txt` provided here contains the dependencies for the benchmark evaluation framework itself. **However, specific ALLMs (e.g., Kimi-Audio, MiniCPM-o-2.6) may require unique environments or specific versions of libraries to run locally.**

Please refer to the official documentation of each model to configure their respective inference environments.

## 📂 Project Structure

```text
HearSay/
├── blind_tests/               # Scripts for Blind Bias Rate 
│   ├── blind_test_kimi.py
│   ├── blind_test_minicpm.py
│   ├── ...
│   └── blind_test_step.py
│
├── dataset/                   # Dataset samples and labels
│   ├── audio/                 # Audio clips (wav/mp3)
│   └── label/                 # Ground truth labels (json/csv)
│
├── model_infer/               # Inference scripts for each ALLM
│   ├── gemini-infer.py
│   ├── kimi-infer.py
│   ├── meralion-infer.py
│   ├── minicpm-infer.py
│   ├── ...
│   └── step-infer.py
│
├── main.py                    # Main entry point for running evaluation
├── requirements.txt           # Project dependencies
```

## 🚀 Running the HearSay Benchmark

### Step 1: Environment Configuration
Create a `.env` file in the root directory and add your API keys (for proprietary models like GPT-4o, Gemini, or Qwen-API). The scripts will automatically load these keys using `python-dotenv`.



### Step 2: Model Path Configuration
Open `main.py` and modify the `MODEL_PATHS` dictionary to point to your local model checkpoints:

```python
# In main.py
MODEL_REGISTRY = {
    "kimi": {
        "path": "/path/to/your/Kimi-Audio",
        "handler": "model_infer.kimi-infer"
    },
    "step": {
        "path": "/path/to/your/StepAudio2",
        "handler": "model_infer.step-infer"
    },
    # ...
}
```
### Step 3: Run Evaluation
Use the following command to run the main privacy inference experiment:

```bash
python main.py --model [model_id] --prompt_type main
```

*   `--model`: Specify the target model ID (e.g., `minicpm`,   `step`).
*   `--prompt_type`: Set to `main` for the standard privacy inference task.

---

## 📊 Dataset Availability

**Full Dataset Release:**
To ensure ethical usage and privacy protection, the complete HearSay dataset (containing 22,000+ clips) is restricted to academic research purposes only. Please contact my email at jingw6956@gmail.com for further inquiries.
