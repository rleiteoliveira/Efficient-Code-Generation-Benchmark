# 🚀 TiCoder-SLM: Efficient Test-Driven Code Generation with Local LLMs

An optimized, architecture-agnostic implementation of **Test-Driven Code Generation (TiCoder)** focused on efficiency, privacy, and zero-cost execution using Local Small Language Models (SLMs).

This repository refactors the original TiCoder logic into a clean architecture, enabling direct comparison between **Oracle-based (TiCoder)** and **Consensus-based (CodeT)** strategies using consumer hardware (e.g., RTX 3060) via LM Studio or the OpenAI API.

---

## 📊 Key Results

Our benchmarks show that Local SLMs can match the performance of proprietary Cloud models in test generation tasks, while **TiCoder significantly outperforms CodeT** in low-resource scenarios.

**Key Findings:**

- **Local vs. Cloud Parity:** The local Granite-Tiny model (~3GB VRAM) achieved **73.81%** reduction, statistically tied with **GPT-4o-mini (74.29%)** and the much larger **Qwen-30B**.
- **Strategy Efficiency:** With small sampling (N=20), **TiCoder** is far superior, reducing the search space by ~74%.  
  CodeT fails to filter incorrect candidates effectively unless sampling is very large (N>100).
- **Zero Cost:** Entire pipeline runs locally with no inference cost.

---

## 🛠️ Features

- **🧩 Agnostic Architecture:** Swap between local LLMs (LM Studio / OpenAI-compatible server) and Cloud APIs using an environment variable.
- **🧹 Clean Implementation:** Modular and Pythonic architecture.
- **📉 Benchmarking Suite:** Run experiments across datasets (HumanEval/MBPP) and auto-generate charts.
- **🧠 In-Context Learning:** Optimized few-shot CoT prompts guide small models to generate edge-case tests.

---

## 📂 Project Structure

```
.
├── data/                   # Generated candidates cache
├── data_history/           # JSON database of benchmark runs
├── datasets/               # Ground truth datasets (HumanEval/MBPP)
├── src/
│   ├── benchmark_runner.py
│   ├── generate_candidates.py
│   ├── plot_dashboard.py
│   ├── core/
│   │   ├── slm_manager.py
│   │   ├── oracle.py
│   │   └── codet.py
│   └── utils/
└── requirements.txt
```

---

## 🚀 Getting Started

### 1. Prerequisites

- Python 3.10+
- LM Studio (recommended for local execution)
- Git

### 2. Installation

```bash
git clone https://github.com/yourusername/ticoder-slm-local.git
cd ticoder-slm-local

python -m venv venv

# Windows
.env\Scripts\Activate

# Linux/Mac
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Configuration (`.env`)

Create a `.env` file:

#### --- OPTION A: LOCAL EXECUTION (Recommended) ---

```
SLM_BASE_URL="http://localhost:1234/v1"
SLM_MODEL="local-model-id"

# Only needed if generating new candidates via GPT-3.5+
OPENAI_API_KEY="sk-..."
```

#### --- OPTION B: CLOUD EXECUTION ---

```
# Comment out SLM_BASE_URL to use OpenAI cloud API
# SLM_BASE_URL="http://localhost:1234/v1"
SLM_MODEL="gpt-4o-mini"
```

---

## 🔬 How to Reproduce Results

### **Step 1 — Generate Candidate Cache**

Requires a stronger model (GPT-3.5/4):

```bash
python src/generate_candidates.py
```

### **Step 2 — Run the Benchmark**

```bash
python src/benchmark_runner.py --model_name "Local-Granite-3B" --candidates_file data/candidates_cache.json
```

### **Step 3 — Visualize**

```bash
python src/plot_dashboard.py
```

Charts are saved in `results_charts/`.

---

## 🧠 Methodology

### **TiCoder (Test-Driven User-Intent Formalization)**

- Generates discriminating tests where solutions disagree.
- Ground Truth (Oracle) gives correct output.
- Incorrect candidates are pruned.
- Works extremely well even with small sampling.

### **CodeT (Code Termination)**

- Generates tests and clusters candidates by outputs.
- Keeps majority cluster.
- Requires *large* number of candidates to work effectively (N>50).

---

## 📚 References

- **TiCoder** – Lahiri et al., Microsoft Research  
- **CodeT** – Chen et al.  
- **HumanEval** – Chen et al.  

---

## 📄 License

MIT License — see `LICENSE` file.
