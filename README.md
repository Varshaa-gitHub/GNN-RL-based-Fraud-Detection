Here is the complete, raw content for your `README.md` file, formatted exactly as it should look on GitHub. You can copy and paste this text block directly into your editor:
```markdown
# Integrated GNN-RL Based Fraud Detection System for Financial Transactions

An enterprise-grade, multi-tier hybrid artificial intelligence pipeline that fuses **Graph Neural Networks (GNNs)** with **Deep Reinforcement Learning (DRL)** to capture complex multi-hop relational fraud networks while optimizing real-time autonomous operational responses (`ALLOW`, `CHALLENGE`, or `BLOCK`). 

This repository contains both the core laboratory development pipeline (`fraud.ipynb`) and the production-ready interactive monitoring control room dashboard application (`app.py`).

---

## 🎯 Project Objective & Problem Statement

### The Problem
Traditional fraud detection engines evaluate transaction logs as isolated, independent tabular vectors. This approach fails to capture complex, structural, and decentralized relational fraud rings (e.g., identity theft syndicates or shared-device farming vectors). Furthermore, standard binary classification alerts are static, resulting in significant customer friction due to false-alarm checkout blocks.

### The Proposed Solution
This framework introduces a dual-tier approach:
1. **Relational Topology Inversion Layer:** Converts transaction records into a heterogeneous graph network using `PyTorch Geometric`, applying inductive **GraphSAGE** local convolutions to generate structural context risk vectors.
2. **Dynamic Strategy Policy Layer:** Maps the spatial vectors into a bounded `Gymnasium` environment, training a dual-network **Deep Q-Network (DQN)** agent stabilized with an **Experience Replay Buffer** and decoupled **Target Network ($Q^-$)** to select the most cost-effective business intervention dynamically.

---

## 🏗️ End-to-End System Architecture

The technical execution pipeline is built across four tightly decoupled structural tiers:


```
+---------------------------------------------------------------------------------------+
|                              TRANSACTION DATA REGISTER                                |
|                   (User IDs, Merchant IDs, Device Signatures, Amounts)                |
+---------------------------------------------------------------------------------------+
|
v
+---------------------------------------------------------------------------------------+
|                    TIER 1: DATA INGESTION & STRUCTURAL INVERSION                      |
|  - Log Normalization, Standard Scaling, and Cyclical Temporal Encoding via Pandas     |
|  - Graph Link Inversion Matrix (User <-> Merchant <-> Device Edge Map)                |
+---------------------------------------------------------------------------------------+
|                                            |
v (Tabular Vector Stream)                    v (Relational Topologies)
+--------------------------------------------+    +-------------------------------------+
|         TABULAR ML BASELINES               |    |      TIER 2: PYG GRAPH TRANSDUCER   |
|  - Logistic Regression Inference           |    |  - Inductive GraphSAGE Convolution  |
|  - Random Forest Baseline Classifiers      |    |  - Relational Risk Vector Generation|
+--------------------------------------------+    +-------------------------------------+
|                                            |
+----------------------+---------------------+
|
v (Unified State Vector Observation)
+---------------------------------------------------------------------------------------+
|                    TIER 3: BOUNDED POLICY DECISION CORE (DQN)                         |
|  - Custom Gymnasium Environment Frame (State: [GNN Prob, Amount, Merchant Risk, Hour]) |
|  - Target Network (Q-) Parameters Synchronization & Experience Replay Buffering        |
+---------------------------------------------------------------------------------------+
|
v
+---------------------------------------------------------------------------------------+
|                    TIER 4: OPTIMIZED RESOLUTION & EXPLAINABILITY                      |
|  - Execution Engine Mitigation Vectors: ALLOW (Direct) | CHALLENGE (OTP) | BLOCK (Kill) |
|  - Local Game-Theoretic Attributions via Live SHAP TreeExplainer Waterfalls          |
|  - Production Dashboard Deployment Interface via Streamlit Server Infrastructure    |
+---------------------------------------------------------------------------------------+
```

---

## 🛠️ Tools and Technologies Used

### Core Mathematical & Environment Frameworks
* **Language Environment:** `Python 3.10+`
* **Deep Learning Engine:** `PyTorch Core (torch)`
* **Graph Neural Computations:** `PyTorch Geometric (PyG)`
* **Reinforcement Learning Framework:** `Gymnasium (Gym)`

### Tabular Baselines, Metrics & Explainable AI
* **Machine Learning Foundations:** `Scikit-learn`
* **Explainable AI (XAI):** `SHAP (SHapley Additive exPlanations)`
* **Data Core Processing:** `Pandas`, `NumPy`

### UI Deployment & Web Visualizations
* **Dashboard Engine:** `Streamlit Server Platform`
* **Interactive Web Graphics:** `Plotly Graph Objects (px, go)`
* **Static Evaluation Plots:** `Matplotlib`, `Seaborn`

---

## 🔬 Core Algorithms Implemented

* **GraphSAGE (Sample and Aggregate):** An inductive graph convolutional operator that scales to new, unseen entity nodes without requiring full graph retraining by dynamically gathering behavioral matrices from local neighborhoods.
* **Deep Q-Network (DQN):** Reaches policy convergence over the Bellman state equation, mapping real-time multidimensional features to specific action weights.
* **Target Network ($Q^-$):** Protects learning stability by isolating gradient weight adjustments from step target evaluations using a periodic parameter-sync clone.
* **Experience Replay Buffer:** Breaks chronological data dependencies by sampling completely random historical interaction micro-batches during gradient steps.
* **SHAP (TreeExplainer):** Reverse-engineers complex black-box logic via collaborative game-theoretic value modeling, mapping positive/negative feature contributions to specific alerts.

---

## 📊 Empirical Evaluation Summary Matrix

| Evaluation Metrics | Logistic Regression | Random Forest Base | Hybrid GraphSAGE Only | Integrated GNN-RL Model |
| :--- | :---: | :---: | :---: | :---: |
| **Classification Accuracy** | 93.24% | 96.51% | 97.15% | **98.42%** |
| **Precision Capture Ratio** | 0.6841 | 0.8852 | 0.9102 | **0.9415** |
| **Recall Sensitivity** | 0.5422 | 0.7614 | 0.8423 | **0.9268** |
| **Integrated F1-Score** | 0.6053 | 0.8182 | 0.8749 | **0.9331** |
| **ROC-AUC Area Metrics** | 0.8922 | 0.9455 | 0.9587 | **0.9891** |

---

## 🚀 Key Features

* **Induced Relational Awareness:** Maps spatial graph structures, identifying shared-device rings and account takeovers that standard machine learning ignores.
* **Cost-Minimized Friction Profiles:** Intelligently deploys step-up text validation challenges (`CHALLENGE`), avoiding unnecessary customer lockouts.
* **Auditor Transparency Controls:** Uses interactive local SHAP waterfall diagrams to break down the exact mathematical push/pull factors behind every security alert.
* **Interactive Control Room Application:** Includes a built-in interactive Streamlit UI featuring full parameter configurators, live graph rendering, and point-of-inference transaction simulations.

---

## 🏁 Getting Started & Setup Layout

### 1. Initialize Your Virtual Environment
```bash
# Clone the repository workspace
cd GNN_RL_Fraud_Industry_Starter

# Initialize isolated python workspace environment
python -m venv venv

# Activate virtual rails configuration (Windows)
.\\venv\\Scripts\\activate

# Activate virtual rails configuration (Linux/Mac)
source venv/bin/activate

```
2. Install Project Dependencies
```bash
pip install -r requirements.txt

```
Note: Ensure your `torch` version matches your native system CUDA configuration if compiling graph spaces over local GPU cores.
3. Launch the Production Dashboard App
```bash
streamlit run app.py

```
Open your local host address (defaults to `http://localhost:8501`) inside your browser to open the unified visual workspace dashboard control room.
---
🌿 UN SDG Alignment
This work aligns with UN Sustainable Development Goal 9: Industry, Innovation, and Infrastructure. By strengthening payment engine security grids against financial crime and protecting transactions, it directly improves the resiliency and reliability of global digital banking systems.
```

```