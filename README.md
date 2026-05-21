# Integrated GNN-RL Based Fraud Detection System for Financial Transactions

An enterprise-grade, multi-tier hybrid artificial intelligence pipeline that fuses **Graph Neural Networks (GNNs)** with **Deep Reinforcement Learning (DRL)** to capture complex relational fraud patterns while optimizing real-time automated responses (`ALLOW`, `CHALLENGE`, or `BLOCK`).

This repository contains:

- `fraud.ipynb` → Core experimentation and model development
- `app.py` → Interactive fraud monitoring dashboard

---

## 🎯 Problem Statement

Traditional fraud detection systems analyze transactions independently using tabular features. This often misses:

- Shared device fraud rings
- Identity theft networks
- Merchant fraud clusters
- Multi-hop relational fraud structures

Static fraud classification also creates customer inconvenience due to unnecessary blocking.

---

## 💡 Proposed Solution

This framework integrates:

### Relational Intelligence Layer (Graph Neural Networks)

Transaction records are converted into a graph:

```
User → Merchant
User → Device
Merchant → Transaction
```

Graph Neural Networks capture hidden relationships:

- **GraphSAGE** → Learns neighborhood patterns and handles unseen entities
- **GCN** → Propagates risk across connected nodes
- **GAT** → Assigns attention to important fraud connections

---

### Decision Optimization Layer (Deep Reinforcement Learning)

A Deep Q Network (DQN) learns optimal business actions:

```
ALLOW
CHALLENGE (OTP / MFA)
BLOCK
```

Components:

- Deep Q Network (DQN)
- Target Network
- Experience Replay Buffer

Goal:

- Reduce fraud loss
- Reduce customer friction
- Optimize decision strategy

---

### Explainability Layer

SHAP explains model decisions.

Example:

```
Fraud Probability = 0.92

+0.31 → Merchant Risk
+0.22 → New Device
+0.18 → High Amount
-0.09 → Trusted History
```

---

## 🏗 System Workflow

Transaction Data

↓

Feature Engineering

↓

Graph Construction

↓

Graph Neural Network

↓

Risk Embedding Generation

↓

Deep Reinforcement Learning Decision Layer

↓

ALLOW / CHALLENGE / BLOCK

↓

SHAP Explainability

---

## 🛠 Technology Stack

### Machine Learning

- Scikit-learn
- PyTorch
- PyTorch Geometric

### Reinforcement Learning

- Gymnasium
- Deep Q Network (DQN)

### Explainable AI

- SHAP

### Data Processing

- Pandas
- NumPy

### Dashboard

- Streamlit
- Plotly

---

## 📊 Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Average Precision Score

---

## 🚀 Run Locally

Clone repository:

```bash
git clone https://github.com/Varshaa-gitHub/GNN-RL-based-Fraud-Detection.git
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run dashboard:

```bash
streamlit run app.py
```

---

## 📌 Key Features

✅ Relational fraud detection

✅ Graph Neural Network modeling

✅ Reinforcement Learning policy optimization

✅ Explainable AI diagnostics

✅ Interactive Streamlit dashboard

---

## 🌍 SDG Alignment

Supports:

**UN SDG 9 — Industry, Innovation and Infrastructure**

Improves digital payment security and financial infrastructure resilience.
