---
title: EV Agent Adnan
emoji: ⚡
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8501
---
# EV Charging Station AI Agent

### From Predictive Analytics to Agentic Intelligence

## Project Overview

This project represents the evolution of an EV Charging Station Classifier into a fully agentic AI system. Initially developed as a machine learning pipeline to predict whether a station supports Fast DC Charging, the system has been extended into an autonomous conversational agent capable of reasoning, retrieving domain knowledge, and executing predictive workflows.

The architecture follows the ReAct (Reasoning and Acting) paradigm, enabling the agent to interact with users, gather required inputs iteratively, and deliver insights supported by both machine learning models and a retrieval-augmented knowledge base.

---

## Key Features

* **Agentic Orchestration**
  Built using LangGraph, the system maintains conversational state and dynamically decides when to invoke tools.

* **Retrieval-Augmented Generation (RAG)**
  ChromaDB is used to store and retrieve EV infrastructure knowledge for contextual responses.

* **Ensemble Machine Learning**
  Combines RandomForest and XGBoost models to improve classification accuracy and recall.

* **Rate Limiting Mechanism**
  A 5-second cooldown prevents excessive API usage and protects system quotas.

* **Automated Credential Management**
  API keys are securely loaded from environment variables or Streamlit secrets.

---

## Technology Stack

| Layer               | Technology                |
| ------------------- | ------------------------- |
| Agent Framework     | LangGraph₹    |
| Core LLM            | Google Gemini (1.5-flash) |
| Vector Database     | ChromaDB                  |
| Machine Learning    | Scikit-learn, XGBoost     |
| Data Processing     | Pandas, NumPy             |
| Embeddings          | ONNX MiniLM-L6-V2         |
| Backend & Interface | Streamlit                 |
| Model Persistence   | Joblib, Pickle            |

---

## System Evolution

### Phase 1: Core ML Pipeline

* Developed baseline classification models using RandomForest and XGBoost.
* Addressed class imbalance using `class_weight='balanced'`.
* Established evaluation metrics including ROC AUC and F1-score.

### Phase 2: Optimization and Enhancement

* Introduced feature engineering techniques such as interaction terms (Latitude × Longitude) and polynomial transformations (Ports²).
* Applied hyperparameter tuning using `RandomizedSearchCV`.
* Built an ensemble model using weighted probability averaging to improve recall performance.

### Phase 3: Agentic Integration

* Encapsulated the ML pipeline as a LangChain tool.
* Integrated a RAG layer for dynamic knowledge retrieval.

# ⚡ EV Charging Station AI Agent
### From Predictive Analytics to Agentic Intelligence

## 🚀 Project Overview
This project represents the evolution of an EV Charging Station Classifier into a sophisticated **Agentic AI System**. Originally designed as a machine learning pipeline to predict 'Fast DC Charger' status, it now features a fully autonomous conversational agent capable of reasoning, researching domain-specific facts, and executing complex predictive workflows.

The system uses a **ReAct (Reasoning and Acting)** pattern to interact with users, gathering necessary data points sequentially and providing insights backed by both statistical models and a Retrieval-Augmented Generation (RAG) knowledge base.

---

## 🌟 Key Features
- **Agentic Orchestration**: Built with **LangGraph**, the agent maintains conversation state and intelligently decides when to use tools.
- **RAG Knowledge Base**: Uses **ChromaDB** to store and retrieve domain-specific EV infrastructure facts.
- **Ensemble ML Prediction**: Combines **RandomForest** and **XGBoost** models for high-precision station classification.
- **Rate Limiting**: Built-in 5-second cooldown to manage API usage and protect quotas.
- **Automated Key Management**: Automatically retrieves credentials from environment variables or Streamlit secrets.

---

## 🛠️ Technology Stack

| Layer | Technology |
| :--- | :--- |
| **Agent Framework** | LangGraph, LangChain |
| **Core LLM** | Google Gemini (1.5-flash) |
| **Vector Database** | ChromaDB (RAG) |
| **Machine Learning** | Scikit-learn, XGBoost |
| **Data Manipulation** | Pandas, Numpy |
| **Embeddings** | ONNX MiniLM-L6-V2 (Lightweight Local) |
| **Backend & UI** | Streamlit |
| **Model Persistence** | Joblib, Pickle |

---

## 📈 System Evolution

### Phase 1: Core ML Pipeline (Milestone 1)
- Developed baseline models using RandomForest and XGBoost.
- Implemented class imbalance handling (`class_weight='balanced'`) to address the rarity of Fast DC Chargers in the dataset.
- Established primary preprocessing and evaluation metrics (ROC AUC, F1-score).

### Phase 2: Optimized Solution (Milestone 2)
- **Advanced Feature Engineering**: Interaction terms (Lat x Lon) and squared terms (Ports²) to capture non-linear relationships.
- **Hyperparameter Tuning**: Optimized models via `RandomizedSearchCV`.
- **Ensemble Modeling**: Created a weighted probability-averaging ensemble for superior recall.

### Phase 3: Agentic Integration (Current)
- Wrapped the ML pipeline into a LangChain Tool.
- Integrated a RAG layer for "Knowledge-on-Demand".
- Implemented the LangGraph state machine to handle multi-turn information gathering.

---

## ⚙️ Setup & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/garariya/EV_charging_station_model.git
>>>>>>> docs/add-model-card
cd EV_charging_station_model
```

### 2. Install Dependencies
<<<<<<< HEAD

It is recommended to use a virtual environment.

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Model and Data Setup

Ensure the following files are available in the root directory:

* `rf_balanced_retrained_fe.joblib`
* `xgb_cs_retrained_fe.joblib`
* `scaler.joblib`
* `label_encoder.pkl`
* `optimal_threshold.pkl`

### 4. Run the Application

=======
It is recommended to use a virtual environment.
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Model & Data Setup
Ensure the following artifacts are in the root directory:
- `rf_balanced_retrained_fe.joblib`
- `xgb_cs_retrained_fe.joblib`
- `scaler.joblib`
- `label_encoder.pkl`
- `optimal_threshold.pkl`

### 4. Run the Application
>>>>>>> docs/add-model-card
```bash
streamlit run app.py
```

---

<<<<<<< HEAD
## Usage Guide

* **Initialization**
  Configure your `GOOGLE_API_KEY` via `.env` or Streamlit secrets before launching the application.

* **Knowledge Queries**
  Ask domain-related questions (e.g., differences between AC and DC charging) to utilize the RAG system.

* **Prediction Workflow**
  Request a prediction, and the agent will guide you through required inputs step-by-step.

* **Rate Limiting Behavior**
  If interactions occur too rapidly, the system will enforce a short delay before accepting the next request.

---

## Contributors

Developed as part of a collaborative effort focused on intelligent intervention in EV infrastructure systems.
=======
## 🤖 How to Interact with the Agent
1. **Automated Setup**: Once you have your `GOOGLE_API_KEY` in your `.env` or Streamlit Secrets, just start the app! No manual token entry required.
2. **Chat Naturally**: Ask questions like *"What is the difference between AC and DC charging?"* to trigger the RAG knowledge search.
3. **Run a Prediction**: Ask the agent to predict a station status, and it will guide you through the necessary inputs.
4. **Rate Limiting**: If you message too quickly, the system will ask you to wait a few seconds before your next interaction.

---

## 👥 Contributors
Developed as part of a team effort focusing on Intelligent Intervention in EV infrastructure.

**Maintained by**: [garariya](https://github.com/garariya)
>>>>>>> docs/add-model-card
