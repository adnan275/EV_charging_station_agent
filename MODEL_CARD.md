# Model Card: EV Charging Station Agent

## Overview
This document summarizes the machine learning models and language agents responsible for predictions and conversational intelligence within the EV Charging AI Agent repository.

## 1. Machine Learning Classifiers
- **Models:** A weighted ensemble of `RandomForest` and `XGBoost`.
- **Task:** Binary Classification (Classifying if an EV location likely supports Fast DC Charging as opposed to Level 2/AC charging).
- **Ensemble Strategy:** Soft-voting average of output probabilities between models, evaluated against `optimal_threshold.pkl` to maximize F1-score due to imbalanced dataset.
- **Handling Imbalance:** The training process utilized class-weight balancing due to the rarity of Fast DC ports compared to standard AC ports.

## 2. Engineered Parameters
- The classification relies heavily on location and density geometry.
- Engineered parameters calculated dynamically at runtime:
  - Coordinate Space Intersections: `latitude * longitude`
  - Density Gradients: `ports * latitude`, `ports * longitude`
  - Non-linear relationships: `latitude_squared`, `longitude_squared`, `ports_squared`

## 3. Large Language Model (Agent) Orchestration
- **Framework:** LangChain and LangGraph `create_react_agent` state machine.
- **Core LLM:** `llama-3.3-70b-versatile` provided by Groq via the API.
- **Agent Limitations:** The agent is given strict system prompts to constrain discussions entirely within the domain of Electric Vehicles and Charging Infrastructure. Queries entirely outside this domain are gracefully declined.

## 4. Knowledge Retrieval (RAG component)
- **Vector Database:** `ChromaDB` initialized dynamically in-memory.
- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2` loaded locally via CPU processing.
- **Domain Scope:** Currently limited to static document constants regarding Fast DC hardware (50kW to 350kW), AC Charging standards (Level 2), and common regional connectors like CCS1, CCS2, and NACS.

## 5. Potential Biases
- **Geographic Bias:** As the models learn spatial features (lat/lon intersections), areas that had poor inclusion in the initial training dataset may experience less precise inferences.
- **Data Volatility:** Infrastructure changes daily. Extreme increases in Fast DC location density may eventually shift the overall threshold distribution away from what the pickled `optimal_threshold` expects.
