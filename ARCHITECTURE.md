# System Architecture

This document provides a high-level overview of the architectural components that make up the **EV Charging Station AI Agent**.

## Overview

The application is structured as an interactive Streamlit frontend communicating with a LangGraph-orchestrated backend Agent. The Agent utilizes two primary specific "tools": a RAG-based knowledge retriever and an Ensemble ML Prediction pipeline.

![Architecture Flow](https://upload.wikimedia.org/wikipedia/commons/thumb/1/1d/Architecture_diagram.png/800px-Architecture_diagram.png) *( illustrative generic diagram )*

## 1. Frontend Layer (UI)
**Technology:** Streamlit
- Provides the web-based conversational interface.
- Handles user inputs, renders chatbot messages iteratively.
- Implements application-level rules such as the 5-second rate limiter to throttle API usage.
- Securely retrieves and injects API keys (`GROQ_API_KEY`) from Streamlit Cloud Secrets or local `.env`.

## 2. Agent Orchestration Layer
**Technology:** LangChain & LangGraph (ReAct Pattern)
- Uses `create_react_agent` with a Llama-3 based Groq LLM.
- Analyzes user intentions to route queries dynamically.
- System prompt constrains the agent strictly to EV-related queries.
- Manages multi-turn conversations if information is missing (e.g., iteratively asking for Latitude, Longitude, Ports).

## 3. Knowledge Base (RAG) Layer
**Technology:** ChromaDB, HuggingFace Embeddings
- Initializes an in-memory vector database containing core domain facts about EV infrastructure (Level 2 vs Fast DC, CCS standards, etc.).
- `all-MiniLM-L6-v2` embedding model (run locally on CPU) converts facts into dense vectors.
- The `search_ev_knowledge` tool performs similarity search across vector representations to provide context back to the LLM.

## 4. Machine Learning & Prediction Layer
**Technology:** Scikit-Learn, XGBoost, Pandas
- Handles requests for Fast DC Station Probability.
- Includes a dedicated offline pipeline to preprocess input parameters (handling categoricals and feature scaling).
- Dynamically computes polynomial and interaction features (e.g. `latitude_x_longitude`) aligning exactly with training conditions.
- Uses an ensemble of **Random Forest** and **XGBoost**. Calculates probability average and compares to an optimal pre-tuned threshold.
- The `predict_fast_dc` tool wraps this entire ML pipeline and allows seamless call execution from the LangGraph agent.
