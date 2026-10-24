<div align="center">

# Mohamed NAJID

### Machine Learning Engineer · LLM Inference & Serving · MLOps

*I build ML systems that ship: custom PyTorch models, LLM fine-tuning and serving, reproducible training and inference pipelines.*

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0077B5?style=for-the-badge&logo=linkedin)](https://linkedin.com/in/mohamed-najid)
[![Email](https://img.shields.io/badge/Email-Contact_Me-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:Mohamednajid070@gmail.com)
![Location](https://img.shields.io/badge/Based_in-France_·_Remote_CET-555?style=for-the-badge)

<img src="https://skillicons.dev/icons?i=python,pytorch,sklearn,docker,aws,git,cpp,fastapi,linux&perline=9" />

</div>

---

## About

Machine Learning Engineer (MSc Artificial Intelligence, Université Claude Bernard Lyon 1; MSc Data Engineering). I work as an **AI & Data Engineer at FIROPA Group** (full-time) and as an **AI Engineer contractor for frontier AI labs** (Alignerr / Outlier.AI), where I create, solve and review ML engineering tasks with objective verifiers.

- Custom **PyTorch** models in production (predictive maintenance, time-series, NLP)
- **LLM** fine-tuning (LoRA / QLoRA), quantization and GPU serving (vLLM, SGLang, llama.cpp)
- Reproducible **MLOps**: configs, fixed seeds, MLflow, Docker, CI/CD
- Numerical debugging: shape/dtype failures, NaN/Inf, masking bugs, OOM
- Co-author of a published study on LLM-based financial sentiment (SentiTrade-HMA)

---

## Experience

| Role | Company | Period |
|---|---|---|
| AI & Data Engineer (full-time) | FIROPA Group | 08/2026 – Present |
| AI Engineer (Contractor) | Alignerr · Outlier.AI | 12/2025 – Present |
| Machine Learning Engineer (MLOps) | Alstom | 01/2026 – 08/2026 |
| Applied AI Engineer | Recruit.IA | 04/2025 – 07/2025 |
| Data Scientist | Orange Business | 09/2023 – 08/2024 |

---

## Tech Stack

**Languages:** ![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white) ![C++](https://img.shields.io/badge/C++-00599C?style=flat-square&logo=cplusplus&logoColor=white) ![SQL](https://img.shields.io/badge/SQL-4479A1?style=flat-square&logo=postgresql&logoColor=white) ![Bash](https://img.shields.io/badge/Bash-4EAA25?style=flat-square&logo=gnubash&logoColor=white)

**ML / DL / LLM:** ![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white) ![JAX](https://img.shields.io/badge/JAX-5E97F6?style=flat-square) ![Transformers](https://img.shields.io/badge/HF_Transformers-FFD21E?style=flat-square&logo=huggingface&logoColor=black) ![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white) ![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=flat-square) ![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat-square&logo=opencv&logoColor=white)

**Inference & Serving:** ![vLLM](https://img.shields.io/badge/vLLM-30A2FF?style=flat-square) ![SGLang](https://img.shields.io/badge/SGLang-111?style=flat-square) ![llama.cpp](https://img.shields.io/badge/llama.cpp-555?style=flat-square) ![ONNX](https://img.shields.io/badge/ONNX-005CED?style=flat-square&logo=onnx&logoColor=white) ![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white)

**MLOps & Cloud:** ![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat-square&logo=docker&logoColor=white) ![MLflow](https://img.shields.io/badge/MLflow-0194E2?style=flat-square&logo=mlflow&logoColor=white) ![GitHub Actions](https://img.shields.io/badge/GitHub_Actions-2088FF?style=flat-square&logo=githubactions&logoColor=white) ![AWS](https://img.shields.io/badge/AWS-232F3E?style=flat-square&logo=amazonaws&logoColor=white) ![Git](https://img.shields.io/badge/Git-F05032?style=flat-square&logo=git&logoColor=white)

---

## Featured Projects

### LLM Inference Optimization & Serving Platform
`JAX` `LoRA` `GGUF` `llama.cpp` `vLLM` `SGLang` `FastAPI` `Docker` `CI/CD`

End-to-end benchmarking and routing platform for LLM inference. LoRA fine-tuning in JAX, GGUF quantization (Q4–Q8) for llama.cpp, and GPU serving with vLLM and SGLang. Numerical parity validated with KL divergence on logits; TTFT, throughput and VRAM measured across traffic profiles; FastAPI router deployed with Docker and CI/CD.

---

### Smart-LSTM: Predictive Maintenance for Railway Traction Motors
`PyTorch` `LSTM` `Attention` `PINN` `C++` `MLOps`

Custom LSTM architecture with temporal attention and residual connections, physics-informed constraints and a real-time C++ inference pipeline, with leakage-free evaluation (splits by configuration, grouped cross-validation, out-of-domain analysis) and a reproducible training workflow (MLflow, Docker, CI/CD).

---

### HybridRAG-Pro: Production-ready Hybrid RAG
`LangChain` `LangGraph` `FAISS` `BM25` `FastAPI` `Streamlit` `Docker` `RAGAS`

Dense + sparse retrieval (FAISS + BM25 + RRF), cross-encoder reranking, HyDE / MultiQuery expansion and a LangGraph agentic router. RAGAS: Faithfulness 0.89, Relevancy 0.86, Precision 0.83, Recall 0.81. Full Docker stack in 3 commands.

[View repository](https://github.com/MeDusk/hybrid-rag-pro)

---

### SentiTrade-HMA: LLM-powered Algorithmic Trading *(co-authored paper)*
`Llama-2-7B` `QLoRA` `Temporal Fusion Transformer` `NLP`

Llama-2-7B fine-tuned with QLoRA (4-bit NF4, LoRA r=64) feeding a Temporal Fusion Transformer for multi-horizon forecasting. 99.0% accuracy vs 97.0% for FinBERT, with walk-forward validation and a Diebold-Mariano test.

[View repository](https://github.com/MeDusk/Sentitrade-HMA)

---

### Computer Vision: Segmentation, Detection & Tracking
`PyTorch` `U-Net` `YOLO` `Kalman` `ONNX`

U-Net with class-imbalance losses for medical image segmentation (Dice, IoU); YOLO detection with Kalman tracking exported to ONNX.

---

### Real-time Data Platform
`Streaming` `Orchestration` `ETL` `Docker`

End-to-end platform for real-time ingestion, transformation and streaming workflow orchestration, from ingestion through processing to analytical storage.

---

### Smart Connect 4: Deep RL on Physical Hardware
`PyTorch` `Dueling DQN` `Arduino`

Dueling DQN agent with experience replay and target network, trained by self-play and deployed on a physical Arduino-connected board.

[View repository](https://github.com/MeDusk/Smart-Connect4-IA)

---

### Musical Clustering: 32,828 Spotify Tracks
`K-means` `DBSCAN` `GMM` `PCA` `t-SNE` `UMAP`

4 algorithms compared with 7 metrics; commercial genres explain only 1–9% of acoustic variance.

[View repository](https://github.com/MeDusk/Data-MINING)

---

## Certifications

Micro1 Certified AI Engineer (May 2026) · IBM Build RAG Applications (Mar 2026) · IBM Machine Learning with Python V2 (May 2025) · AWS Educate ML Foundations (Apr 2025) · Oracle OCI AI Foundations Associate (Apr 2025)

---

<div align="center">

**Open to ML Engineering and applied research opportunities** · LLM systems · MLOps · Time-series

</div>
