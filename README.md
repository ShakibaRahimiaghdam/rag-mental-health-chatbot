# ==================================================
# 🧠 Mental Health Chatbot - Initial POC (Omdena Project)
# ==================================================

This repository contains the first version of a **Retrieval-Augmented Generation (RAG)** powered mental health chatbot built using LlamaIndex, FAISS, HuggingFace embeddings, and Groq's LLaMA 3.1 models. It was developed as part of an Omdena project to demonstrate how agentic workflows can improve support systems for mental health awareness and education.

---

## 📌 Features
- 🔄 Conversational assistant that uses contextual memory via RAG
- 🧠 Mental health dataset embedded using `BAAI/bge-small-en-v1.5`
- ⚙️ FAISS vector indexing for efficient similarity search
- 🤗 HuggingFace + LlamaIndex integration
- 🚀 Powered by Groq’s `llama-3.1-8b-instant` for real-time, cost-efficient LLM responses
- 📋 Streamlit web UI with feedback form (rating + optional comments)
- 💾 Saves feedback to `feedback_log.csv`

---

## 📁 Project Structure
```
mental_health_chatbot/
├── .env                 # Your API keys
├── .env.template        # Template with empty keys to copy from
├── app.py               # Main Streamlit application
├── agents.py            # Handles model + retriever + user query routing
├── rag_pipeline.py      # Data loading, chunking, vector indexing
├── feedback_log.csv     # Stores chat feedback (auto-created)
├── data/
│   └── labeled_with_severity_nuanced.csv  # Input data
├── vector_store/        # Auto-generated FAISS index files
├── ScreenShots/         # UI snapshots for documentation
├── README.md            # This file
├── requirements.txt     # All required dependencies
└── .gitignore           # Prevents committing environment files, indexes, logs
```

---

## 🚀 How to Run

### 1. 🔧 Setup your environment
```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### 2. 🔑 Set your API key
Copy `.env.template` ➝ `.env` and add your **GROQ_API_KEY**.
```env
GROQ_API_KEY=your-key-here
```

### 3. 🧠 Prepare your index (auto-created if missing)
The system will build the FAISS index the first time if it does not exist.
Make sure the `data/labeled_with_severity_nuanced.csv` file exists.

### 4. ▶️ Launch the chatbot
```bash
streamlit run app.py
```

---

## 📸 Screenshots
Screenshots are available under the `ScreenShots/` folder:
- `UI_1.png`: User introduction form
- `UI_2.png`: Chat window with assistant reply
- `UI_3.png`: Feedback and rating UI

---

## 📌 .env.template
```env
# .env.template
GROQ_API_KEY=
```

---

## ✅ TODOs for Future Iterations
- Add session memory & multi-turn conversation
- LangGraph/AutoGen for multi-agent routing
- Rerankers or hybrid retrieval
- Agentic workflows or planner support
- Additional model support (Gemini, OpenRouter)
- Improve UI accessibility and clarity

---

## 📦 Data Acknowledgement
I gratefully acknowledge **Md Kaif** for preparing and sharing the labeled mental health dataset used in this project.

---

## 🙏 Resources & Tools Used
- [LlamaIndex](https://github.com/jerryjliu/llama_index) for RAG orchestration
- [FAISS](https://github.com/facebookresearch/faiss) for vector search
- [HuggingFace](https://huggingface.co/BAAI/bge-small-en-v1.5) for embeddings
- [Groq API](https://console.groq.com/) for blazing-fast LLaMA 3.1 model inference
- [Streamlit](https://streamlit.io/) for interactive UI development

---

## 🙌 Contributors
This project is developed as part of a collaborative initiative under Omdena.