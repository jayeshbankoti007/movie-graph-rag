# 🎬 Agentic Movie Graph RAG 

Movie Graph RAG is an **agentic retrieval-augmented generation (RAG) system** that fuses a **knowledge graph** with **vector similarity search** to enable reasoning-driven queries in the movie domain.

Traditional RAG pipelines only match embeddings.  
Movie Graph RAG adds **graph-based reasoning** so you can answer relationship-heavy questions like:

- “Which actors worked with Christopher Nolan who also collaborated with Tim Burton?”  
- “How is Tom Hanks connected to Christopher Nolan through co-actors?”  
- “Which 2003 movies share crew ties with Quentin Tarantino?”

---

## 🔑 Key Features

- **Graph + RAG Hybrid** — Combine FAISS vector search with NetworkX graph traversal for multi-hop reasoning.  
- **Agentic Workflow** — The agent plans → retrieves → reasons → explains.  
- **Explainability** — Every answer has a traceable reasoning path.  
- **Extensible** — The framework can scale beyond movies to any relational dataset (finance, sports, enterprise knowledge).  

---

## 📂 Project Structure

```
.
├── agentic_graph_rag.py   # Core agent orchestration
├── agentic_tools.py       # Modular tools for graph + vector queries
├── faiss_setup.py         # FAISS index build and search
├── network_setup.py       # Graph construction with NetworkX
└── README.md              # Documentation
```
👉 [[KAGGLE DATA CSVs](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset)]

---

## ⚡ Quickstart

### 1. Clone the repo
```bash
git clone https://github.com/jayeshbankoti007/movie-graph-rag.git
cd movie-graph-rag
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Build the graph + FAISS index
```bash
python network_setup.py
python faiss_setup.py
```

### 4. Test the agent
```bash
python testing_agent.ipynb
```

### 5. Run the agent
```bash
python agentic_graph_rag.py
```

---

## 🧠 Example Queries

```text
    "Find all movies where Christopher Nolan worked with actors who also worked with Tim Burton."
    "How is Tom Hanks connected to Christopher Nolan through co-actors?"
    "Which movies released in 2003 have crew members tied to Quentin Tarantino?"
```

---

## 🎥 Demo

I’ve recorded a **12-minute walkthrough** that explains:  
- Why Graph RAG is different from standard RAG  
- How the pipeline is structured  
- Live examples of reasoning across the movie graph  

👉 [[Youtube link](https://www.youtube.com/watch?v=qF5DUKEQ8sw)]

---

## 🚀 Roadmap

- [ ] Graph visualization of reasoning paths  
- [ ] Expand dataset beyond movies → multi-domain  
- [ ] Natural language → Cypher/Gremlin translation  
- [ ] API + Streamlit demo  

---

## 🤝 Contributing

PRs and discussions are welcome! If you’re exploring **Graph RAGs, agentic AI, or knowledge graphs**, feel free to collaborate.  

---

## 🙏 Acknowledgements

- [NetworkX](https://networkx.org/) for graph construction  
- [FAISS](https://faiss.ai/) for vector similarity search  
- Recent advances in **Graph RAG + Agentic AI workflows** that inspired this work  
