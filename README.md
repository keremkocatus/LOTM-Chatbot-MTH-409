# 🔮 Lord of the Mysteries RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot specialized in answering questions about the "Lord of the Mysteries" novel universe. Built with LangChain, LangGraph, ChromaDB, and Streamlit.

## 📖 About

This chatbot uses a sophisticated RAG pipeline to answer questions about Beyonder pathways, sequences, and abilities from the Lord of the Mysteries novel series. It combines vector search with web search fallback for comprehensive answers.

## 🏗️ Architecture

```
┌─────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Question  │────▶│  Question Router │────▶│  Vector Store   │
└─────────────┘     └──────────────────┘     │   (ChromaDB)    │
                            │                └────────┬────────┘
                            │                         │
                            ▼                         ▼
                    ┌───────────────┐        ┌───────────────┐
                    │  Web Search   │        │   Retrieve    │
                    │ (DuckDuckGo)  │        │   Documents   │
                    └───────┬───────┘        └───────┬───────┘
                            │                         │
                            │                         ▼
                            │                ┌───────────────┐
                            │                │    Grade      │
                            │                │   Documents   │
                            │                └───────┬───────┘
                            │                         │
                            │         ┌───────────────┴───────────────┐
                            │         │                               │
                            │         ▼                               ▼
                            │  ┌─────────────┐              ┌───────────────┐
                            │  │  Generate   │──────────────│  Web Search   │
                            │  │   Answer    │   (fallback) │  (DuckDuckGo) │
                            │  └──────┬──────┘              └───────────────┘
                            │         │
                            │         ▼
                            │  ┌─────────────┐
                            │  │   Grade     │
                            │  │ Generation  │
                            │  └──────┬──────┘
                            │         │
                            ▼         ▼
                    ┌─────────────────────────┐
                    │        Response         │
                    └─────────────────────────┘
```

## 🛠️ Tech Stack

- **LangChain & LangGraph**: Orchestration and workflow management
- **ChromaDB**: Vector database for semantic search
- **OpenAI**: Embeddings (`text-embedding-3-large`) and LLM (`gpt-4o`)
- **Google Gemini**: Alternative LLM (`gemini-3-flash-preview`)
- **DuckDuckGo**: Free web search fallback
- **Streamlit**: Web UI with model selection

## 📁 Project Structure

```
LOTM-Chatbot-MTH-409/
├── app_ui.py              # Streamlit web interface
├── ingestion.py           # Data indexing to ChromaDB
├── main.py                # CLI entry point
├── requirements.txt       # Python dependencies
├── .env                   # Environment variables (not in repo)
├── data/                  # Scraped pathway data (JSON)
│   ├── fool_sequences.json
│   ├── door_sequences.json
│   └── ... (22 pathway files)
├── graph/
│   ├── graph.py           # LangGraph workflow definition
│   ├── state.py           # GraphState TypedDict
│   ├── chains/            # LLM chains
│   │   ├── router.py              # Question routing
│   │   ├── generation.py          # Answer generation
│   │   ├── retrieval_grader.py    # Document relevance grading
│   │   ├── hallucination_grader.py
│   │   ├── answer_grader.py
│   │   └── query_expander.py      # Turkish→English translation
│   └── nodes/             # Graph nodes
│       ├── retrieve.py            # Vector search with filters
│       ├── grade_documents.py     # Document grading
│       ├── generate.py            # LLM generation
│       └── web_search.py          # DuckDuckGo search
└── .chroma_lotm/          # ChromaDB persistent storage
```

## 🚀 Installation

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/LOTM-Chatbot-MTH-409.git
cd LOTM-Chatbot-MTH-409
```

### 2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure environment variables
Create a `.env` file:
```env
OPENAI_API_KEY=your_openai_api_key_here
GOOGLE_API_KEY=your_google_api_key_here
LOTM_DATA_DIR=./data
CHROMA_DB_PATH=./.chroma_lotm
CHROMA_COLLECTION=lotm-chroma
RETRIEVAL_K=6
OPENAI_EMBEDDING_MODEL=text-embedding-3-large
```

### 5. Index the data
```bash
python ingestion.py
```

### 6. Run the application
```bash
streamlit run app_ui.py
```

## 💡 Features

### 🤖 Multi-Model Support
- **OpenAI GPT-4o**: High-quality responses with OpenAI
- **Google Gemini 3 Flash**: Fast and efficient alternative
- Switch between models via sidebar toggle in the UI

### Smart Query Routing
- Automatically routes LoTM-related questions to vector store
- Non-related questions go to web search

### Metadata Filtering
- Filter by **pathway**: "Fool yetenekleri", "Red Priest abilities"
- Filter by **sequence number**: "sıra 5", "sequence 3"
- Combined filters: "Fool sıra 7 yetenekleri"

### Query Expansion
- Turkish queries automatically translated to English for better semantic search
- Example: "Kahin yetenekleri" → "Seer abilities"

### Fallback Mechanisms
- If documents aren't relevant → Web Search
- If generation isn't grounded → Web Search
- If answer isn't useful → Web Search

## 📊 Data Collection

The pathway data was scraped from [Lord of the Mysteries Wiki](https://lordofthemysteries.fandom.com/) using the following JavaScript code in browser console:

```javascript
(async () => {
  // Wiki URL'lerinde kullanılan Pathway isimleri
  const pathwayNames = [
    "Fool", "Door", "Error",
    "Sun", "Tyrant", "White_Tower",
    "Visionary", "Hanged_Man",
    "Darkness", "Death", "Twilight_Giant",
    "Hunter", "Demoness", 
    "Paragon", "Hermit",
    "Wheel_of_Fortune",
    "Moon", "Mother",
    "Justiciar", "Black_Emperor",
    "Chained", "Abyss"
  ];

  const clean = (s) =>
    (s || "")
      .replace(/\u00a0/g, " ")
      .replace(/\[\d+\]/g, "")
      .replace(/\n{3,}/g, "\n\n")
      .trim();

  const parseDoc = (doc, url) => {
    const h2s = [...doc.querySelectorAll('h2[id^="Sequence_"]')];
    return h2s.map(h2 => {
      const title = (h2.innerText || "").trim();
      const h2_id = h2.id;
      const section_id = h2.getAttribute("aria-controls") || "";
      const section = section_id ? doc.getElementById(section_id) : null;
      const text = section ? clean(section.innerText) : "";
      const m = title.match(/Sequence\s+(\d+)/i);
      const sequence = m ? Number(m[1]) : null;
      return { page_url: url, h2_id, section_id, sequence, title, text };
    });
  };

  const downloadJSON = (data, filename) => {
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = filename;
    a.style.display = 'none';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  };

  const sleep = (ms) => new Promise(r => setTimeout(r, ms));

  console.log("İndirme işlemi başlıyor...");

  for (const name of pathwayNames) {
    const url = `https://lordofthemysteries.fandom.com/wiki/${name}_Pathway/Abilities`;
    const filename = `${name.toLowerCase()}_sequences.json`;

    try {
      console.log(`${filename} işleniyor...`);
      
      const response = await fetch(url);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const html = await response.text();
      const doc = new DOMParser().parseFromString(html, "text/html");
      
      const items = parseDoc(doc, url);

      const dataToSave = [
        { url, count: items.length, items }
      ];

      downloadJSON(dataToSave, filename);
      
      await sleep(1000);

    } catch (e) {
      console.error(`${name} indirilemedi:`, e);
    }
  }
  
  console.log("Tüm indirmeler tamamlandı.");
})();
```

## 📝 Example Queries

| Query | Description |
|-------|-------------|
| `Seer yetenekleri nelerdir?` | Get Seer (Fool pathway) abilities |
| `Red Priest sıra 5` | Get Red Priest Sequence 5 abilities |
| `Demoness pathway hakkında bilgi ver` | Information about Demoness pathway |
| `Klein Moretti kimdir?` | Falls back to web search (character info) |
| `Fool sıra 0 yetkileri` | Fool Sequence 0 (The Fool) authorities |

## 🔧 Configuration

### Adjustable Parameters (via UI or API)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_provider` | openai | LLM provider (openai / gemini) |
| `k_retrieved` | 6 | Number of documents to retrieve |
| `temperature` | 0.3 | LLM temperature (0-1) |

## 📄 License

MIT License

## 🙏 Acknowledgments

- [Lord of the Mysteries Wiki](https://lordofthemysteries.fandom.com/) for the pathway data
- [Cuttlefish That Loves Diving](https://www.novelupdates.com/nauthor/cuttlefish-that-loves-diving/) - Author of Lord of the Mysteries

---

**MTH-409 Course Project** | December 2025
