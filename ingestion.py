import json
import os
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

DATA_DIR = os.getenv("LOTM_DATA_DIR", "./data")
CHROMA_DIR = os.getenv("CHROMA_DB_PATH", "./.chroma_lotm")
COLLECTION = os.getenv("CHROMA_COLLECTION", "lotm-chroma")
K = int(os.getenv("RETRIEVAL_K", "6"))
EMBED_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")

def process_single_json(file_path, filename):
    docs = []
    pathway_name = filename.replace("_sequences.json", "").replace("_sequence.json", "").replace(".json", "").capitalize()
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
        
        # JSON yapısı: [ { "items": [...] } ]
        all_items = []
        if isinstance(raw_data, list) and len(raw_data) > 0:
            all_items = raw_data[0].get("items", [])
        elif isinstance(raw_data, dict):
            all_items = raw_data.get("items", [])
            
        print(f"   -> Processing '{pathway_name}': Found {len(all_items)} sequences.")

        for item in all_items:
            content = item.get("text", "").strip()
            title = item.get("title", f"{pathway_name} Sequence")
            sequence_num = item.get("sequence", -1)
            url = item.get("page_url", "")
            
            if not content:
                continue

            docs.append(
                Document(
                    page_content=content,
                    metadata={
                        "source": filename,
                        "pathway": pathway_name, 
                        "title": title,
                        "sequence": sequence_num,
                        "url": url
                    },
                )
            )
    except Exception as e:
        print(f"   ERROR reading {filename}: {e}")
        
    return docs

def build_index() -> int:
    if not os.path.exists(DATA_DIR):
        print(f"HATA: Klasör bulunamadı -> {DATA_DIR}")
        return 0

    all_docs = []
    
    # Klasördeki tüm dosyaları tara
    print(f"Scanning directory: {DATA_DIR} ...")
    files = [f for f in os.listdir(DATA_DIR) if f.endswith(".json")]
    
    if not files:
        print("Klasörde hiç .json dosyası bulunamadı.")
        return 0

    # Her dosyayı işle ve listeye ekle
    for filename in files:
        file_path = os.path.join(DATA_DIR, filename)
        file_docs = process_single_json(file_path, filename)
        all_docs.extend(file_docs)

    print(f"\nTotal documents collected from {len(files)} files: {len(all_docs)}")

    # Text Splitting
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
    )
    
    chunks = splitter.split_documents(all_docs)
    print(f"Created {len(chunks)} chunks total.")

    # Embedding ve Vectorstore
    embeddings = OpenAIEmbeddings(model=EMBED_MODEL)

    vectorstore = Chroma(
        collection_name=COLLECTION,
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
    )

    BATCH_SIZE = 50  
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i:i + BATCH_SIZE]
        vectorstore.add_documents(batch)
        print(f"Indexed {i + len(batch)} / {len(chunks)}")

    print("Index updated successfully.")
    return len(chunks)

def get_retriever():
    embeddings = OpenAIEmbeddings(model=EMBED_MODEL)
    vs = Chroma(
        collection_name=COLLECTION,
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
    )
    return vs.as_retriever(search_kwargs={"k": K})

if __name__ == "__main__":
    build_index()