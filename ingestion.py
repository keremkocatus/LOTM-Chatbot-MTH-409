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
        
        all_items = []
        if isinstance(raw_data, list) and len(raw_data) > 0:
            all_items = raw_data[0].get("items", [])
        elif isinstance(raw_data, dict):
            all_items = raw_data.get("items", [])
            
        print(f"   -> Processing '{pathway_name}': Found {len(all_items)} sequences.")

        for item in all_items:
            raw_text = item.get("text", "").strip()
            title = item.get("title", f"{pathway_name} Sequence")
            sequence_num = item.get("sequence", -1)
            url = item.get("page_url", "")
            
            if not raw_text:
                continue
            
            docs.append(
                Document(
                    page_content=raw_text, 
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
    
    print(f"Scanning directory: {DATA_DIR} ...")
    files = [f for f in os.listdir(DATA_DIR) if f.endswith(".json")]
    
    if not files:
        print("Klasörde hiç .json dosyası bulunamadı.")
        return 0

    for filename in files:
        file_path = os.path.join(DATA_DIR, filename)
        file_docs = process_single_json(file_path, filename)
        all_docs.extend(file_docs)

    print(f"\nTotal documents collected from {len(files)} files: {len(all_docs)}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200,
        separators=[
            "\nAuthorities",             
            "\nMythical Creature Form",  
            "\nNew Abilities",           
            "\nStrengthened Abilities",  
            "\n\n",                      
            "\n",                       
            ". ",                        
            " ",                         
            ""                           
        ]
    )
    
    chunks = splitter.split_documents(all_docs)
    print(f"Splitting complete. Created {len(chunks)} raw chunks.")
    
    for chunk in chunks:
        meta = chunk.metadata
        pathway = meta.get("pathway", "Unknown")
        title = meta.get("title", "Unknown Title")
        
        header_text = f"Pathway: {pathway}\nTitle: {title}\n\nContent:\n"
        
        chunk.page_content = header_text + chunk.page_content

    print("Metadata headers injected into all chunks.")

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

def get_retriever_with_params(k: int = None, filter_dict: dict = None):
    """Parametreli retriever döndürür."""
    embeddings = OpenAIEmbeddings(model=EMBED_MODEL)
    vs = Chroma(
        collection_name=COLLECTION,
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
    )
    
    search_kwargs = {"k": k or K}
    if filter_dict:
        # Chroma birden fazla filtre için $and operatörü gerektirir
        if len(filter_dict) > 1:
            search_kwargs["filter"] = {
                "$and": [{key: value} for key, value in filter_dict.items()]
            }
        else:
            search_kwargs["filter"] = filter_dict
    
    return vs.as_retriever(search_kwargs=search_kwargs)

if __name__ == "__main__":
    build_index()