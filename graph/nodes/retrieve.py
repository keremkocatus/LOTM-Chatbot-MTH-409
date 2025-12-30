import re
import os
from ingestion import get_retriever_with_params
from graph.state import GraphState
from graph.chains.query_expander import get_query_expander

# Data klasöründen pathway listesini al
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")
PATHWAYS = [f.replace("_sequences.json", "") for f in os.listdir(DATA_DIR) if f.endswith("_sequences.json")]

def extract_sequence_number(question: str) -> int:
    """Sorudan sequence numarasını çıkar."""
    seq_pattern = r'(?:s[ıi]ra|sequence)\s*(\d+)'
    seq_match = re.search(seq_pattern, question.lower())
    return int(seq_match.group(1)) if seq_match else None


def extract_pathway(question: str) -> str:
    """Sorudan pathway ismini çıkar (data klasöründeki dosyalarla eşleştir)."""
    q = question.lower().replace(" ", "_") 
    
    for pathway in PATHWAYS:
        if pathway in q or pathway.replace("_", " ") in question.lower():
            # Metadata'daki format: "Red_priest", "Wheel_of_fortune" (sadece ilk harf büyük)
            return pathway.capitalize()
    return None


def retrieve(state: GraphState):
    question = state["question"]
    k = state.get("k_retrieved") or 6
    provider = state.get("model_provider") or "openai"
    
    sequence_num = extract_sequence_number(question)
    pathway = extract_pathway(question)
    
    # Filtre oluştur
    filters = []
    if sequence_num is not None:
        filters.append({"sequence": sequence_num})
    if pathway:
        filters.append({"pathway": pathway})
    
    if len(filters) > 1:
        filter_dict = {"$and": filters}
    elif len(filters) == 1:
        filter_dict = filters[0]
    else:
        filter_dict = None
    
    # Provider'a göre query expander al
    query_expander = get_query_expander(provider)
    expanded_query = query_expander.invoke({"question": question})
    combined_query = f"{question} {expanded_query}"
    
    print(f"\n{'='*50}")
    print(f"🔍 RETRIEVE")
    print(f"   Query: {combined_query}")
    print(f"   Filter: {filter_dict if filter_dict else 'Yok'}")
    
    if filter_dict:
        retriever = get_retriever_with_params(k=k, filter_dict=filter_dict)
    else:
        retriever = get_retriever_with_params(k=k)
    
    docs = retriever.invoke(combined_query)
    docs = docs or []
    
    print(f"   Bulunan: {len(docs)} belge")
    for i, d in enumerate(docs):
        print(f"   [{i+1}] {d.metadata.get('pathway', '?')} - {d.metadata.get('title', '?')}")
    
    return {"question": question, "documents": docs}