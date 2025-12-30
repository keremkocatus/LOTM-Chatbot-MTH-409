from graph.chains.retrieval_grader import get_retrieval_grader
from graph.state import GraphState

def grade_documents(state: GraphState):
    """Getirilen belgelerin soruyla ilgili olup olmadığını değerlendirir."""
    question = state["question"]
    documents = state["documents"]
    provider = state.get("model_provider") or "openai"
    
    print(f"\n📋 GRADING ({len(documents)} belge)")
    
    filtered = []
    web_search = False
    
    # Provider'a göre grader al
    retrieval_grader = get_retrieval_grader(provider)

    for i, d in enumerate(documents):
        score = retrieval_grader.invoke({
            "question": question,
            "document": d.page_content
        })
        status = "✓" if score.binary_score else "✗"
        title = d.metadata.get('title', '?')
        print(f"   [{i+1}] {status} {title}")
        
        if score.binary_score:
            filtered.append(d)

    # Hiç ilgili belge yoksa OpenAI bilgisine yönlendir
    if len(filtered) == 0:
        web_search = True
        print(f"   ⚠️  Hiç ilgili belge yok -> Web Search'e yönlendiriliyor")
    else:
        print(f"   ✅ {len(filtered)} ilgili belge bulundu")
    print(f"{'='*50}\n")

    return {
        "question": question,
        "documents": filtered,
        "web_search": web_search
    }
