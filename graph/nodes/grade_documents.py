from graph.chains.retrieval_grader import retrieval_grader
from graph.state import GraphState

def grade_documents(state: GraphState):
    """Getirilen belgelerin soruyla ilgili olup olmadığını değerlendirir."""
    question = state["question"]
    documents = state["documents"]
    
    print(f"\n📋 GRADING ({len(documents)} belge)")
    
    filtered = []
    web_search = False

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
        print(f"   ⚠️  Hiç ilgili belge yok -> OpenAI'a yönlendiriliyor")
    else:
        print(f"   ✅ {len(filtered)} ilgili belge bulundu")
    print(f"{'='*50}\n")

    return {
        "question": question,
        "documents": filtered,
        "web_search": web_search
    }
