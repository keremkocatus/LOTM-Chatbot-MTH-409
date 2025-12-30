from graph.chains.generation import get_generation_chain
from graph.state import GraphState


def generate(state: GraphState):
    """Getirilen belgelere dayanarak cevap üretir."""
    question = state["question"]
    documents = state.get("documents") or []
    temperature = state.get("temperature") or 0.5  # None ise varsayılan 0.5

    if len(documents) == 0:
        generation = "Aradığınız bilgi veritabanındaki pathway veya yetenekler arasında bulunamadı."
        return {
            "question": question,
            "documents": [],
            "generation": generation,
            "web_search": False,
            "source_type": "no_docs",
        }

    context = "\n\n".join(
        f"[{d.metadata.get('pathway', 'Unknown Pathway')} - {d.metadata.get('title', 'Unknown Sequence')}]\n{d.page_content}"
        for d in documents
    )

    # Temperature ile dinamik chain oluştur
    generation_chain = get_generation_chain(temperature)
    
    generation = generation_chain.invoke({
        "context": context,
        "question": question
    })

    return {
        "question": question,
        "documents": documents,
        "generation": generation,
        "web_search": False,
        "source_type": "vectorstore",
    }