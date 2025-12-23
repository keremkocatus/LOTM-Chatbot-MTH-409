from graph.chains.generation import generation_chain
from graph.state import GraphState

def generate(state: GraphState):
    question = state["question"]
    documents = state.get("documents") or []

    if len(documents) == 0:
        generation = "Bu bilgi verilen bölümlerde bulunmuyor."
        return {
            "question": question,
            "documents": [],
            "generation": generation,
            "web_search": False,
        }

    context = "\n\n".join(
        f"[Chapter {d.metadata.get('chapter_id')} - {d.metadata.get('chapter_title','')}]\n{d.page_content}"
        for d in documents
    )

    generation = generation_chain.invoke({
        "context": context,
        "question": question
    })

    return {
        "question": question,
        "documents": documents,
        "generation": generation,
        "web_search": False,
    }
