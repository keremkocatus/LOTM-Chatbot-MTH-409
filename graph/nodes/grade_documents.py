from graph.chains.retrieval_grader import retrieval_grader
from graph.state import GraphState

def grade_documents(state: GraphState):
    question = state["question"]
    documents = state["documents"]

    filtered = []
    web_search = False

    for d in documents:
        score = retrieval_grader.invoke({
            "question": question,
            "document": d.page_content
        })
        if score.binary_score.lower() == "yes":
            filtered.append(d)
        else:
            web_search = True

    return {
        "question": question,
        "documents": filtered,
        "web_search": web_search
    }
