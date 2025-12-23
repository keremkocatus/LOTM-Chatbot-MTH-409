from graph.state import GraphState

def web_search(state: GraphState):
    return {
        "question": state["question"],
        "documents": state.get("documents", []),
        "web_search": False,
    }
