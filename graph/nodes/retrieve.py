from ingestion import get_retriever
from graph.state import GraphState

retriever_tool = get_retriever()

def retrieve(state: GraphState):
    question = state["question"]
    
    docs = retriever_tool.invoke(question)
    
    docs = docs or []
    return {"question": question, "documents": docs}