from dotenv import load_dotenv
from langgraph.graph import StateGraph, END

from graph.state import GraphState
from graph.nodes.node_constants import RETRIEVE, GRADE_DOCUMENTS, GENERATE, OFF_TOPIC, WEBSEARCH

from graph.nodes.retrieve import retrieve
from graph.nodes.grade_documents import grade_documents
from graph.nodes.generate import generate
from graph.nodes.off_topic import off_topic
from graph.nodes.web_search import web_search

from graph.chains.router import get_question_router, RouteQuery
from graph.chains.answer_grader import get_answer_grader
from graph.chains.hallucination_grader import get_hallucination_grader

load_dotenv()


def route_question(state: GraphState) -> str:
    question = state["question"]
    provider = state.get("model_provider") or "openai"
    question_router = get_question_router(provider)
    source: RouteQuery = question_router.invoke({"question": question})
    if source.datasource == OFF_TOPIC:
        return OFF_TOPIC
    return RETRIEVE


def decide_to_generate(state: GraphState) -> str:
    if state.get("web_search"):
        return WEBSEARCH
    return GENERATE


def grade_generation_grounded_in_documents_and_question(state: GraphState) -> str:
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]
    provider = state.get("model_provider") or "openai"
    
    hallucination_grader = get_hallucination_grader(provider)
    answer_grader = get_answer_grader(provider)

    h = hallucination_grader.invoke({"documents": documents, "generation": generation})
    if h.binary_score:
        a = answer_grader.invoke({"question": question, "generation": generation})
        if a.binary_score:
            return "useful"
        else:
            print("   ⚠️ Cevap faydalı değil -> Web Search'e yönlendiriliyor")
            return "not useful"
    print("   ⚠️ Cevap belgelerle desteklenmiyor -> Web Search'e yönlendiriliyor")
    return "not supported"


workflow = StateGraph(GraphState)

workflow.add_node(RETRIEVE, retrieve)
workflow.add_node(GRADE_DOCUMENTS, grade_documents)
workflow.add_node(GENERATE, generate)
workflow.add_node(OFF_TOPIC, off_topic)
workflow.add_node(WEBSEARCH, web_search)

workflow.set_conditional_entry_point(
    route_question,
    {OFF_TOPIC: OFF_TOPIC, RETRIEVE: RETRIEVE},
)

workflow.add_edge(RETRIEVE, GRADE_DOCUMENTS)

workflow.add_conditional_edges(
    GRADE_DOCUMENTS,
    decide_to_generate,
    {WEBSEARCH: WEBSEARCH, GENERATE: GENERATE},
)

workflow.add_edge(OFF_TOPIC, END)
workflow.add_edge(WEBSEARCH, END)

workflow.add_conditional_edges(
    GENERATE,
    grade_generation_grounded_in_documents_and_question,
    {"useful": END, "not useful": WEBSEARCH, "not supported": WEBSEARCH},
)

app = workflow.compile()
