from dotenv import load_dotenv
from langgraph.graph import StateGraph, END

from graph.state import GraphState
from graph.nodes.node_constants import RETRIEVE, GRADE_DOCUMENTS, GENERATE, WEBSEARCH

from graph.nodes.retrieve import retrieve
from graph.nodes.grade_documents import grade_documents
from graph.nodes.generate import generate
from graph.nodes.web_search import web_search

from graph.chains.router import question_router, RouteQuery
from graph.chains.answer_grader import answer_grader
from graph.chains.hallucination_grader import hallucination_grader

load_dotenv()


def route_question(state: GraphState) -> str:
    question = state["question"]
    source: RouteQuery = question_router.invoke({"question": question})
    if source.datasource == WEBSEARCH:
        return WEBSEARCH
    return RETRIEVE


def decide_to_generate(state: GraphState) -> str:
    if state.get("web_search"):
        return WEBSEARCH
    return GENERATE


def grade_generation_grounded_in_documents_and_question(state: GraphState) -> str:
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    h = hallucination_grader.invoke({"documents": documents, "generation": generation})
    if h.binary_score:
        a = answer_grader.invoke({"question": question, "generation": generation})
        return "useful" if a.binary_score else "not useful"
    return "not supported"


workflow = StateGraph(GraphState)

workflow.add_node(RETRIEVE, retrieve)
workflow.add_node(GRADE_DOCUMENTS, grade_documents)
workflow.add_node(GENERATE, generate)
workflow.add_node(WEBSEARCH, web_search)

workflow.set_conditional_entry_point(
    route_question,
    {WEBSEARCH: WEBSEARCH, RETRIEVE: RETRIEVE},
)

workflow.add_edge(RETRIEVE, GRADE_DOCUMENTS)

workflow.add_conditional_edges(
    GRADE_DOCUMENTS,
    decide_to_generate,
    {WEBSEARCH: WEBSEARCH, GENERATE: GENERATE},
)

workflow.add_edge(WEBSEARCH, END)

workflow.add_conditional_edges(
    GENERATE,
    grade_generation_grounded_in_documents_and_question,
    {"useful": END, "not useful": END, "not supported": END},
)

app = workflow.compile()
