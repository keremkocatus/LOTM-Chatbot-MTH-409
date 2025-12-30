from typing import List, TypedDict, Optional, Literal
from langchain_core.documents import Document


class GraphState(TypedDict):
    """LangGraph state tanımı."""
    question: str
    generation: str
    web_search: bool
    documents: List[Document]
    k_retrieved: Optional[int]
    temperature: Optional[float]
    source_type: Optional[str]
    model_provider: Optional[Literal["openai", "gemini"]]
