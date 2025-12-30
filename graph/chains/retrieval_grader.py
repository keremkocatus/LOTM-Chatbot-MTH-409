from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

class GradeDocuments(BaseModel):
    binary_score: bool = Field(description="True if relevant, False otherwise")

system = """You are a grader assessing relevance of a document to a user question.

The document is from Lord of the Mysteries wiki about Beyonder pathways, sequences, and abilities.

Return True if the document contains ANY information that could help answer the question, even partially.
Return False ONLY if the document is completely unrelated to the question.

Be lenient - if there's any chance the document is useful, return True."""

grade_prompt = ChatPromptTemplate.from_messages([
    ("system", system),
    ("human", "Document:\n{document}\n\nQuestion:\n{question}")
])

def get_retrieval_grader(provider: str = "openai"):
    """Provider'a göre retrieval grader döndürür."""
    if provider == "gemini":
        llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", temperature=0)
    else:
        llm = ChatOpenAI(model="gpt-4o", temperature=0)
    
    structured_llm_grader = llm.with_structured_output(GradeDocuments)
    return grade_prompt | structured_llm_grader

# Varsayılan (geriye dönük uyumluluk)
retrieval_grader = get_retrieval_grader("openai")
