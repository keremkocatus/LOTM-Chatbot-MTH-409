from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

class GradeHallucination(BaseModel):
    binary_score: bool = Field(description="yes or no")

prompt = ChatPromptTemplate.from_messages([
    ("system", "Is the answer supported by the documents?"),
    ("human", "Documents: {documents}\nAnswer: {generation}")
])

def get_hallucination_grader(provider: str = "openai"):
    """Provider'a göre hallucination grader döndürür."""
    if provider == "gemini":
        llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", temperature=0)
    else:
        llm = ChatOpenAI(model="gpt-4o", temperature=0)
    
    structured_llm = llm.with_structured_output(GradeHallucination)
    return prompt | structured_llm

# Varsayılan (geriye dönük uyumluluk)
hallucination_grader = get_hallucination_grader("openai")
