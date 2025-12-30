from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

class GradeAnswer(BaseModel):
    binary_score: bool = Field(description="yes or no")

prompt = ChatPromptTemplate.from_messages([
    ("system", "Does the answer address the question?"),
    ("human", "Question: {question}\nAnswer: {generation}")
])

def get_answer_grader(provider: str = "openai"):
    """Provider'a göre answer grader döndürür."""
    if provider == "gemini":
        llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", temperature=0)
    else:
        llm = ChatOpenAI(model="gpt-4o", temperature=0)
    
    structured_llm = llm.with_structured_output(GradeAnswer)
    return prompt | structured_llm

# Varsayılan (geriye dönük uyumluluk)
answer_grader = get_answer_grader("openai")
