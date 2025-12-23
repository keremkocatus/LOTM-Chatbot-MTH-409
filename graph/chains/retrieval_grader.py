from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI

class GradeDocuments(BaseModel):
    binary_score: str = Field(description="yes or no")

llm = ChatOpenAI(temperature=0)
structured_llm_grader = llm.with_structured_output(GradeDocuments)

system = """You are a grader assessing relevance of a document to a question.
Return 'yes' if the document is relevant to answering the question, otherwise return 'no'."""

grade_prompt = ChatPromptTemplate.from_messages([
    ("system", system),
    ("human", "Document:\n{document}\n\nQuestion:\n{question}")
])

retrieval_grader = grade_prompt | structured_llm_grader
