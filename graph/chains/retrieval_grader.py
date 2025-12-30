from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI

class GradeDocuments(BaseModel):
    binary_score: bool = Field(description="True if relevant, False otherwise")

llm = ChatOpenAI(model="gpt-4o", temperature=0)
structured_llm_grader = llm.with_structured_output(GradeDocuments)

system = """You are a grader assessing relevance of a document to a user question.

The document is from Lord of the Mysteries wiki about Beyonder pathways, sequences, and abilities.

Return True if the document contains ANY information that could help answer the question, even partially.
Return False ONLY if the document is completely unrelated to the question.

Be lenient - if there's any chance the document is useful, return True."""

grade_prompt = ChatPromptTemplate.from_messages([
    ("system", system),
    ("human", "Document:\n{document}\n\nQuestion:\n{question}")
])

retrieval_grader = grade_prompt | structured_llm_grader
