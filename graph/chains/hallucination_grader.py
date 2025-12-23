from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI

class GradeHallucination(BaseModel):
    binary_score: bool = Field(description="yes or no")

llm = ChatOpenAI(temperature=0)
structured_llm = llm.with_structured_output(GradeHallucination)

prompt = ChatPromptTemplate.from_messages([
    ("system", "Is the answer supported by the documents?"),
    ("human", "Documents: {documents}\nAnswer: {generation}")
])

hallucination_grader = prompt | structured_llm
