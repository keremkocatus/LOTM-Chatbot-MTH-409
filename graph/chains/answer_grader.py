from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI

class GradeAnswer(BaseModel):
    binary_score: bool = Field(description="yes or no")

llm = ChatOpenAI(temperature=0)
structured_llm = llm.with_structured_output(GradeAnswer)

prompt = ChatPromptTemplate.from_messages([
    ("system", "Does the answer address the question?"),
    ("human", "Question: {question}\nAnswer: {generation}")
])

answer_grader = prompt | structured_llm
