from typing import Literal
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI

class RouteQuery(BaseModel):
    datasource: Literal["vectorstore", "websearch"] = Field(...)

llm = ChatOpenAI(temperature=0)
structured_llm_router = llm.with_structured_output(RouteQuery)

system = """
If the question is about Lord of the Mysteries (plot, characters, sequences, pathways),
route to vectorstore. Otherwise route to websearch.
"""

route_prompt = ChatPromptTemplate.from_messages([
    ("system", system),
    ("human", "{question}")
])

question_router = route_prompt | structured_llm_router
