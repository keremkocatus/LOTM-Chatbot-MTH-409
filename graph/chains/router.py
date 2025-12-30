from typing import Literal
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI

class RouteQuery(BaseModel):
    datasource: Literal["vectorstore", "websearch"] = Field(...)

llm = ChatOpenAI(temperature=0)
structured_llm_router = llm.with_structured_output(RouteQuery)

system = """You are an expert at routing user questions.

Route to 'vectorstore' for ANY question about:
- Lord of the Mysteries (LoTM) novel
- Beyonders, pathways, sequences, potions
- Characters: Klein Moretti, Amon, Adam, Evernight Goddess, etc.
- Organizations: Tarot Club, Rose Redemption, Aurora Order, etc.
- Abilities, powers, Sefirot, Uniqueness, Angels, Gods
- Any fantasy/supernatural content that could be from the novel

Route to 'websearch' ONLY for:
- Completely unrelated real-world topics (sports, politics, cooking, etc.)
- Technical programming questions
- Current events

When in doubt, route to 'vectorstore'.
"""

route_prompt = ChatPromptTemplate.from_messages([
    ("system", system),
    ("human", "{question}")
])

question_router = route_prompt | structured_llm_router
