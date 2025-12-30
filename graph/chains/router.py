from typing import Literal
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

class RouteQuery(BaseModel):
    datasource: Literal["vectorstore", "off_topic"] = Field(...)

system = """You are an expert at routing user questions for a Lord of the Mysteries (LoTM) chatbot.

Route to 'vectorstore' for ANY question about:
- Lord of the Mysteries (LoTM) novel
- Beyonders, pathways, sequences, potions
- Characters: Klein Moretti, Amon, Adam, Evernight Goddess, etc.
- Organizations: Tarot Club, Rose Redemption, Aurora Order, etc.
- Abilities, powers, Sefirot, Uniqueness, Angels, Gods
- Any fantasy/supernatural content that could be from the novel

Route to 'off_topic' for:
- Completely unrelated real-world topics (sports, politics, cooking, etc.)
- Technical programming questions
- Current events
- Any question NOT related to Lord of the Mysteries

When in doubt, route to 'vectorstore'.
"""

route_prompt = ChatPromptTemplate.from_messages([
    ("system", system),
    ("human", "{question}")
])

def get_question_router(provider: str = "openai"):
    """Provider'a göre question router döndürür."""
    if provider == "gemini":
        llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", temperature=0)
    else:
        llm = ChatOpenAI(model="gpt-4o", temperature=0)
    
    structured_llm_router = llm.with_structured_output(RouteQuery)
    return route_prompt | structured_llm_router

# Varsayılan (geriye dönük uyumluluk)
question_router = get_question_router("openai")
