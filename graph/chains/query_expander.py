from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

system = """You are a simple query translator for a Lord of the Mysteries wiki search.

Your ONLY job is to translate Turkish terms to English. Do NOT add any information you're not sure about.

Translation rules:
- "Sıra X" or "sıra X" -> "Sequence X"
- "yetenek/yetenekler" -> "abilities"
- "güç/güçler" -> "powers"  
- "yol" -> "pathway"
- Keep pathway names as-is (Fool, Error, Door, Sun, Demoness, etc.)
- Keep character names as-is

Do NOT:
- Add sequence names you're unsure about
- Add extra information not in the original question
- Make up any LoTM-specific terms

Return ONLY the translated query.

Examples:
- "Fool sıra 5 yetenekleri" -> "Fool Sequence 5 abilities"
- "Error pathway sıra 2 nedir" -> "Error pathway Sequence 2"
- "Seer yetenekleri nelerdir" -> "Seer abilities"
- "Door yolu sıra 4 güçleri" -> "Door pathway Sequence 4 powers"
"""

expand_prompt = ChatPromptTemplate.from_messages([
    ("system", system),
    ("human", "{question}")
])

query_expander = expand_prompt | llm | StrOutputParser()
