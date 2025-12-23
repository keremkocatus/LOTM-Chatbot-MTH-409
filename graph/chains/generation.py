from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.5,
)

prompt = ChatPromptTemplate.from_template("""
You are a strict lore assistant for the novel "Lord of the Mysteries".

Rules:
- Answer using the provided context.
- Do NOT add information not present in the context.
- If the answer cannot be found in the context, say:
  "Bu bilgi verilen bölümlerde bulunmuyor."
- Keep answers concise and factual.

Context:
{context}

Question:
{question}

Answer:
""")

generation_chain = prompt | llm | StrOutputParser()
