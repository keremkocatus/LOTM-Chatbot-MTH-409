from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

def get_generation_chain(temperature: float = 0.5, provider: str = "openai"):
    """Temperature ve provider parametreleriyle generation chain döndürür."""
    if provider == "gemini":
        llm = ChatGoogleGenerativeAI(
            model="gemini-3-flash-preview",
            temperature=temperature,
        )
    else:
        llm = ChatOpenAI(
            model="gpt-4o",
            temperature=temperature,
        )

    prompt = ChatPromptTemplate.from_template("""
You are a knowledgeable lore assistant for the novel "Lord of the Mysteries".

Rules:
- Answer the user's question using ONLY the provided context documents.
- The context is in English but user may ask in Turkish - translate/interpret as needed.
- "Sıra X" or "Sequence X" refers to Beyonder sequence levels (Sıra 9 = Sequence 9 = lowest, Sıra 0/1 = highest).
- Summarize and explain the relevant information from the context.
- If the context contains relevant information, USE IT to answer - don't say "not found".
- Only say "Bu bilgi verilen bölümlerde bulunmuyor" if the context truly has NO relevant info.
- Answer in the same language as the question (Turkish question = Turkish answer).

Context:
{context}

Question:
{question}

Answer:
""")

    return prompt | llm | StrOutputParser()

# Varsayılan chain (geriye dönük uyumluluk için)
generation_chain = get_generation_chain()
