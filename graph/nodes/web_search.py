from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from graph.state import GraphState
import os

llm = ChatOpenAI(
    model="gpt-4o", 
    temperature=0.5  
)

def web_search(state: GraphState):
    """
    Belgelerde bilgi bulunamadığında LLM'in kendi hafızasını kullanarak cevap üretmesini sağlar.
    """
    print("--- NO RELEVANT DOCS FOUND: ANSWERING FROM MEMORY ---")
    
    question = state["question"]
    
    prompt = ChatPromptTemplate.from_template(
        """Sen Lord of the Mysteries evreni hakkında uzman bir asistansın. 
        Kullanıcının sorduğu soru hakkında veritabanımızda (vector store) doğrudan bir kaynak belge bulunamadı.
        
        Bu yüzden, kendi eğitim verini ve genel bilgini kullanarak bu soruyu en doğru şekilde ve tam olarak cevapla.
        Eğer cevabı gerçekten bilmiyorsan veya emin değilsen, uydurmak yerine nazikçe bilmediğini belirt.
        Konudan bağımsız ise soru kibarca cevap veremeyeceğini belirt.
        
        Soru: {question}
        Cevap:"""
    )
    
    chain = prompt | llm | StrOutputParser()
    
    generation = chain.invoke({"question": question})
    
    return {
        "question": question,
        "documents": [], 
        "generation": generation,
        "web_search": False, 
    }