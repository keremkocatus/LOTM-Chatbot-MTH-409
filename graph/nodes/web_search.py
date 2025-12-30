from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from ddgs import DDGS
from graph.state import GraphState

load_dotenv()


def web_search(state: GraphState):
    """DuckDuckGo ile gerçek web araması yapar ve sonuçları özetler."""
    question = state["question"]
    temperature = state.get("temperature") or 0.5
    provider = state.get("model_provider") or "openai"
    
    print(f"\n🌐 WEB SEARCH")
    print(f"   Sorgu: {question}")
    
    # DuckDuckGo araması (yeni ddgs paketi)
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(f"Lord of the Mysteries {question}", max_results=5))
            search_results = "\n".join([f"- {r['title']}: {r['body']}" for r in results])
        print(f"   ✅ Web sonuçları alındı ({len(results)} sonuç)")
    except Exception as e:
        print(f"   ❌ Web arama hatası: {e}")
        search_results = ""
    
    # Provider'a göre LLM seç
    if provider == "gemini":
        llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", temperature=temperature)
    else:
        llm = ChatOpenAI(model="gpt-4o", temperature=temperature)
    
    prompt = ChatPromptTemplate.from_template(
        """Sen Lord of the Mysteries evreni hakkında uzman bir asistansın.
        
        Kullanıcının sorusu: {question}
        
        Web arama sonuçları:
        {search_results}
        
        Bu web arama sonuçlarını kullanarak soruyu Türkçe olarak detaylı ve doğru bir şekilde cevapla.
        Eğer sonuçlarda yeterli bilgi yoksa, kendi bilgini de kullanabilirsin ama bunu belirt.
        
        Cevap:"""
    )
    
    chain = prompt | llm | StrOutputParser()
    generation = chain.invoke({"question": question, "search_results": search_results})
    
    return {
        "question": question,
        "documents": [],
        "generation": generation,
        "web_search": False,
        "source_type": "web_search",
    }