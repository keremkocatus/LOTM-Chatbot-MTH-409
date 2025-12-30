import streamlit as st
from dotenv import load_dotenv
from graph.graph import app

load_dotenv()

st.set_page_config(
    page_title="LotM Beyonder Archives",
    page_icon="🔮",
    layout="centered"
)

st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
    }
    .stMarkdown {
        color: #e0e0e0;
    }
    h1 {
        color: #9d4edd !important;
        text-align: center;
    }
    .stChatMessage {
        background-color: #1e2329;
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

st.title("🔮 Lord of the Mysteries Chatbot")

# Sidebar'da model seçimi
with st.sidebar:
    st.header("⚙️ Ayarlar")
    
    model_provider = st.radio(
        "🤖 Model Seçimi",
        options=["openai", "gemini"],
        format_func=lambda x: "🟢 OpenAI (GPT-4o)" if x == "openai" else "🔵 Google Gemini",
        index=0,
        help="Kullanılacak AI modelini seçin"
    )
    
    st.divider()
    st.caption("OpenAI: GPT-4o modeli kullanır")
    st.caption("Gemini: Google Gemini 3 Flash modeli kullanır")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    avatar = "👤" if message["role"] == "user" else "🔮"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            sources = message["sources"]
            if isinstance(sources, list) and len(sources) > 0:
                with st.expander(f"📚 Kaynaklar ({len(sources)})", expanded=False):
                    for s in sources:
                        st.markdown(f"• {s}")

if prompt := st.chat_input("Sorunuzu yazın..."):
    
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant", avatar="🔮"):
        message_placeholder = st.empty()
        
        model_emoji = "🟢" if model_provider == "openai" else "🔵"
        model_name = "OpenAI" if model_provider == "openai" else "Gemini"
        
        with st.spinner(f"{model_emoji} {model_name} ile bilgi çekiliyor..."):
            try:
                result = app.invoke({
                    "question": prompt,
                    "k_retrieved": 6,
                    "temperature": 0.3,
                    "model_provider": model_provider
                })
                
                answer = result.get("generation", "Bilgiye erişilemedi.")
                documents = result.get("documents", [])
                source_type = result.get("source_type", "unknown")

                message_placeholder.markdown(answer)

                # Kaynakları dropdown olarak göster
                sources_list = []
                if source_type == "vectorstore" and documents:
                    for doc in documents:
                        title = doc.metadata.get("title", "?")
                        pathway = doc.metadata.get("pathway", "?")
                        sources_list.append(f"{pathway}: {title}")
                    
                    if sources_list:
                        with st.expander(f"📚 Kaynaklar ({len(sources_list)})", expanded=False):
                            for s in sources_list:
                                st.markdown(f"• {s}")
                elif source_type == "web_search":
                    st.info("🌐 Kaynak: Web Araması (DuckDuckGo)")
                elif source_type == "off_topic":
                    pass  # Off-topic mesajı zaten cevabın içinde

                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": answer,
                    "sources": sources_list
                })

            except Exception as e:
                message_placeholder.error(f"Hata: {str(e)}")