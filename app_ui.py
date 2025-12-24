import streamlit as st
import time
from dotenv import load_dotenv
from graph.graph import app

load_dotenv()

st.set_page_config(
    page_title="LotM Beyonder Archives",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
    }
    .stTextInput textarea {
        background-color: #1e2329;
        color: white;
    }
    .stMarkdown {
        color: #e0e0e0;
    }
    h1 {
        color: #9d4edd !important;
        font-family: 'Helvetica Neue', sans-serif;
    }
    h2 {
        color: #9d4edd !important;
    }
    .stChatMessage {
        background-color: #1e2329;
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 10px;
    }
    .stExpander {
        background-color: #151920;
        border-radius: 5px;
        border: 1px solid #2d3436;
    }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.image("https://i.pinimg.com/736x/55/e5/22/55e5227181c4793836881c03472096a6.jpg", width=150)
    st.markdown("<h2>🔮 The Fool's Library</h2>", unsafe_allow_html=True)
    st.caption("Spirit World Navigator v1.2")
    
    st.markdown("---")

    st.subheader("⚙️ Ayin Parametreleri")
    
    with st.expander("📜 Gizli Bilgi Ayarları", expanded=True):
        retriever_k = st.slider(
            "Bilgi Parçası (Chunk Sayısı)", 
            min_value=1, 
            max_value=10, 
            value=3
        )
        
        creativity = st.slider(
            "Delilik Seviyesi (Temp)", 
            min_value=0.0, 
            max_value=1.0, 
            value=0.3,
            step=0.1
        )

    selected_pathway = st.selectbox(
        "Odaklanılacak Pathway",
        ["Tümü (Genel)", "Fool (Seer)", "Door (Apprentice)", "Error (Marauder)", "Red Priest (Hunter)"],
        index=0
    )

    st.markdown("---")

    col1, col2 = st.columns([1, 4])
    with col1:
        st.write("🧹") 
    with col2:
        if st.button("Hafızayı Temizle", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    st.markdown("---")
    
    if creativity > 0.7:
        st.warning("⚠️ Dikkat: Yüksek delilik seviyesi! Model saçmalayabilir.")
    else:
        st.success("✅ Ruhsal durum stabil.")

st.title("🔮 Lord of the Mysteries Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    if message["role"] == "user":
        avatar = "👤"
    else:
        avatar = "🔮"
    
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

if prompt := st.chat_input("Gizli bilgiye erişmek için sorunuzu yazın..."):
    
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant", avatar="🔮"):
        message_placeholder = st.empty()
        
        with st.spinner("Ruhlar dünyasından bilgi çekiliyor..."):
            try:
                inputs = {
                    "question": prompt,
                    "k_retrieved": retriever_k,
                    "temperature": creativity,
                    "filter_pathway": selected_pathway
                }
                result = app.invoke(inputs)
                
                answer = result.get("generation", "Bilgiye erişilemedi.")
                documents = result.get("documents", [])

                message_placeholder.markdown(answer)

                if documents:
                    with st.expander("📜 Kadim Kaynaklar (Referanslar)"):
                        for i, doc in enumerate(documents):
                            pathway = doc.metadata.get("pathway", "Unknown")
                            title = doc.metadata.get("title", "Unknown")
                            content_preview = doc.page_content.split("Content:")[-1].strip()[:200]
                            
                            st.markdown(f"**{i+1}. {pathway} Pathway - {title}**")
                            st.caption(f"_{content_preview}..._")
                            if i < len(documents) - 1:
                                st.divider()

                st.session_state.messages.append({"role": "assistant", "content": answer})

            except Exception as e:
                message_placeholder.error(f"Bir anomali tespit edildi: {str(e)}")