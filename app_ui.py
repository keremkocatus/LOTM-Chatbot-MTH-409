import streamlit as st
import os
from dotenv import load_dotenv

# Senin oluşturduğun LangGraph yapısını çağırıyoruz
from graph.graph import app

# Ortam değişkenlerini yükle
load_dotenv()

# --- Sayfa Ayarları ---
st.set_page_config(
    page_title="LotM Beyonder Chatbot",
    page_icon="🔮",
    layout="centered"
)

st.title("🔮 Lord of the Mysteries Chatbot")
st.caption("Beyonder yolları ve yetenekleri hakkında sorular sorabilirsiniz.")

# --- Session State (Sohbet Geçmişi) ---
# Streamlit her etkileşimde sayfayı yenilediği için geçmişi hafızada tutmamız lazım.
if "messages" not in st.session_state:
    st.session_state.messages = []

# Geçmiş mesajları ekrana yazdır
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Kullanıcı Girdisi ve Bot Cevabı ---
if prompt := st.chat_input("Bir soru sorun (Örn: Sequence 7 Witch yetenekleri neler?)..."):
    
    # 1. Kullanıcı mesajını ekrana yaz ve geçmişe ekle
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 2. LangGraph uygulamasını çalıştır (Botun düşünme süreci)
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("Thinking... 🔮")
        
        try:
            # LangGraph'ı invoke ediyoruz (State yapına uygun şekilde)
            inputs = {"question": prompt}
            result = app.invoke(inputs)
            
            answer = result.get("generation", "Bir hata oluştu veya cevap bulunamadı.")
            documents = result.get("documents", [])

            # Cevabı yazdır
            message_placeholder.markdown(answer)

            # 3. Kaynakları Göster (Opsiyonel ama RAG için çok şık olur)
            if documents:
                with st.expander("📚 Kullanılan Kaynaklar / Sources"):
                    for i, doc in enumerate(documents):
                        # Metadata'dan başlıkları çekiyoruz (senin son ingestion yapına göre)
                        pathway = doc.metadata.get("pathway", "Unknown Pathway")
                        title = doc.metadata.get("title", "Unknown Title")
                        source_file = doc.metadata.get("source", "")
                        
                        st.markdown(f"**{i+1}. {pathway} - {title}**")
                        st.caption(f"Dosya: {source_file}")
                        st.text(doc.page_content[:200] + "...") # Metnin ilk 200 karakteri

            # Bot cevabını geçmişe ekle
            st.session_state.messages.append({"role": "assistant", "content": answer})

        except Exception as e:
            message_placeholder.error(f"Bir hata oluştu: {e}")