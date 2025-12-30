from graph.state import GraphState


def off_topic(state: GraphState):
    """Lord of the Mysteries ile alakalı olmayan sorulara kibarca cevap verir."""
    question = state["question"]
    
    response = """🔮 Merhaba! Ben Lord of the Mysteries evrenine özel bir asistanım.

Maalesef bu sorunuz romanla ilgili görünmüyor. Size yalnızca şu konularda yardımcı olabilirim:

• **Beyonder Yolları** (Pathways) ve Sıraları (Sequences)
• **Karakterler** (Klein Moretti, Amon, Adam, vb.)
• **Organizasyonlar** (Tarot Club, Aurora Order, vb.)
• **Yetenekler ve Güçler**
• **Sefirot, Uniqueness, Melekler ve Tanrılar**

Lütfen Lord of the Mysteries ile ilgili bir soru sorun, size yardımcı olmaktan mutluluk duyarım! 📚✨"""

    print(f"\n🚫 OFF-TOPIC: Soru romanla ilgili değil")
    print(f"   Soru: {question[:50]}...")
    
    return {
        "question": question,
        "documents": [],
        "generation": response,
        "web_search": False,
        "source_type": "off_topic",
    }
