from graph.graph import app

print("LoTM Chatbot (exit yaz çık)")

while True:
    q = input("\nYou: ")
    if q.lower() == "exit":
        break

    res = app.invoke({"question": q})

    print("\nAssistant:", res["generation"])
    print("\nSources:")
    for d in res["documents"][:3]:
        source = d.metadata.get('source', 'Bilinmiyor')
        title = d.metadata.get('title', 'Başlıksız')
        print(f"- {source} | {title}")
