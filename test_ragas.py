import os
import pandas as pd
from dotenv import load_dotenv
load_dotenv()

from graph.graph import app
from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.metrics import Faithfulness, LLMContextRecall, ResponseRelevancy, ContextPrecision
from ragas.llms import LangchainLLMWrapper
from langchain_google_genai import ChatGoogleGenerativeAI

# =========================================================================
# TEST VERİ SETİ - Lord of the Mysteries Soruları
# =========================================================================

TEST_QUESTIONS = [
    {
        "question": "Fool Pathway'de Sequence 9 Seer'ın yetenekleri nelerdir?",
        "ground_truth": "Seer potionunu içenler hafıza artışı, Spirituality gelişimi, Spirit Vision (ruhani görüş), Divination Arts & Ritualistic Magic yetenekleri ve Danger Intuition (tehlike sezgisi) kazanır."
    },
    {
        "question": "Sequence 8 Clown hangi fiziksel yeteneklere sahiptir?",
        "ground_truth": "Clown, vücutları üzerinde güçlü kontrol, akrobatik yetenek, mükemmel denge, el becerisi ve hız artışı kazanır. Ayrıca Paper Daggers (kağıt hançerler) oluşturabilir."
    },
    {
        "question": "Marionettist'in Spirit Body Threads Manipulation yeteneği nasıl çalışır?",
        "ground_truth": "Marionettist, 100 metreye kadar Spirit Body Threads'i algılayıp manipüle edebilir. Hedefin Soul Body, Astral Projection, Body of Heart and Mind ve Ether Body'sini etkileyerek kontrol sağlar."
    },
    {
        "question": "Bizarro Sorcerer'ın Bestowal yeteneği nedir?",
        "ground_truth": "Bizarro Sorcerer, Worms of Spirit'lerini Marionette'lerine 'hediye' edebilir, bu sayede Marionette'ler Beyonder güçlerini kullanabilir. Başlangıçta 50 Worms of Spirit ayırabilirler."
    },
    {
        "question": "Scholar of Yore Historical Void'i nasıl kullanır?",
        "ground_truth": "Scholar of Yore, Historical Void Borrowing (geçmiş benliğinden güç ödünç alma), Historical Projection Summoning (geçmişten projeksiyon çağırma) ve Historical Void Hiding (Historical Void'de saklanma) yeteneklerine sahiptir."
    },
    {
        "question": "Miracle Invoker nasıl mucize gerçekleştirir?",
        "ground_truth": "Miracle Invoker, önce başkalarının dileklerini yerine getirerek güç biriktirir, ardından kendi dilekleri için bu birikimi kullanarak mucize yaratır. Dilekler bozulmaya açıktır ve büyük dilekler daha fazla bozulur."
    },
]


def run_evaluation():
    """RAG sistemini test et ve RAGAS ile değerlendir."""
    
    print("="*60)
    print("🔮 LoTM Chatbot - RAGAS Değerlendirmesi")
    print("="*60)
    
    samples = []
    
    print("\n📝 Test soruları çalıştırılıyor...\n")
    
    for i, test_item in enumerate(TEST_QUESTIONS, 1):
        question = test_item["question"]
        ground_truth = test_item["ground_truth"]
        
        print(f"[{i}/{len(TEST_QUESTIONS)}] Soru: {question[:50]}...")
        
        try:
            # Sisteme soruyu sor
            result = app.invoke({"question": question})
            
            answer = result.get("generation", "")
            docs = result.get("documents", [])
            
            # Context'leri al
            context_list = [doc.page_content for doc in docs]
            
            # RAGAS SingleTurnSample oluştur
            sample = SingleTurnSample(
                user_input=question,
                response=answer,
                retrieved_contexts=context_list,
                reference=ground_truth
            )
            samples.append(sample)
            
            print(f"   ✅ Cevap alındı ({len(docs)} belge bulundu)")
            
        except Exception as e:
            print(f"   ❌ Hata: {e}")
    
    # RAGAS Dataset oluştur
    print("\n" + "="*60)
    print("📊 RAGAS Değerlendirmesi Başlıyor...")
    print("="*60)
    
    eval_dataset = EvaluationDataset(samples=samples)
    
    # Gemini LLM wrapper
    gemini_llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
    evaluator_llm = LangchainLLMWrapper(gemini_llm)
    
    # Metrikleri ayarla
    faithfulness_metric = Faithfulness(llm=evaluator_llm)
    context_recall_metric = LLMContextRecall(llm=evaluator_llm)
    relevancy_metric = ResponseRelevancy(llm=evaluator_llm)
    precision_metric = ContextPrecision(llm=evaluator_llm)
    
    metrics = [
        faithfulness_metric,    # Cevap belgelerden mi geldi?
        context_recall_metric,  # Bilgiyi bulma başarısı (Retrieval Recall)
        relevancy_metric,       # Cevap soruyla ne kadar alakalı?
        precision_metric,       # Context'in doğruluğu
    ]
    
    print("\n⏳ Değerlendirme yapılıyor (bu birkaç dakika sürebilir)...\n")
    
    try:
        results = evaluate(
            dataset=eval_dataset,
            metrics=metrics,
        )
        
        # Rapor
        print("\n" + "="*60)
        print("📈 DOĞRULUK RAPORU")
        print("="*60)
        
        print("\n🎯 GENEL SKORLAR:")
        print("-"*40)
        
        # DataFrame'den sonuçları al
        df = results.to_pandas()
        
        # Her metrik için ortalama hesapla
        metric_columns = ['faithfulness', 'llm_context_recall', 'response_relevancy', 'context_precision']
        
        for col in metric_columns:
            if col in df.columns:
                score = df[col].mean()
                emoji = "🟢" if score >= 0.7 else "🟡" if score >= 0.5 else "🔴"
                display_name = col.replace('_', ' ').title()
                print(f"{emoji} {display_name}: {score:.4f}")
        
        print("\n" + "-"*40)
        print("\n📋 ÖNCELİKLİ METRİKLER:")
        
        if 'faithfulness' in df.columns:
            faith_score = df['faithfulness'].mean()
            print(f"\n1️⃣  FAITHFULNESS (Belgeye Sadakat): {faith_score:.4f}")
            print("   → Cevaplar ne kadar belgelere dayalı?")
            if faith_score >= 0.8:
                print("   ✅ Mükemmel! Cevaplar belgelere sadık.")
            elif faith_score >= 0.6:
                print("   ⚠️ İyi ama iyileştirilebilir.")
            else:
                print("   ❌ Dikkat! Model hallüsinasyon yapıyor olabilir.")
        
        if 'llm_context_recall' in df.columns:
            recall_score = df['llm_context_recall'].mean()
            print(f"\n2️⃣  CONTEXT RECALL (Retrieval Recall): {recall_score:.4f}")
            print("   → Gerekli bilgi ne kadar başarıyla bulunuyor?")
            if recall_score >= 0.8:
                print("   ✅ Mükemmel! Retrieval sistemi çok iyi çalışıyor.")
            elif recall_score >= 0.6:
                print("   ⚠️ Orta düzey. Retrieval iyileştirilebilir.")
            else:
                print("   ❌ Zayıf. Retrieval sistemi gözden geçirilmeli.")
        
        # Detaylı sonuçları kaydet
        print("\n" + "="*60)
        print("💾 Detaylı sonuçlar 'ragas_results.csv' dosyasına kaydedildi.")
        print("="*60)
        
        df.to_csv("ragas_results.csv", index=False)
        
        # Soru bazlı detayları göster
        print("\n📊 SORU BAZLI DETAYLAR:")
        print("-"*60)
        for idx, row in df.iterrows():
            q = TEST_QUESTIONS[idx]["question"][:40] + "..."
            print(f"\n{idx+1}. {q}")
            for col in metric_columns:
                if col in df.columns:
                    val = row[col]
                    if isinstance(val, (int, float)) and not pd.isna(val):
                        emoji = "🟢" if val >= 0.7 else "🟡" if val >= 0.5 else "🔴"
                        print(f"   {emoji} {col}: {val:.3f}")
        
        return results
        
    except Exception as e:
        print(f"\n❌ Değerlendirme hatası: {e}")
        import traceback
        traceback.print_exc()
        return None


def quick_test():
    """Hızlı bir test yap - sadece sistemi kontrol et."""
    print("\n🧪 Hızlı Test - Sistem Kontrolü")
    print("-"*40)
    
    test_q = "Fool Pathway'de Sequence 9 Seer'ın yetenekleri nelerdir?"
    
    try:
        result = app.invoke({"question": test_q})
        print(f"✅ Sistem çalışıyor!")
        print(f"\nSoru: {test_q}")
        print(f"\nCevap: {result.get('generation', 'Cevap alınamadı')[:500]}...")
        print(f"\nBulunan belge sayısı: {len(result.get('documents', []))}")
        return True
    except Exception as e:
        print(f"❌ Sistem hatası: {e}")
        return False


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        quick_test()
    else:
        print("\n💡 Önce hızlı test yapmak için: python test_ragas.py --quick")
        print("   Tam değerlendirme için: python test_ragas.py\n")
        run_evaluation()
