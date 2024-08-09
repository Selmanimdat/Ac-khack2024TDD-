import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field
import spacy
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

app = FastAPI()

# SpaCy ve VADER modellerini yükle
nlp = spacy.load("ner_model")  # Kendi modelinizi yükleyin
analyzer = SentimentIntensityAnalyzer()

# Veri modeli
class Item(BaseModel):
    text: str = Field(..., example="""Fiber 100mb SuperOnline kullanıcısıyım yaklaşık 2 haftadır @Twitch @Kick_Turkey gibi canlı yayın platformlarında 360p yayın izlerken donmalar yaşıyoruz. Başka hiç bir operatörler bu sorunu yaşamazken ben parasını verip alamadığım hizmeti neden ödeyeyim ? @Turkcell """)

@app.post("/predict/", response_model=dict)
async def predict(item: Item):
    text = item.text
    
    # Metni analiz et
    doc = nlp(text)
    
    # Sonuçları saklamak için liste
    entity_list = []
    results = []

    # Entity'lere duygu analizi ekle
    for ent in doc.ents:
        ent_text = ent.text
        ent_start = ent.start_char
        ent_end = ent.end_char
        ent_context = text[ent_start:ent_end]

        # Duygu analizi yap
        ent_sentiment = analyzer.polarity_scores(ent_context)
        sentiment_label = 'nötr'
        if ent_sentiment['compound'] >= 0.05:
            sentiment_label = 'olumlu'
        elif ent_sentiment['compound'] <= -0.05:
            sentiment_label = 'olumsuz'

        # Entity ve duygu etiketini sakla
        entity_list.append(ent_text)
        results.append({
            "entity": ent_text,
            "sentiment": sentiment_label
        })

    # JSON formatında sonuçları döndür
    result = {
        "entity_list": entity_list,
        "results": results
    }

    return result

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
