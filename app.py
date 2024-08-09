import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# FastAPI uygulaması
app = FastAPI()

# Model ve tokenizer'ı yükleme
model_path = "./results"  # Eğittiğin modelin kaydedildiği yol
model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained(model_path)

# Modeli değerlendirme moduna al
model.eval()

class Item(BaseModel):
    text: str  # Analiz edilecek yorum
    entities: list  # Yorumdaki entity'ler

@app.post("/predict/", response_model=dict)
async def predict(item: Item):
    text = item.text
    entities = item.entities

    results = []
    for entity in entities:
        # Yorum ve entity'yi tokenize etme
        inputs = tokenizer(text, entity, return_tensors="pt", padding="max_length", truncation=True, max_length=128)

        # Model ile tahmin yapma
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

        # Sonuçları işleme
        predicted_class_id = torch.argmax(logits, dim=1).item()
        label_map = {0: 'olumlu', 1: 'olumsuz', 2: 'nötr'}
        predicted_sentiment = label_map[predicted_class_id]

        results.append({
            "entity": entity,
            "sentiment": predicted_sentiment
        })

    result = {
        "entity_list": entities,
        "results": results
    }

    return result

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
