import streamlit as st
import spacy
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import json

# SpaCy ve VADER yükle
nlp = spacy.load("ner_model")  # Modelinizi buraya yükleyin
analyzer = SentimentIntensityAnalyzer()

# Sayfa yapılandırması
st.set_page_config(page_title="Teknofest NLP Projesi", layout="centered")

# Başlık ve stil
st.title("HİSAR")
st.markdown("""
<style>
    .main {
        background-color: #f5f5f5;
    }
    h1 {
        color: #2c3e50;
        text-align: center;
        font-family: 'Arial', sans-serif;
    }
    .stButton button {
        background-color: #2c3e50;
        color: white;
        border-radius: 5px;
        font-size: 18px;
    }
    .stTextArea textarea {
        font-size: 16px;
    }
</style>
""", unsafe_allow_html=True)

# Giriş alanı
st.subheader("Lütfen Yorumunuzu Girin")
text_input = st.text_area("Yorumunuzu buraya yazın...", height=150)

# Buton ve model analizi
if st.button("Analiz Et"):
    if text_input:
        # Metni analiz et
        doc = nlp(text_input)

        # Sonuçları saklamak için liste
        entity_list = []
        results = []

        # Entity'lere duygu analizi ekle
        for ent in doc.ents:
            # Entity etrafındaki metni al
            ent_text = ent.text
            ent_start = ent.start_char
            ent_end = ent.end_char
            ent_context = text_input[ent_start:ent_end]

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

        # JSON formatında çıktı oluştur
        output = {
            "entity_list": entity_list,
            "results": results
        }

        # JSON çıktısını yazdır
        st.subheader("Analiz Sonuçları")
        st.json(output)
    else:
        st.write("Lütfen bir yorum girin.")

# Alt bilgi
st.markdown("---")
st.markdown("© HİSAR")
