# app.py
import streamlit as st
import pickle


# Cargar modelo y vectorizador
with open("modelo_viral.pkl", "rb") as f:
    modelo = pickle.load(f)
with open("vectorizer_viral.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Configuración de la página
st.set_page_config(page_title="Predicción de Tuits Virales", layout="wide")

# Interfaz
st.title("Tuits Virales Predictor")
st.markdown("""
*¿Será tu próximo tuit un éxito?*  
Ingresa tu texto y descúbrelo.
""")

tweet = st.text_area("Escribe o pega tu tuit aquí:", height=150)

if st.button("Predecir"):
    if tweet:
        # Preprocesamiento mínimo (igual que en el entrenamiento)
        tweet_limpio = tweet.lower().replace("#", "").replace("@", "")
        X = vectorizer.transform([tweet_limpio])
        pred = modelo.predict(X)[0]
        
        # Mostrar resultado con estilo
        if pred:
            st.success("¡Viral!  (Alta probabilidad de muchos retuits)")
        else:
            st.error("No viral...  (Sigue intentando)")
        
     
    else:
        st.warning("Por favor, escribe un tuit primero.")