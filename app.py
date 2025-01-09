import streamlit as st
import onnxruntime as ort
import pickle
from transformers import CamembertTokenizer
import numpy as np

def main():
    st.title("Application de Classification de Texte avec CamemBERT (ONNX)")

    # Saisir le texte à prédire
    user_input = st.text_area("Entrez votre texte en français :", height=150)

    # Charger le label encoder
    with open("label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)

    # Charger le tokenizer
    tokenizer = CamembertTokenizer.from_pretrained("camembert-base")

    # Charger la session ONNX
    session = ort.InferenceSession("camembert_sequence_classification.onnx")

    if st.button("Prédire"):
        if user_input.strip():
            # Prétraiter le texte (tokenisation)
            inputs = tokenizer(
                user_input,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=128
            )

            # Convertir en numpy pour ONNX Runtime
            input_ids = inputs["input_ids"].numpy()
            attention_mask = inputs["attention_mask"].numpy()

            # Effectuer la prédiction avec le modèle ONNX
            outputs = session.run(None, {
                "input_ids": input_ids,
                "attention_mask": attention_mask
            })

            # outputs est un tuple contenant la sortie du réseau
            # Généralement, on obtient une liste de logits (une matrice 2D si batch > 1)
            logits = outputs[0]

            # Prendre l'argmax pour trouver la classe prédite
            predicted_class_id = np.argmax(logits, axis=1)[0]

            # Convertir l'indice en label
            predicted_label = label_encoder.inverse_transform([predicted_class_id])[0]

            st.success(f"Label prédit : {predicted_label}")
        else:
            st.warning("Veuillez saisir un texte avant de prédire.")

if __name__ == "__main__":
    main()
