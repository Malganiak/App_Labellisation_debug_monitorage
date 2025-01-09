import streamlit as st
import onnxruntime as ort
import pickle
from transformers import CamembertTokenizer
import numpy as np

def main():
    st.title("Application de Classification de Texte avec CamemBERT (ONNX)")

    # Saisir le texte à prédire
    user_input = st.text_area("Entrez votre texte en français :", height=150)

    # Charger le label encoder ou les labels
    try:
        with open("label_encoder.pkl", "rb") as f:
            label_encoder = pickle.load(f)
    except Exception as e:
        st.error(f"Erreur lors du chargement de 'label_encoder.pkl': {e}")
        return

    # Charger le tokenizer
    tokenizer = CamembertTokenizer.from_pretrained("camembert-base")

    # Charger la session ONNX
    try:
        session = ort.InferenceSession("camembert_sequence_classification.onnx")
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle ONNX: {e}")
        return

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

            try:
                # Effectuer la prédiction avec le modèle ONNX
                outputs = session.run(None, {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask
                })

                # Récupérer les logits
                logits = outputs[0]

                # Prendre l'argmax pour trouver la classe prédite
                predicted_class_id = np.argmax(logits, axis=1)[0]

                # Vérifier si label_encoder est un vrai LabelEncoder
                if hasattr(label_encoder, "inverse_transform"):
                    # Utilisation d'un vrai LabelEncoder
                    predicted_label = label_encoder.inverse_transform([predicted_class_id])[0]
                elif isinstance(label_encoder, (list, np.ndarray)):
                    # Utilisation d'une liste ou d'un tableau de labels
                    label_mapping_dict = {idx: label for idx, label in enumerate(label_encoder)}
                    predicted_label = label_mapping_dict.get(predicted_class_id, "Inconnu")
                else:
                    st.error("Le label_encoder n'est pas compatible.")
                    return

                st.success(f"Label prédit : {predicted_label}")
            except Exception as e:
                st.error(f"Erreur lors de la prédiction : {e}")
        else:
            st.warning("Veuillez saisir un texte avant de prédire.")

if __name__ == "__main__":
    main()
