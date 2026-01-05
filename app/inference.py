import os
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


class ModelService:
    def __init__(self, model_path: str = None, tokenizer_path: str = None):
        # Chemins par défaut pointant vers tes nouveaux fichiers
        self.model_path = os.getenv("MODEL_PATH", model_path or "models/model_lstm_w2v.h5")
        self.tokenizer_path = os.getenv("TOKENIZER_PATH", tokenizer_path or "models/tokenizer.pickle")
        self.max_len = 64  # Identique à ton entraînement

        # Chargement unique au démarrage pour la rapidité
        self.model = self._load_model(self.model_path)
        self.tokenizer = self._load_tokenizer(self.tokenizer_path)

    def _load_model(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Modèle H5 introuvable à : {path}")
        # Charge le modèle Keras/TensorFlow
        return load_model(path)

    def _load_tokenizer(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Tokenizer Pickle introuvable à : {path}")
        with open(path, "rb") as f:
            return pickle.load(f)

    def predict(self, text: str):
        if text is None or not str(text).strip():
            raise ValueError("Le texte ne peut pas être vide")

        # 1. Prétraitement : Texte -> Séquence -> Padding
        # On utilise le tokenizer chargé au démarrage
        sequence = self.tokenizer.texts_to_sequences([text])
        padded = pad_sequences(sequence, maxlen=self.max_len, padding='post')

        # 2. Inférence : Calcul du score (0 à 1)
        # verbose=0 pour la performance en API
        proba = self.model.predict(padded, verbose=0)[0][0]

        # 3. Décision (Seuil à 0.5)
        # 1 = Positif, 0 = Négatif
        pred = 1 if proba >= 0.5 else 0

        # On renvoie l'étiquette et la probabilité associée
        return int(pred), float(proba)