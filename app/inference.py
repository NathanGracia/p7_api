import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

class ModelService:
    def __init__(self, model_path: str = None, tokenizer_path: str = None):
        # Mise à jour vers l'extension .keras par défaut
        self.model_path = os.getenv("MODEL_PATH", model_path or "models/model_lstm_w2v.keras")
        self.tokenizer_path = os.getenv("TOKENIZER_PATH", tokenizer_path or "models/tokenizer.pickle")
        self.max_len = 64  # Doit correspondre à la taille utilisée lors de l'entraînement

        # Chargement unique au démarrage
        self.model = self._load_model(self.model_path)
        self.tokenizer = self._load_tokenizer(self.tokenizer_path)

    def _load_model(self, path: str):
        """Charge le modèle Keras avec gestion d'erreurs et compatibilité d'inférence."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Fichier modèle introuvable : {path}")

        try:
            # On utilise tf.keras.models.load_model pour une meilleure gestion du format natif .keras
            # compile=False est crucial car on n'a pas besoin des fonctions d'entraînement sur le serveur
            return load_model(path, compile=False)
        except Exception as e:
            print(f"Erreur lors du chargement du modèle (.keras) : {e}")
            raise e

    def _load_tokenizer(self, path: str):
        """Charge le dictionnaire tokenizer sauvegardé via pickle."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Tokenizer Pickle introuvable à : {path}")
        with open(path, "rb") as f:
            return pickle.load(f)

    def predict(self, text: str):
        """Prédit le sentiment d'un texte (0=Négatif, 1=Positif)."""
        if text is None or not str(text).strip():
            raise ValueError("Le texte ne peut pas être vide")

        # 1. Prétraitement (Tokenization + Padding)
        sequence = self.tokenizer.texts_to_sequences([text])
        padded = pad_sequences(sequence, maxlen=self.max_len, padding='post')

        # 2. Inférence (Sortie sigmoid entre 0 et 1)
        # verbose=0 évite les barres de progression dans les logs serveur
        proba = self.model.predict(padded, verbose=0)[0][0]

        # 3. Seuil de décision
        pred = 1 if proba >= 0.5 else 0

        return int(pred), float(proba)