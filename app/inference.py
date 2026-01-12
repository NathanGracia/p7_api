import os
import pickle

class ModelService:
    def __init__(self, model_path: str = None, vectorizer_path: str = None):
        self.model_path = os.getenv("MODEL_PATH", model_path or "models/logistic_model.pkl")
        self.vectorizer_path = os.getenv("VECTORIZER_PATH", vectorizer_path or "models/tfidf_vectorizer.pkl")

        # Chargement des fichiers pickle
        self.model = self._load_file(self.model_path)
        self.vectorizer = self._load_file(self.vectorizer_path)

    def _load_file(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Fichier introuvable : {path}")
        with open(path, "rb") as f:
            return pickle.load(f)

    def predict(self, text: str):
        if not text or not str(text).strip():
            raise ValueError("Le texte est vide")

        # 1. Transformer le texte avec le vectoriseur TF-IDF
        # On met le texte dans une liste [text] car le vectoriseur attend un itérable
        text_vectorized = self.vectorizer.transform([text])

        # 2. Prédiction (0 ou 1)
        prediction = self.model.predict(text_vectorized)[0]

        # 3. Probabilité (score de confiance)
        # predict_proba renvoie [[prob_classe_0, prob_classe_1]]
        proba = self.model.predict_proba(text_vectorized)[0][1]

        return int(prediction), float(proba)