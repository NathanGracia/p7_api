import os
import pickle


class ModelService:
    """
    Service qui gère le chargement du modèle ML et les prédictions.
    """

    def __init__(self, model_path: str = None, vectorizer_path: str = None):
        self.model_path = os.getenv("MODEL_PATH", model_path or "models/logistic_model.pkl")
        self.vectorizer_path = os.getenv("VECTORIZER_PATH", vectorizer_path or "models/tfidf_vectorizer.pkl")

        self.model = self._load_file(self.model_path)
        self.vectorizer = self._load_file(self.vectorizer_path)

    def _load_file(self, path: str):
        """Charge un fichier pickle depuis le disque."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Fichier introuvable : {path}")
        with open(path, "rb") as f:
            return pickle.load(f)

    def predict(self, text: str):
        """
        Prédit le sentiment d'un texte.
        Retourne le label (0 ou 1) et la probabilité associée.
        """
        if not text or not str(text).strip():
            raise ValueError("Le texte est vide")

        # Vectorisation du texte avec TF-IDF
        text_vectorized = self.vectorizer.transform([text])

        # Prédiction et récupération de la proba pour la classe positive
        prediction = self.model.predict(text_vectorized)[0]
        proba = self.model.predict_proba(text_vectorized)[0][1]

        return int(prediction), float(proba)