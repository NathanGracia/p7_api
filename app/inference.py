import os
import pickle
from typing import Tuple

import numpy as np

#  recommande fortement d'utiliser joblib pour scikit-learn :
from joblib import load

class ModelService:
    #  gère le chargement et l'inférence du modèle
    def __init__(self, model_path: str = None):
        #  lit le chemin du modèle depuis une variable d'environnement si fournie
        self.model_path = os.getenv("MODEL_PATH", model_path or "models/model.pkl")
        #  charge le modèle au démarrage pour éviter les latences à chaud
        self.model = self._load_model(self.model_path)

    def _load_model(self, path: str):
        #  vérifie l'existence du fichier modèle
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found at: {path}")

        #  charge le pipeline sklearn (vectorizer + classifier) picklé
        with open(path, "rb") as f:
            #model = pickle.load(f)
            model = load(path)  # Si joblib est utilisé
        #  valide que le modèle dispose bien des méthodes attendues
        required = all(hasattr(model, m) for m in ["predict", "predict_proba"])
        if not required:
            raise TypeError("Loaded model does not implement predict/predict_proba")
        return model

    def predict(self, text: str) -> Tuple[int, float]:
        #  vérifie l'entrée utilisateur
        if text is None or not str(text).strip():
            raise ValueError("Input text must be a non-empty string")

        #  applique la prédiction (on suppose un pipeline compatible)
        # predict -> classe {0,1}, predict_proba -> [[p0, p1]]
        pred = self.model.predict([text])[0]
        proba = self.model.predict_proba([text])[0]

        #  déduit la proba associée à la classe positive (1)
        # Hypothèse: la classe positive est indexée par 1
        if hasattr(self.model, "classes_"):
            #  localise l'indice de la classe "1"
            classes = list(self.model.classes_)
            if 1 in classes:
                pos_index = classes.index(1)
            else:
                #  gère le cas où la classe positive est True ou "positive"
                if True in classes:
                    pos_index = classes.index(True)
                elif "positive" in classes:
                    pos_index = classes.index("positive")
                else:
                    #  fallback sur la proba max si l'étiquette est inconnue
                    pos_index = int(np.argmax(proba))
        else:
            pos_index = int(np.argmax(proba))

        pos_proba = float(proba[pos_index])

        return int(pred), pos_proba
