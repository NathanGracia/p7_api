# crée un modèle de test et l'enregistre en pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
import joblib

# crée un pipeline très simple : vectorizer + logistic regression
X = ["ce tweet est génial", "je déteste ce film", "quelle belle journée", "c'est horrible"]
y = [1, 0, 1, 0]

pipeline = make_pipeline(
    CountVectorizer(),
    LogisticRegression()
)

# entraîne le modèle sur les exemples jouets
pipeline.fit(X, y)

# sauvegarde le modèle pour vos tests d’API
joblib.dump(pipeline, "models/model_test.pkl")
print("✔️ Modèle de test sauvegardé dans models/model_test.pkl")
