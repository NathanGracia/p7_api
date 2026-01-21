# Tweet Sentiment API

API REST pour l'analyse de sentiment de tweets, construite avec FastAPI et déployée sur Azure App Service.

## Présentation

Cette API permet de classifier le sentiment d'un texte (positif ou négatif) grâce à un modèle de régression logistique entraîné sur des données de tweets. Le projet inclut un pipeline CI/CD complet avec GitHub Actions pour le déploiement automatique.

## Stack technique

| Composant | Version |
|-----------|---------|
| Python | 3.11+ |
| FastAPI | 0.115.0 |
| Uvicorn | 0.30.6 |
| Gunicorn | 21.2.0 |
| Scikit-learn | 1.3.2 |
| Azure Monitor | 1.6.4 |

## Structure du projet

```
p7_api/
├── app/
│   ├── main.py          # Point d'entrée FastAPI
│   ├── schemas.py       # Modèles Pydantic
│   └── inference.py     # Service de prédiction ML
├── models/
│   ├── logistic_model.pkl
│   └── tfidf_vectorizer.pkl
├── tests/
│   ├── test_api.py      # Tests unitaires
│   └── local_test.py    # Script de test manuel
├── .github/workflows/   # Pipeline CI/CD
├── Procfile
├── requirements.txt
└── README.md
```

## Installation

1. Cloner le dépôt :
```bash
git clone https://github.com/NathanGracia/p7_api.git
cd p7_api
```

2. Créer un environnement virtuel :
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. Installer les dépendances :
```bash
pip install -r requirements.txt
```

4. Lancer le serveur :
```bash
uvicorn app.main:app --reload
```

L'API est accessible sur `http://localhost:8000`. La documentation Swagger est disponible sur `/docs`.

## Endpoints

### GET /health
Vérifie que l'API fonctionne.

**Réponse :**
```json
{"status": "ok"}
```

### POST /predict
Analyse le sentiment d'un texte.

**Requête :**
```json
{"text": "I love this project!"}
```

**Réponse :**
```json
{
  "is_positive": true,
  "score": 0.94
}
```

### POST /feedback
Permet de signaler une prédiction incorrecte. Les feedbacks négatifs sont loggués sur Azure Monitor.

**Requête :**
```json
{
  "text": "I hate this",
  "prediction": true,
  "is_correct": false
}
```

## Tests

Lancer les tests avec pytest :
```bash
pytest tests/test_api.py -v
```

## Déploiement

Le déploiement est automatisé via GitHub Actions. À chaque push sur `main` :
1. Les tests sont exécutés
2. Si les tests passent, l'application est déployée sur Azure App Service

### Variables d'environnement

| Variable | Description |
|----------|-------------|
| `APPLICATIONINSIGHTS_CONNECTION_STRING` | Connexion Azure Monitor |
| `MODEL_PATH` | Chemin du modèle (optionnel) |
| `VECTORIZER_PATH` | Chemin du vectoriseur (optionnel) |

## Licence

MIT
