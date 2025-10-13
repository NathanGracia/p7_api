# 🧠 Tweet Sentiment API – FastAPI + Azure App Service

Une API de classification de sentiments développée avec **FastAPI** et déployée automatiquement sur **Azure Web App** grâce à **GitHub Actions**.  
Ce projet illustre un workflow de CI/CD moderne pour le Machine Learning et la Data Science.

---

## 🚀 Fonctionnalités

- API REST **FastAPI** pour analyser le sentiment d’un texte (positif, neutre, négatif).  
- **Déploiement continu** automatique sur Azure à chaque `push` sur la branche `main`.  
- **Installation automatique des dépendances** pendant le déploiement grâce à Oryx.  
- Hébergement sur **Azure App Service Linux** avec un serveur de production **Gunicorn + UvicornWorker**.

---

## 🧩 Architecture du projet

```
.
├── app/
│   ├── main.py              # Point d’entrée FastAPI (contient l’objet app)
│   ├── model/               # Fichiers du modèle de Machine Learning
│   └── utils/               # Prétraitement, fonctions auxiliaires
├── requirements.txt         # Dépendances Python
├── .github/
│   └── workflows/
│       └── azure-webapp.yml # Workflow CI/CD GitHub Actions
└── README.md
```

---

## 🐍 Technologies utilisées

| Composant | Version / Description |
|------------|------------------------|
| Python | 3.12 |
| FastAPI | 0.115.0 |
| Uvicorn | 0.30.6 |
| Gunicorn | 21.2.0 |
| Scikit-learn | 1.4.2 |
| Azure Web App | Linux, Oryx build |
| GitHub Actions | CI/CD pipeline |

---

## ⚙️ Installation locale

1. **Cloner le dépôt**
   ```bash
   git clone https://github.com/<votre-utilisateur>/<votre-repo>.git
   cd <votre-repo>
   ```

2. **Créer un environnement virtuel**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # sous Linux/Mac
   .venv\Scripts\activate     # sous Windows
   ```

3. **Installer les dépendances**
   ```bash
   pip install -r requirements.txt
   ```

4. **Lancer l’API localement**
   ```bash
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

5. Ouvrez [http://localhost:8000/docs](http://localhost:8000/docs) pour accéder à la documentation interactive Swagger UI.

---

## ☁️ Déploiement sur Azure

Le déploiement est **automatique** dès qu’un commit est poussé sur `main`.

### 📦 Étapes principales du workflow
1. GitHub Actions récupère le code (`actions/checkout@v4`).
2. L’action `azure/login@v2` s’authentifie à Azure avec vos **secrets GitHub** :
   - `AZUREAPPSERVICE_CLIENTID_...`
   - `AZUREAPPSERVICE_TENANTID_...`
   - `AZUREAPPSERVICE_SUBSCRIPTIONID_...`
3. Azure exécute un **build Oryx** :
   - Crée un environnement virtuel `antenv`
   - Installe `requirements.txt`
4. Azure démarre l’API via :
   ```
   gunicorn -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000 app.main:app
   ```

---

## 🔑 Variables d’environnement (Azure)

Les paramètres peuvent être définis dans **Azure Portal → App Service → Configuration → Paramètres d’application**.

Exemple :
| Nom | Valeur | Description |
|------|---------|-------------|
| `MODEL_PATH` | `/home/site/wwwroot/models/model.pkl` | Chemin du modèle ML |
| `API_KEY` | `xxxxxxxxx` | Clé privée pour sécuriser l’API |

Elles sont accessibles dans le code via :
```python
import os
model_path = os.getenv("MODEL_PATH")
```

---

## 🧠 Exemple d’appel à l’API

```bash
curl -X POST "https://tweet-sentiment-api-gracia.azurewebsites.net/predict"      -H "Content-Type: application/json"      -d '{"text": "I love this project!"}'
```

Réponse :
```json
{
  "sentiment": "positive",
  "confidence": 0.94
}
```

---

## 🛠️ Dépannage

| Problème | Cause probable | Solution |
|-----------|----------------|-----------|
| `ModuleNotFoundError: No module named 'uvicorn'` | `gunicorn` ou `uvicorn` manquant dans `requirements.txt` | Ajouter `gunicorn` et `uvicorn[standard]` |
| `Could not find virtual environment 'antenv'` | Oryx non exécuté | Vérifier `SCM_DO_BUILD_DURING_DEPLOYMENT=1` |
| L’API ne démarre pas | Mauvais module dans le startup-command | Vérifier que `app.main:app` correspond bien à la variable FastAPI |

---

## 🧾 Licence

Ce projet est sous licence **MIT**.  
Vous pouvez l’utiliser librement pour vos propres projets éducatifs ou professionnels.
