"""
Script de test manuel pour vérifier l'API en local.
"""
import requests

# URL de l'API (local ou prod)
url = "http://127.0.0.1:8000/predict"
# url = "https://tweet-sentiment-api-gracia-ashmgea6ard7e4cf.francecentral-01.azurewebsites.net/predict"

payload = {"text": "je déteste vraiment ce film"}

r = requests.post(url, json=payload)
print(r.status_code)
print(r.json())
