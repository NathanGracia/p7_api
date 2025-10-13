import requests

# définit l'URL de votre API locale
url = "http://127.0.0.1:8000/predict"

#url = "https://tweet-sentiment-api-gracia-ashmgea6ard7e4cf.francecentral-01.azurewebsites.net/predict"

# prépare un exemple de payload
payload = {"text": "je déteste vraiment ce film"}

# fait un appel POST
r = requests.post(url, json=payload)

print(r.status_code)
print(r.json())
