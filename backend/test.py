import requests

url = "http://127.0.0.1:5000/predict"

# Format recommandé pour les intervalles d'années: [start, end]
data =  {"years": None, "mass": None, "continents": ['Africa']}
 
response = requests.post(url, json=data)
print(response.json())