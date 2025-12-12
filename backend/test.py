import requests

url = "https://mini-projet-ml-e264.onrender.com/predict"

# Format recommandé pour les intervalles d'années: [start, end]
data =  {"years": None, "mass": ["1-10g"], "continents": None}
 
response = requests.post(url, json=data)
print(response.json())