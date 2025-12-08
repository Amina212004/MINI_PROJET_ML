import requests

url = "http://127.0.0.1:5000/predict"
data = {
    "years": [1950, 1960],
    "mass": [100, 1000],
    "continents": ["Europe"]
}

response = requests.post(url, json=data)
print(response.json())