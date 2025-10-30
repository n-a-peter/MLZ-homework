import requests

# url = "http://0.0.0.0:9696/predict"
url = "http://localhost:9696/predict"
client = {
    "lead_source": "organic_search",
    "number_of_courses_viewed": 4,
    "annual_income": 80304.0
}
conversion = requests.post(url, json=client).json()
print("Probability of converting = ", conversion)