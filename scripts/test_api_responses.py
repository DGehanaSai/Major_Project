
import requests
import json

url = "http://localhost:8000/predict/predict"

def test_prediction(input_data, name):
    try:
        response = requests.post(url, json=input_data)
        if response.status_code == 200:
            res = response.json()
            print(f"[{name}] Yield: {res.get('prediction')} | Adv: {res.get('advisory_codes')}")
        else:
            print(f"[{name}] Failed: {response.text}")
    except Exception as e:
        print(f"[{name}] Error: {str(e)}")

# Case 1: Rice, Low Nitrogen (Should be low)
test_prediction({
    "crop": "Rice",
    "soil": "Clay",
    "temp": 30,
    "rain": 200,
    "ndvi": 0.5,
    "nitrogen": 20,
    "ph": 6.5
}, "Low Nitro Rice")

# Case 2: Maize, High Nitrogen (Should be higher)
test_prediction({
    "crop": "Maize",
    "soil": "Loam",
    "temp": 25,
    "rain": 100,
    "ndvi": 0.8,
    "nitrogen": 120,
    "ph": 7.0
}, "High Nitro Maize")

# Case 3: Identical to Case 1 to check determinism
test_prediction({
    "crop": "Rice",
    "soil": "Clay",
    "temp": 30,
    "rain": 200,
    "ndvi": 0.5,
    "nitrogen": 20,
    "ph": 6.5
}, "Repeat Low Rice")
