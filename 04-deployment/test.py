import requests

url = 'http://localhost:9698/predict'

ride = {
    "PU_DO":"9_14",
    "trip_distance": 12.0
}

ride_2 = {
    "PU_DO": "9_14",
    "trip_distance": 40.0
}

response = requests.post(url, json=ride)

predictions = response.json()

print(f'estimated ride time: {predictions['duration_prediction']}')
