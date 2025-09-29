import requests

url = 'http://localhost:9698/predict'

ride = {
    "PU_DO": "9_14",
    "trip_distance": 12.0
}

ride_2 = {
    "PU_DO": "7_14",
    "trip_distance": 80.0
}

response = requests.post(url, json=ride_2)

predictions = response.json()

print("Status code:", response.status_code)
print(f'estimated ride time: {predictions['duration_prediction']}')
