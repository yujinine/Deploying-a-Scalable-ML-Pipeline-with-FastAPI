import json
import requests

# TODO: send a GET using the URL http://127.0.0.1:8000
r = requests.get("http://127.0.0.1:8000")
print("Status Code:", r.status_code)

# TODO: print the status code
# If the status code is 200, print the JSON response
if r.status_code == 200:
    print("Response:", r.json())
else:
    # If the request failed, print the error text
    print("GET Request failed:", r.text)

# Data to be sent with the POST request
data = {
    "age": 37,
    "workclass": "Private",
    "fnlgt": 173986,
    "education": "HS-grad",
    "education-num": 10,
    "marital_status": "Married-civ-spouse",
    "occupation": "Prof-specialty",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital-gain": 0,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native_country": "United-States"
}


# TODO: send a POST using the data above
r = requests.post("http://127.0.0.1:8000/infer", json=data)
print("Status Code:", r.status_code)

# TODO: print the status code
# If the status code is 200, print the JSON result
if r.status_code == 200:
    print("Result:", r.json())
else:
    # If the request failed, print the error text
    print("POST Request failed:", r.text)
