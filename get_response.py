import requests
url = "http://127.0.0.1:5000"
response = requests.get(url)
print("The status code is :", response.status_code)
print("The response is :", response.text)