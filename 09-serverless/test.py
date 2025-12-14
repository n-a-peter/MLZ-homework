






import requests

url = "http://localhost:8080/2015-03-31/functions/function/invocations"

hair_request = {"url": 'https://habrastorage.org/webt/yf/_d/ok/yf_dokzqy3vcritme8ggnzqlvwa.jpeg'}

result = requests.post(url, json=hair_request).json()
print(result)
