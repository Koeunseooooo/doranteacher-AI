import requests
r = requests.post(
    "https://api.deepai.org/api/text2img",
    data={
        'text': 'apple',
    },
    headers={'api-key': '5e2df0ad-409e-450c-95a1-fb0e9b31fe01'}
)
print(r.json())