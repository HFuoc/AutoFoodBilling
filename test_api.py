import requests

try:
    resp = requests.post(
        'http://127.0.0.1:8002/predict-cells',
        files={'file': ('demo-tray.png', open('frontend/public/demo-tray.png', 'rb'), 'image/png')}
    )
    print(resp.status_code)
    print(resp.text)
except Exception as e:
    print("Error:", e)