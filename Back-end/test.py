import requests
import json

def test_register():
    url = "http://localhost:5000/api/register"
    data = {
        "username": "testuser",
        "password": "testpassword"
    }
    
    try:
        response = requests.post(
            url,
            json=data,
            headers={"Content-Type": "application/json"}
        )
        
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")
        print(f"Headers: {dict(response.headers)}")
        
    except Exception as e:
        print(f"请求失败: {e}")

if __name__ == "__main__":
    test_register()