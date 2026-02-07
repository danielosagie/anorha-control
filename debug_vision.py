import requests
import base64
import json
import io
from PIL import Image

def test_vision():
    # 1. Create a tiny red test image
    img = Image.new('RGB', (100, 100), color='red')
    buf = io.BytesIO()
    img.save(buf, format='JPEG')
    img_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    
    url = "http://localhost:8081/v1/chat/completions"
    
    payload = {
        "model": "qwen3-vl:2b",  # Name doesn't matter much for llama.cpp usually
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "What color is this image?"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
            ]
        }],
        "max_tokens": 50
    }
    
    print(f"📡 Connecting to {url}...")
    try:
        response = requests.post(url, json=payload, timeout=10)
        
        print(f"\n✅ Status Code: {response.status_code}")
        print(f"📜 Headers: {json.dumps(dict(response.headers), indent=2)}")
        print(f"📦 Response Body:\n{response.text}")
        
        if response.status_code == 200:
            print("\n🎉 SUCCESS! The server accepts images.")
        else:
            print("\n❌ FAILURE! The server rejected the image.")
            print("   This confirms the server running on port 8081 does NOT have vision enabled.")
            
    except Exception as e:
        print(f"\n❌ CONNECTION ERROR: {e}")
        print("   Is the server running?")

if __name__ == "__main__":
    test_vision()
