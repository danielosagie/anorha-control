import requests
import base64
import json
import time
from PIL import Image, ImageDraw

def test_vision():
    # 1. Create a simple test image (red circle on white background)
    img = Image.new('RGB', (500, 500), color='white')
    draw = ImageDraw.Draw(img)
    draw.ellipse([200, 200, 300, 300], fill='red', outline='black')
    
    # Save and convert to base64
    img.save("test_circle.png")
    with open("test_circle.png", "rb") as f:
        img_bytes = f.read()
        base64_img = base64.b64encode(img_bytes).decode('utf-8')
    
    print("✅ Created test image (red circle at center 250,250)")
    
    # 2. Define payload for qwen3-vl:2b using /api/chat
    payload = {
        "model": "qwen3-vl:2b",
        "messages": [
            {
                "role": "user",
                "content": "Where is the red circle? Output center coordinates as JSON {x, y}.",
                "images": [base64_img]
            }
        ],
        "stream": False,
        "options": {"temperature": 0.0}
    }
    
    # 3. Call Ollama
    print("⏳ Calling Ollama (qwen3-vl:2b)...")
    start = time.time()
    
    try:
        response = requests.post("http://localhost:11434/api/chat", json=payload, timeout=60)
        elapsed = time.time() - start
        
        print(f"⏱️ Time: {elapsed:.2f}s")
        print(f"📥 Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            content = result.get("message", {}).get("content", "NO CONTENT")
            print(f"📝 Response:\n{content}")
        else:
            print(f"❌ Error: {response.text}")
            
    except Exception as e:
        print(f"💥 Exception: {e}")

if __name__ == "__main__":
    test_vision()
