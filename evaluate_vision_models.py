import requests
import base64
import json
import time
from PIL import Image, ImageDraw
import io

# 1. Models to test (based on user's list and standard ones)
CANDIDATES = [
    "qwen2.5vl:7b",       # User has this (no dash?)
    "qwen2.5-vl:7b",      # Standard name
    "qwen3-vl:2b",        # We were using this
    "qwen3-vl:4b",        # User listed this
    "llava",              # Fallback
    "moondream",          # Fast fallback
]

def create_test_image():
    img = Image.new('RGB', (500, 500), color='white')
    draw = ImageDraw.Draw(img)
    # Draw red circle at top-left (100, 100)
    draw.ellipse([50, 50, 150, 150], fill='red', outline='black')
    # Draw blue square at bottom-right (400, 400)
    draw.rectangle([350, 350, 450, 450], fill='blue', outline='black')
    
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return img_b64

def get_installed_models():
    try:
        res = requests.get("http://localhost:11434/api/tags", timeout=5)
        if res.status_code == 200:
            models = [m['name'] for m in res.json().get('models', [])]
            return models
    except Exception as e:
        print(f"⚠️ Could not list models: {e}")
    return []

def test_model(model_name, img_b64):
    print(f"\n🧪 Testing {model_name}...")
    
    prompt = "Find the red circle. Output center coordinates as JSON: {\"x\": <number>, \"y\": <number>}."
    
    payload = {
        "model": model_name,
        "messages": [
            {"role": "user", "content": prompt, "images": [img_b64]}
        ],
        "stream": False,
        "options": {"temperature": 0.0}
    }
    
    start = time.time()
    try:
        res = requests.post("http://localhost:11434/api/chat", json=payload, timeout=180)  # 3 min for cold start
        elapsed = time.time() - start
        
        if res.status_code == 200:
            content = res.json().get("message", {}).get("content", "").strip()
            print(f"   ⏱️ {elapsed:.2f}s | Status: 200")
            print(f"   📝 Output: {content}")
            
            # Simple check if meaningful
            if "x" in content and "y" in content:
                print("   ✅ Valid JSON response")
            elif "0, 0" in content:
                print("   ❌ Blind response (0,0)")
            else:
                print("   ⚠️ Non-JSON or unexpected response")
        else:
            print(f"   ❌ Error {res.status_code}: {res.text}")
            
    except Exception as e:
        print(f"   💥 Exception: {e}")

def main():
    print("🔍 Checking installed models...")
    installed = get_installed_models()
    print(f"   found: {installed}")
    
    img_b64 = create_test_image()
    print("MATCHING CANDIDATES:")
    
    found_any = False
    for cand in CANDIDATES:
        if cand in installed:
            found_any = True
            test_model(cand, img_b64)
    
    if not found_any:
        print("\n❌ No candidate models found in your Ollama library!")
        print(f"   User has: {installed}")
        print("   We need 'qwen2.5-vl:7b' or similar.")

if __name__ == "__main__":
    main()
