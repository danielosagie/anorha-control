import requests
import base64
import json
import io
import argparse
from PIL import Image

def test_vision(backend="ollama", url="http://localhost:11434", model="qwen3-vl:2b"):
    print(f"\n🔍 Testing VLM Planning Capabilities")
    print(f"   Backend: {backend}")
    print(f"   URL: {url}")
    print(f"   Model: {model}")
    
    # 1. Create a simulated "Login Form" image with specific instructions
    img = Image.new('RGB', (800, 600), color='white')
    from PIL import ImageDraw
    d = ImageDraw.Draw(img)
    # Draw "Accepted usernames" box
    d.rectangle([50, 50, 750, 150], outline='black')
    d.text((60, 60), "Login Form", fill="black")
    d.text((60, 80), "Accepted usernames: standard_user, locked_out_user", fill="red")
    # Draw input fields
    d.rectangle([200, 200, 600, 240], outline='gray')
    d.text((210, 210), "Username", fill="gray")
    d.rectangle([200, 260, 600, 300], outline='gray')
    d.text((210, 270), "Password", fill="gray")
    d.rectangle([200, 320, 300, 360], fill='blue')
    d.text((220, 330), "Login", fill="white")
    
    buf = io.BytesIO()
    img.save(buf, format='JPEG')
    img_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    
    # 2. Construct Prompt (Conflicting Task)
    # Task says "Alice", Screen says "standard_user" -> This triggers reasoning loops
    prompt = """Task: Login as Alice Williams (username: Alice_Williams)

Look at the screenshot and create SPECIFIC, ATOMIC steps for mouse/keyboard control.

RULES:
1. Each step = ONE click or ONE keyboard action
2. For typing: Include exact text in "value" field
3. Use coordinates visible in the current screen
4. For login: Use PROVIDED credentials exactly (unless screenshot overrides)

Output JSON array (IMMEDIATELY - DO NOT THINK):
[{"action": "click|type|scroll", "target": "specific element", "value": "text to type"}]"""

    payload = {}
    endpoint = ""
    
    # Use generous but safe limits for reasoning models
    max_tokens = 8000
    context_window = 16384 
    
    if backend == "ollama":
        endpoint = f"{url}/api/chat"
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt, "images": [img_b64]}],
            "stream": True,  # STREAMING ON: See output immediately
            "format": "json",  # FORCE JSON OUTPUT
            "think": False,  # Disable thinking so qwen3-vl outputs to content, not thinking field
            "options": {
                "temperature": 0.1, 
                "num_predict": max_tokens,
                "num_ctx": context_window
            }
        }
    else:  # llama.cpp (streaming not implemented yet for this script)
        endpoint = f"{url}/v1/chat/completions"
        payload = {
            "model": model,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
                ]
            }],
            "max_tokens": max_tokens,
            "temperature": 0.1
        }
    
    print(f"\n📡 Connecting to {endpoint} with context={context_window}, predict={max_tokens}...")
    try:
        content = ""
        thinking_buffer = ""
        is_thinking = False

        if backend == "ollama":
            with requests.post(endpoint, json=payload, stream=True, timeout=300) as response:
                print("\n🌊 Streaming response:")
                for line in response.iter_lines():
                    if line:
                        try:
                            chunk = json.loads(line.decode('utf-8'))
                            msg_chunk = chunk.get("message", {})
                            delta = msg_chunk.get("content", "")
                            
                            # Handle thinking
                            thinking_chunk = msg_chunk.get("thinking", "")
                            if thinking_chunk:
                                if not is_thinking:
                                    print("\n🧠 Thinking...", end="", flush=True)
                                    is_thinking = True
                                thinking_buffer += thinking_chunk
                                # Print dot every 100 chars of thinking
                                if len(thinking_buffer) % 100 < len(thinking_chunk):
                                    print(".", end="", flush=True)
                            
                            # Handle content
                            if delta:
                                if is_thinking:
                                    print("\n💡 Done thinking!\n")
                                    is_thinking = False
                                print(delta, end="", flush=True)
                                content += delta
                                
                            if chunk.get("done"):
                                print("\n\n✅ Done.")
                                if "eval_duration" in chunk:
                                    dur = chunk['eval_duration'] / 1e9
                                    cnt = chunk['eval_count']
                                    print(f"   Speed: {cnt/dur:.1f} tok/s ({cnt} tokens in {dur:.1f}s)")

                        except:
                            pass
        else:
            # Non-streaming fallback for llama.cpp
            response = requests.post(endpoint, json=payload, timeout=300)
            result = response.json()
            content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
        
        print(f"\n📦 RAW RESPONSE:\n{'-'*40}\n{content}\n{'-'*40}")
        
        # Try parsing
        import re
        array_match = re.search(r'\[[\s\S]*?\]', content)
        if array_match:
            print("\n✅ JSON Found!")
            data = json.loads(array_match.group())
            print(json.dumps(data, indent=2))
        else:
            print("\n❌ NO JSON ARRAY FOUND!")
            print("   The model is chatting instead of outputting JSON.")
            
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        if "response" in locals():
            print(f"Response text: {response.text}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--llamacpp", action="store_true")
    parser.add_argument("--url", type=str)
    parser.add_argument("--model", type=str, default="qwen3-vl:2b")
    args = parser.parse_args()
    
    backend = "llamacpp" if args.llamacpp else "ollama"
    url = args.url or ("http://localhost:8081" if args.llamacpp else "http://localhost:11434")
    
    test_vision(backend, url, args.model)
