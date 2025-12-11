import google.generativeai as genai
import os
import sys

# Get API key from env or args
api_key = os.environ.get("GOOGLE_API_KEY")
if len(sys.argv) > 1:
    api_key = sys.argv[1]

if not api_key:
    print("❌ No API Key provided")
    sys.exit(1)

genai.configure(api_key=api_key)

print(f"🔍 Checking available models for key: ...{api_key[-4:]}")

try:
    print("\n--- Available Models ---")
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(f"✅ {m.name}")
    print("------------------------")
except Exception as e:
    print(f"❌ Error listing models: {e}")
