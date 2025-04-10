from google import genai
from google.genai import types
from dotenv import load_dotenv
import  os

load_dotenv()
apiKey=os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=apiKey)

response = client.models.generate_content(model='gemini-2.0-flash',contents="Do you remember what question i have asked you 5 minutes ")

print(f"Response => {response.text}")