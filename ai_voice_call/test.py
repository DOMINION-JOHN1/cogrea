from google.genai import types
from google import genai
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
client = genai.Client(api_key=api_key)

with open('shared image.jpg', 'rb') as f:
    image_bytes = f.read()

prompt=""" You are an AI that processes images of multiple-choice question sheets.
Your task is to:
1. **Extract** each question exactly as it appears.
2. **Extract** all the available options in the format `"A": "Option text"`.
3. **Identify** the option(s) circled by the student and record it under `"StudAnswer"` in the format `{ "OptionLetter": "Option text" }`.
4. Maintain the following JSON structure for your output:

```json
{
  "1": {
    "question": "Question text here",
    "options": {
      "A": "Option text",
      "B": "Option text",
      "C": "Option text"
    },
    "StudAnswer": {
      "A": "Option text"
    }
  },
  "2": {
    "question": "Question text here",
    "options": {
      "A": "Option text",
      "B": "Option text",
      "C": "Option text",
      "D": "Option text"
    },
    "StudAnswer": {
      "B": "Option text"
    }
  }
...more depending on the number of questions}"""
response = client.models.generate_content(
    model='gemini-2.5-flash',
    contents=[
      types.Part.from_bytes(
        data=image_bytes,
        mime_type='image/jpeg',
      ),
      prompt
    ]
  )

print(response.text)