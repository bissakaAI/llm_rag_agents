from google import genai
import os
from dotenv import load_dotenv

load_dotenv()
api=os.getenv('api_key')

client = genai.Client(api_key=api)
print("=="*10)
print("Welcom to HamidCo AI Assistant")
print("=="*10)


while True:
    prompt= input("User:")
    if prompt != 'exit':
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        print(response.text)


    else:
        print("Thank you for using Hmidco AI Assistant")
        break
    
    