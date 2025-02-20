import openai
import os
from dotenv import load_dotenv

# Load API key from environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def chatbot_response(user_input):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # Use "gpt-4" if you have access
            messages=[{"role": "system", "content": "You are a helpful assistant for crop recommendations."},
                      {"role": "user", "content": user_input}]
        )
        return response["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"Error: {str(e)}"

# Simple test
if __name__ == "__main__":
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break
        print("Bot:", chatbot_response(user_input))
