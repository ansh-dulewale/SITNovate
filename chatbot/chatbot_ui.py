import google.generativeai as genai
import os
import gradio as gr
from dotenv import load_dotenv

# Load API key from environment variables
load_dotenv()
genai.configure(api_key=os.getenv("store.env"))

def chatbot_response(user_input):
    try:
        model = genai.GenerativeModel("gemini-pro")  # Use Gemini Pro model
        response = model.generate_content(user_input)
        return response.text
    except Exception as e:
        return f"Error: {str(e)}"

# Gradio UI
iface = gr.Interface(
    fn=chatbot_response,
    inputs=gr.Textbox(lines=2, placeholder="Ask about crops..."),
    outputs=gr.Textbox(label="Chatbot Response"),
    title="Crop Advisor Chatbot",
    description="Ask any questions related to crop yield, soil, and recommendations."
)

# Run the app
if __name__ == "__main__":
    iface.launch()
