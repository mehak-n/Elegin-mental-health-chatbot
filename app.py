from flask import Flask, request, jsonify
from google.cloud import dialogflow_v2 as dialogflow  # Updated import
import os
import requests
from dotenv import load_dotenv
from urllib.parse import quote

# Load environment variables from .env file
load_dotenv()

# Initialize Flask app
app = Flask(__name__)



os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "dialogflow_credentials.json"


# Route for the home page
@app.route('/')
def index():
    return "Welcome to the Gemini AI Mental Health Chatbot!"

# Route to handle chatbot interactions
@app.route('/chat', methods=['POST'])
def chat():
    message = request.json.get('message')
    if not message:
        return jsonify({"error": "No message provided"}), 400

    # First process the message with Dialogflow
    dialogflow_response = detect_intent_text(message)

    # Then analyze the message with Gemini AI's NLU service
    gemini_nlu_response = gemini_nlu_analysis(message)

    # Get an empathy-based response from Gemini AI
    empathy_response = gemini_empathy_response(gemini_nlu_response)

    return jsonify({
        "dialogflow_response": dialogflow_response,
        "gemini_nlu_response": gemini_nlu_response,
        "empathy_response": empathy_response
    })

# Function to detect user intent with Dialogflow
def detect_intent_text(text, session_id="12345"):
    session_client = dialogflow.SessionsClient()
    session = session_client.session_path(os.getenv('DIALOGFLOW_PROJECT_ID'), session_id)

    text_input = dialogflow.TextInput(text=text, language_code='en')  # Updated usage
    query_input = dialogflow.QueryInput(text=text_input)

    response = session_client.detect_intent(session=session, query_input=query_input)

    return response.query_result.fulfillment_text

# Function to analyze text with Gemini AI's NLU service
def gemini_nlu_analysis(text):
    url = os.getenv('GEMINI_API_URL_NLU')
    headers = {
        "Authorization": f"Bearer {os.getenv('GEMINI_AI_API_KEY')}",
        "Content-Type": "application/json"
    }
    data = {"text": text}
    response = requests.post(url, json=data, headers=headers)

    if response.status_code == 200:
        return response.json()
    else:
        return {"error": "Failed to connect to Gemini NLU API"}

# Function to get empathy-based response from Gemini AI
def gemini_empathy_response(nlu_response):
    url = os.getenv('GEMINI_API_URL_EMPATHY')
    headers = {
        "Authorization": f"Bearer {os.getenv('GEMINI_AI_API_KEY')}",
        "Content-Type": "application/json"
    }
    data = {"nlu_response": nlu_response}
    response = requests.post(url, json=data, headers=headers)

    if response.status_code == 200:
        return response.json()
    else:
        return {"error": "Failed to connect to Gemini Empathy API"}

# Example usage of quote function for URL encoding
@app.route('/quote_test')
def quote_test():
    raw_url = 'https://example.com/some path'
    encoded_url = quote(raw_url)
    return f"Encoded URL: {encoded_url}"

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
