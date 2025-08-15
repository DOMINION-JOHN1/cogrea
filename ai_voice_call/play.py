import os
import io
from flask import Flask
from flask_sock import Sock
import vertexai
from vertexai.preview.generative_models import GenerativeModel, Part
from dotenv import load_dotenv
load_dotenv()

# Initialize Flask app and WebSocket extension
app = Flask(__name__)
sock = Sock(app)

# 1. Initialize Vertex AI
# This assumes the GOOGLE_APPLICATION_CREDENTIALS environment variable is set.
PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "project-cogrea-one")
LOCATION = os.environ.get("GCP_LOCATION", "us-central1")
vertexai.init(project=PROJECT_ID, location=LOCATION)

# 2. Load the Gemini 1.5 Flash model
# We set response_mime_type to audio/mp3 to get audio output.
model = GenerativeModel(
    "gemini-live-2.5-flash-preview-native-audio",
    generation_config={"response_mime_type": "audio/mp3"}
)

@sock.route("/stream-audio")
def stream_audio(ws):
    """
    Handles a WebSocket connection for real-time audio streaming with Gemini.
    """
    print("WebSocket connection established.")
    try:
        # A list to store the audio chunks received from the client.
        audio_chunks = []

        # 3. Receive streaming audio from the client.
        # This loop continues as long as the WebSocket connection is open.
        while True:
            # Receive a single audio chunk from the client.
            audio_data = ws.receive()

            # If the received data is a string, it's a control message.
            if isinstance(audio_data, str):
                if audio_data == 'end_of_audio':
                    break  # Break the loop to process the full audio.
                # You could add other control messages here, e.g., 'start_recording'.
                continue

            # Append the binary audio data to our list of chunks.
            audio_chunks.append(audio_data)

        print(f"Received {len(audio_chunks)} audio chunks from client.")

        # 4. Prepare the audio input for Gemini.
        # Concatenate all chunks into a single byte stream.
        full_audio_bytes = b"".join(audio_chunks)
        audio_part = Part.from_data(data=full_audio_bytes, mime_type="audio/webm")

        # 5. Create the prompt.
        prompt_with_audio = [
            "You are an AI assistant. I will provide an audio recording of a user's question. Respond to it with an audio answer. The response should be a friendly, conversational audio response.",
            audio_part,
        ]

        # 6. Stream the audio input to Gemini and get a streaming response.
        # The stream=True parameter is key to getting a streaming response.
        responses = model.generate_content(prompt_with_audio, stream=True)

        # 7. Stream the Gemini response back to the client.
        for response in responses:
            # Check if the response contains audio data.
            if response.candidates and response.candidates[0].content.parts:
                response_audio_bytes = response.candidates[0].content.parts[0].blob.data
                # Send the audio chunk back to the client immediately.
                ws.send(response_audio_bytes)

        print("Finished streaming response to client.")

    except Exception as e:
        print(f"An error occurred: {e}")
        ws.close()

if __name__ == "__main__":
    # The server will run on localhost port 5000.
    # To run this with WebSockets, you'll need a WSGI server like gevent.
    # Example: gevent-websocket is an option.
    # This simple run command is for development and might not be suitable for production.
    # Use gunicorn --worker-class geventwebsocket.gunicorn.workers.GeventWebSocketWorker -b 0.0.0.0:5000 your_app_name:app for production.
    from gevent import pywsgi
    from geventwebsocket.handler import WebSocketHandler
    server = pywsgi.WSGIServer(('0.0.0.0', 5000), app, handler_class=WebSocketHandler)
    server.serve_forever()