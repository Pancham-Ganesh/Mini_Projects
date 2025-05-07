import string
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import logging
import os
import shutil
import subprocess
import tempfile
import whisper
import openai

app = Flask(__name__)
CORS(app)  # Enable CORS for all domains

# Load environment variables from .env file
load_dotenv()

# Set up logging for detailed error output
logging.basicConfig(level=logging.DEBUG)

# Load Whisper model
model = whisper.load_model("base")

# Set your OpenAI API Key
openai.api_key = os.getenv('OPENAI_API_KEY')

import openai

@app.route('/convert-to-isl', methods=['POST'])
def convert_to_isl():
    try:
        # Log request data
        data = request.get_json()
        logging.info("Request data: %s", data)

        # Retrieve the transcribed speech
        transcribed_speech = data.get("transcribedSpeech", "")
        if not transcribed_speech:
            return jsonify({"error": "No transcribed speech provided"}), 400

        logging.info("Received transcribed speech: %s", transcribed_speech)
        logging.info("Sending text to OpenAI for ISL grammar conversion...")

        # Use the correct ChatCompletion interface
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert linguist specializing in Indian Sign Language (ISL). "
                        "Your task is to convert spoken English sentences into ISL-compatible grammar using a predefined vocabulary. "
                        "The ISL translation must only use the following words:\n\n"
                        "0, 1, 2, 3, 4, 5, 6, 7, 8, 9, A, AFTER, AGAIN, AGAINST, AGE, ALL, ALONE, ALSO, AND, ASK, AT, B, BE, "
                        "BEAUTIFUL, BEFORE, BEST, BETTER, BUSY, BUT, BYE, C, CAN, CANNOT, CHANGE, COLLEGE, COME, COMPUTER, D, "
                        "DAY, DISTANCE, DO, E, EAT, ENGINEER, F, FIGHT, FINISH, FROM, G, GLITTER, GO, GOD, GOLD, GOOD, GREAT, H, "
                        "HAND, HANDS, HAPPY, HELLO, HELP, HER, HERE, HIS, HOME, HOMEPAGE, HOW, I, INVENT, IT, J, K, KEEP, L, "
                        "LANGUAGE, LAUGH, LEARN, M, ME, MORE, MY, N, NAME, NEXT, NOT, NOW, O, OF, ON, OUR, OUT, P, PRETTY, Q, R, "
                        "RIGHT, S, SAD, SAFE, SEE, SELF, SIGN, SING, SO, SOUND, STAY, STUDY, T, TALK, TELEVISION, THANK, THAT, "
                        "THEY, THIS, THOSE, TIME, TO, TYPE, U, US, V, W, WALK, WASH, WAY, WE, WELCOME, WHAT, WHEN, WHERE, WHICH, "
                        "WHO, WHOLE, WHOSE, WHY, WILL, WITH, WITHOUT, WORDS, WORK, WORLD, WRONG, X, Y, YOU, YOUR, YOURSELF, Z.\n\n"
                        "If any word in the input sentence is not in this list, replace it with a word from the list that has a "
                        "similar meaning or is the closest match. Simplify the sentence structure by emphasizing key ideas, "
                        "removing unnecessary articles, auxiliary verbs, and aligning it with ISL grammar conventions."
                    )
                },
                {
                    "role": "user",
                    "content": f"Convert the following English text into ISL-compatible grammar: {transcribed_speech}"
                }
            ]
        )


        # Extract the response and strip any leading/trailing whitespace
        isl_text = response['choices'][0]['message']['content'].strip()

        # Remove punctuation from the extracted response
        isl_text_no_punctuation = isl_text.translate(str.maketrans('', '', string.punctuation))

        # Log the punctuation-free ISL grammar text
        logging.info("ISL grammar conversion successful: %s", isl_text_no_punctuation)

        return jsonify({"islText": isl_text_no_punctuation}), 200

    except Exception as e:
        logging.error("Error during ISL grammar conversion using OpenAI: %s", str(e))
        return jsonify({"error": "Failed to convert text to ISL grammar"}), 500


@app.route('/test', methods=['GET'])
def test():
    return jsonify({"message": "Server is accessible!"}), 200


@app.route('/speech-to-text', methods=['POST'])
def speech_to_text():
    try:
        # Log the request headers and body
        logging.info("Received request with headers: %s", request.headers)
        logging.info("Request form data: %s", request.form)
        logging.info("Request files data: %s", request.files)
        
        # Get the audio data from the request
        audio_file = request.files.get('audio')

        if audio_file.content_type not in ["audio/wav", "audio/x-wav", "audio/wave", "audio/vnd.wave"]:
            logging.error(f"Unexpected MIME type: {audio_file.content_type}")
            return jsonify({"error": f"Unsupported audio format: {audio_file.content_type}"}), 400


        if not audio_file or len(audio_file.read()) == 0:
            logging.error("Received an empty or invalid audio file.")
            return jsonify({"error": "Invalid audio file"}), 400

        if audio_file:
            logging.info("Audio file received: %s", audio_file.filename)
            logging.info("Audio file content type: %s", audio_file.content_type)
            audio_file.seek(0)  # Ensure pointer is reset
            logging.info("Audio file size: %d bytes", len(audio_file.read()))
        else:
            logging.error("No audio file received")
        
        if not audio_file:
            logging.error("No audio file provided")
            return jsonify({"error": "No audio file provided"}), 400

        # Log details of the received audio file
        logging.info("Received audio file: %s", audio_file.filename)
        audio_file.seek(0)  # Reset file pointer before saving
        logging.info("Audio file size: %d bytes", len(audio_file.read()))
        
        # Save the audio file temporarily
        temp_amr_path = os.path.join(tempfile.gettempdir(), "temp_audio.amr")
        audio_file.seek(0)  # Reset file pointer to the start before saving
        audio_file.save(temp_amr_path)

        # Log the file path where the audio file is saved
        logging.info("Audio file saved to: %s", temp_amr_path)

        # Ensure file is saved and accessible by copying to a new location
        temp_wav_path = os.path.join(tempfile.gettempdir(), "temp_audio.wav")

        # Convert AMR to WAV using ffmpeg
        try:
            logging.info("Converting AMR to WAV using ffmpeg...")
            # Define the ffmpeg command
            command = [
                "ffmpeg","-y", "-i", temp_amr_path,  # Input file
                "-ar", "16000",  # Set the sample rate
                "-ac", "1",  # Set the number of channels (mono)
                "-f", "wav",  # Set the format to WAV
                temp_wav_path  # Output file
            ]
            # Execute the command
            subprocess.run(command, check=True)
            logging.info("AMR to WAV conversion successful. WAV file saved to: %s", temp_wav_path)
        except Exception as e:
            logging.error(f"Error converting AMR to WAV using ffmpeg: {str(e)}")
            return jsonify({"error": "Failed to convert AMR to WAV"}), 500

        # Transcribe the WAV audio using Whisper model
        try:
            logging.info("Detecting language and transcribing...")
            result = model.transcribe(temp_wav_path, task="translate")  # task="transcribe" enables language detection
            detected_language = result.get("language")
            transcription = result['text']
            logging.info("Detected language: %s", detected_language)
            logging.info("Transcription: %s", transcription)
        except Exception as e:
            logging.error(f"Error during transcription: {str(e)}")
            return jsonify({"error": "Failed to transcribe audio"}), 500

        # Return the transcription result
        return jsonify({"transcription": transcription}), 200

    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")
        return jsonify({"error": "An unexpected error occurred"}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=4000)
