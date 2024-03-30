from gtts import gTTS
import streamlit as st
import io
from openai import OpenAI, OpenAIError
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')


def transcribe_audio(audio):
    """
    Transcribes audio to text using OpenAI whisper API.
    
    Args:
        audio (dict): Dictionary containing audio data.
    """
    client = OpenAI(api_key=openai_api_key)
    # Converting audio data to BytesIO object
    audio_bio = io.BytesIO(audio['bytes'])
    try:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_bio,
            language='eng'
        )
        output = transcript.text
        st.write(output)
        print("hello")
    except OpenAIError as e:
        st.write(e)  # log the exception in the Streamlit app
    

def text_to_speech(response):
    """
    Converts text to speech and saves it as a WAV file using gTTs library.
    
    Args:
        response (str): Text to be converted to speech.
    """
    try:
        tts = gTTS(text=response, lang='en')
        tts.save('response.wav')
    except:
        st.write("An error occured!")

def play_audio(file_name):
    """
    Plays audio file in the Streamlit app.
    
    Args:
        file_name (str): Name of the audio file.
    """
    try:
        audio_file = open(file_name, 'rb') 
        audio_bytes = audio_file.read()
        st.audio(audio_bytes,format="audio/wav",)
    except Exception as e:
        print(e)
        