from gtts import gTTS
import streamlit as st
import io
from openai import OpenAI, OpenAIError
from dotenv import load_dotenv
import os

load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')


def transcribe_audio(audio):
    client = OpenAI(api_key=openai_api_key)
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
        st.write(e)  # log the exception in the terminal
    

def text_to_speech(response):
    try:
        tts = gTTS(text=response, lang='en')
        tts.save('response.wav')
    except:
        st.write("An error occured!")

def play_audio(file_name):
    try:
        audio_file = open(file_name, 'rb') 
        audio_bytes = audio_file.read()
        st.audio(audio_bytes,format="audio/wav",)
    except Exception as e:
        print(e)
        