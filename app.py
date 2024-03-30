import streamlit as st
from streamlit_mic_recorder import mic_recorder
from langchain.memory import StreamlitChatMessageHistory
from chatbot import ChatBot
import speech_recognition as sr
from audio_utils import *

def set_send_input():
    st.session_state.send_input = True
    clear_input_field()

def clear_input_field():
    st.session_state.user_question = st.session_state.user_input
    st.session_state.user_input = ""

def main():
    st.title("Banking Customer Chat System ")
    chat_container = st.container()

    if "send_input" not in st.session_state:
        st.session_state.send_input = False
        st.session_state.user_question = ""

    chat_history = StreamlitChatMessageHistory(key = 'history')

    chatbot = ChatBot(chat_history=chat_history)

    user_input = st.text_input("Type your message", key="user_input", on_change=set_send_input)

    #Column for text field and voice recording field
    voice_recording_column, send_button_column = st.columns([4,1])

    #voice recording field
    with voice_recording_column:
        voice_recording = mic_recorder(start_prompt="Start Recording", stop_prompt="Stop Recording", key="voice_recording")

    with send_button_column:
        send_button = st.button("Send", key="send_button", on_click=clear_input_field)
    
    if voice_recording:
        try:
            transcribed_audio = transcribe_audio(voice_recording)
            response = chatbot.get_response(transcribed_audio)
            text_to_speech(response)
            play_audio("response.wav")
        except:
            pass

    if send_button or st.session_state.send_input:
        if st.session_state.user_question != "":
            response = chatbot.get_response(st.session_state.user_question)
            st.session_state.user_question = ""
            text_to_speech(response)
            play_audio("response.wav")

    if chat_history.messages != []:
        with chat_container:
            st.write("Chat History:")
            for message in chat_history.messages:
                st.chat_message(message.type).write(message.content)

if __name__ == "__main__":
    main()