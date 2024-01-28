import streamlit as st
import numpy as np
import nltk
import heapq
import speech_recognition as sr
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer

nltk.download('punkt')
nltk.download('stopwords')

def main():
    # Initialize session state
    if 'is_recording' not in st.session_state:
        st.session_state.is_recording = False
        st.session_state.audio_data = []

    st.title("Echobyte ")

    col1, col2 = st.columns([1, 2])

    # Add GIF to the first column
    col1.image('https://media.tenor.com/DIQuMhQ7i_IAAAAj/voice-wikipedia-voice.gif', use_column_width=True)

    # Add description to the second column
    col2.markdown("**EchoByte: Transforming audio into actionable insights.** 🎙️✨")
    col2.markdown("Unleash the power of transformation as we convert spoken words into concise summaries, delivering essential takeaways. Experience a revolutionary approach to data understanding and make every word count.")

    # Audio-to-Text Section
    start_recording = st.button("Start Recording")
    stop_recording = st.button("Stop Recording")

    if start_recording and not stop_recording:
        st.session_state.is_recording = True
        st.session_state.audio_data = []

    if stop_recording:
        st.session_state.is_recording = False
        st.write("Recording stopped. Processing...")

        if st.session_state.audio_data:
            audio_data_bytes = np.concatenate(st.session_state.audio_data)
            audio = sr.AudioData(audio_data_bytes, 44100, 2)  # Assuming 44100 Hz sample rate and 2 channels

            try:
                # Convert audio to text using Google's speech recognition
                article = sr.Recognizer().recognize_google(audio)

                # Summarization logic using sumy
                parser = PlaintextParser.from_string(article, Tokenizer('english'))
                summarizer = LsaSummarizer()
                summary = summarizer(parser.document, sentences_count=10)

                st.subheader('Summary:')
                for sentence in summary:
                    st.write(f"- {sentence}")

            except sr.UnknownValueError:
                st.warning("Speech Recognition could not understand audio.")
            except Exception as e:
                st.error(f"Error: {str(e)}")

        else:
            st.warning("No audio data recorded.")

    if st.session_state.is_recording:
        st.write("Recording... Speak into the microphone.")

        # Record audio
        r = sr.Recognizer()

        with sr.Microphone() as source:
            r.adjust_for_ambient_noise(source)
            audio_data = r.listen(source, timeout=10)

        # Update the session state with the recorded audio data
        st.session_state.audio_data.append(np.frombuffer(audio_data.frame_data, dtype=np.int16))

if __name__ == "__main__":
    main()


