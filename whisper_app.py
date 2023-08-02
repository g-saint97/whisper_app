import streamlit as st
import whisper
import librosa
import numpy as np
from pydub import AudioSegment
import io
import resampy
import os 

# Set environment variable to suppress Tensorflow warning
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

st.title("Whisper App")

# Upload audio file with Streamlit
audio_file = st.file_uploader("Upload Audio", type=["wav", "mp3", "m4a"])

model = whisper.load_model("base")
st.text("Whisper Model Loaded")

if st.sidebar.button("Transcribe Audio"):
    if audio_file is not None:
        st.sidebar.success("Transcribing Audio")

        try:
            # Read the audio data using pydub
            audio_data = AudioSegment.from_file(io.BytesIO(audio_file.read()))

            # Convert to mono if the audio has multiple channels
            if audio_data.channels > 1:
                audio_data = audio_data.set_channels(1)

            # Get the sample rate
            sample_rate = audio_data.frame_rate

            # Convert audio data to numpy array
            audio_array = np.array(audio_data.get_array_of_samples())

            # Resample the audio to 16kHz using resampy.resample
            target_sample_rate = 16000
            resampled_audio = resampy.resample(audio_array, sample_rate, target_sample_rate)

            # Compute the mel spectrogram using librosa
            mel_spectrogram = librosa.feature.melspectrogram(y=resampled_audio, sr=target_sample_rate)

            # Transcribe the audio using the Whisper model
            transcription = model.transcribe(mel_spectrogram)

            st.sidebar.success("Transcription Complete")
            st.markdown(transcription["text"])
        except Exception as e:
            st.sidebar.error("Error during transcription. Please check the audio format.")
            print(e)
    else:
        st.sidebar.error("Please upload an audio file")

st.sidebar.header("Play Original Audio File")
st.sidebar.audio(audio_file)
