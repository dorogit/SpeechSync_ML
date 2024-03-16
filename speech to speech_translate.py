import torch
import torchaudio
from transformers import AutoProcessor, SeamlessM4TModel
import os
import speech_recognition as sr
import pyaudio
import wave

#change to make input audio 
def record_audio(output_file, channels=1, rate=44100, chunk=1024):
    audio = pyaudio.PyAudio()

    # Set up recording stream
    stream = audio.open(format=pyaudio.paInt16,
                        channels=channels,
                        rate=rate,
                        input=True,
                        frames_per_buffer=chunk)

    print("Recording... (Press Ctrl+C to stop)")

    frames = []
    try:
        while True:
            data = stream.read(chunk)
            frames.append(data)
    except KeyboardInterrupt:
        print("\nRecording stopped.")

    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    audio.terminate()

    # Save the recorded audio to a WAV file in the current directory
    with wave.open(output_file, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
        wf.setframerate(rate)
        wf.writeframes(b''.join(frames))

    print(f"Audio recorded and saved as WAV file: {output_file}")

if __name__ == "__main__":
    # Get the current directory
    current_directory = os.getcwd()

    # Output WAV file name (in the same directory as the script)
    output_wav_file = os.path.join(current_directory, "output_audio.wav")

    # Record audio
    record_audio(output_wav_file)
    
# Set the device to CUDA if available, otherwise fallback to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the processor and model, moving the model to the specified device
processor = AutoProcessor.from_pretrained("facebook/hf-seamless-m4t-medium")
model = SeamlessM4TModel.from_pretrained("facebook/hf-seamless-m4t-medium").to(device)

# Load the audio file and resample it
audio, orig_freq = torchaudio.load("output_audio.wav")
audio = torchaudio.functional.resample(audio, orig_freq=orig_freq, new_freq=16000)

# Before processing, move the audio tensor back to CPU
audio = audio.to("cpu")

# Now, process the audio using the processor
audio_inputs = processor(audios=audio, return_tensors="pt")

# Generate the audio output, move it to CPU, and convert to numpy array for saving
audio_output = model.generate(**audio_inputs.to(device), tgt_lang="hin")[0]
audio_array_from_audio = audio_output.cpu().numpy().squeeze()

audio_array_from_audio = audio_array_from_audio.reshape(1, -1)
# Save the output audio
torchaudio.save(
    "hindi.wav",
    torch.tensor(audio_array_from_audio),
    sample_rate=model.config.sampling_rate,
)
import torch

# Your existing code goes here...

# After a batch of processing
torch.cuda.empty_cache()