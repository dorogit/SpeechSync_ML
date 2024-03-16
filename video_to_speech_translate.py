import torch
import torchaudio
from transformers import AutoProcessor, SeamlessM4TModel
from moviepy.editor import VideoFileClip


#function to convert video to WAV audio file  
def video_to_audio(video_path, audio_path):
    # Load the video file
    video_clip = VideoFileClip(video_path)
    
    # Extract audio from the video clip
    audio_clip = video_clip.audio
    
    # Write the audio to a WAV file
    audio_clip.write_audiofile(audio_path, codec='pcm_s16le')
    
    # Close the video and audio clips
    video_clip.close()
    audio_clip.close()
 # Main function 
if __name__ == "__main__":
    video_file = "video og.mp4"
    audio_file = "output1.wav"
    #convert video to audio 
    video_to_audio(video_file, audio_file)

# Set the device to CUDA if available, otherwise fallback to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the processor and model, moving the model to the specified device
processor = AutoProcessor.from_pretrained("facebook/hf-seamless-m4t-medium")
model = SeamlessM4TModel.from_pretrained("facebook/hf-seamless-m4t-medium").to(device)

# Load the audio file and resample it
audio, orig_freq = torchaudio.load("output1.wav")
audio = torchaudio.functional.resample(audio, orig_freq=orig_freq, new_freq=16000)

# Before processing, move the audio tensor back to CPU
audio = audio.to("cpu")

# Now, process the audio using the processor
audio_inputs = processor(audios=audio, return_tensors="pt")

# After processing the audio, clear memory
del audio
torch.cuda.empty_cache()

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

# After a batch of processing
torch.cuda.empty_cache()