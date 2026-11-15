import os
import tempfile
import wave
from pathlib import Path

import gradio as gr
import numpy as np
import ollama
import soundfile as sf
from faster_whisper import WhisperModel
from piper import PiperVoice

# ====================== CONFIG ======================
OLLAMA_MODEL = "llama3.2:3b"          # Change to any local model you prefer
WHISPER_MODEL = "base"                # tiny / base / small / medium
DEFAULT_VOICE = "en_US-lessac-medium" # Good neutral starting voice
SYSTEM_PROMPT = """You are a helpful, natural-sounding AI assistant created and controlled by your owner.
Keep answers concise and conversational, suitable for spoken replies.
"""

# Directory where Piper voices will be stored
VOICES_DIR = Path("voices")
VOICES_DIR.mkdir(exist_ok=True)

# ====================== MODELS ======================
print("Loading Whisper...")
whisper = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")

print("Loading Piper voice...")
# This will auto-download the voice the first time
voice = PiperVoice.load(str(VOICES_DIR / f"{DEFAULT_VOICE}.onnx") if (VOICES_DIR / f"{DEFAULT_VOICE}.onnx").exists() 
                        else DEFAULT_VOICE)

# Cache for loaded voices
voice_cache = {DEFAULT_VOICE: voice}

# ====================== AVAILABLE VOICES ======================
# Popular English voices with different genders / tones
AVAILABLE_VOICES = {
    "Lessac (Neutral, clear – Medium)": "en_US-lessac-medium",
    "Amy (Female, warm)": "en_US-amy-medium",
    "Joe (Male, natural)": "en_US-joe-medium",
    "Kristin (Female, calm)": "en_US-kristin-medium",
    "Bryce (Male, deeper)": "en_US-bryce-medium",
    "HFC Female (Female, expressive)": "en_US-hfc_female-medium",
    "HFC Male (Male, expressive)": "en_US-hfc_male-medium",
    "Cori (British Female – High quality)": "en_GB-cori-high",
    "Alan (British Male)": "en_GB-alan-medium",
}

def load_voice(voice_name: str):
    """Load a Piper voice (download if needed)."""
    if voice_name in voice_cache:
        return voice_cache[voice_name]
    
    print(f"Loading voice: {voice_name}")
    try:
        # Piper can download automatically
        v = PiperVoice.load(voice_name, download_dir=str(VOICES_DIR))
        voice_cache[voice_name] = v
        return v
    except Exception as e:
        print(f"Failed to load {voice_name}: {e}")
        return voice_cache[DEFAULT_VOICE]

# ====================== CORE FUNCTIONS ======================
def transcribe(audio):
    """Convert microphone audio to text."""
    if audio is None:
        return ""
    
    sample_rate, data = audio
    if data.ndim > 1:
        data = data.mean(axis=1)  # mono
    
    # Save temporary wav
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        sf.write(f.name, data, sample_rate)
        segments, _ = whisper.transcribe(f.name, language="en")
        text = " ".join([s.text for s in segments]).strip()
        os.unlink(f.name)
    
    return text

def chat(message: str, history: list, system_prompt: str, model: str):
    """Send message to local Ollama model."""
    messages = [{"role": "system", "content": system_prompt}]
    
    for user_msg, assistant_msg in history:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": assistant_msg})
    
    messages.append({"role": "user", "content": message})
    
    try:
        response = ollama.chat(model=model, messages=messages)
        return response["message"]["content"]
    except Exception as e:
        return f"Error talking to Ollama: {e}\n\nMake sure Ollama is running and the model is pulled."

def synthesize(text: str, voice_key: str):
    """Convert text to speech using selected Piper voice."""
    if not text.strip():
        return None
    
    voice_id = AVAILABLE_VOICES.get(voice_key, DEFAULT_VOICE)
    current_voice = load_voice(voice_id)
    
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        with wave.open(f.name, "wb") as wav_file:
            current_voice.synthesize_wav(text, wav_file)
        return f.name

def process(
    audio,
    text_input,
    history,
    system_prompt,
    model_name,
    voice_choice,
):
    # Prefer typed text if provided, otherwise use microphone
    user_text = text_input.strip() if text_input and text_input.strip() else transcribe(audio)
    
    if not user_text:
        return history, None, history, ""
    
    # Get AI reply
    reply = chat(user_text, history, system_prompt, model_name)
    
    # Update history
    history = history + [[user_text, reply]]
    
    # Generate speech
    audio_path = synthesize(reply, voice_choice)
    
    return history, audio_path, history, ""

# ====================== GRADIO UI ======================
with gr.Blocks(title="Local Voice Chat", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # Local Voice Chat
        Fully offline • Controlled by you • Choose voice gender & tone
        
        **Requirements:**
        1. [Ollama](https://ollama.com) running
        2. At least one model pulled (e.g. `ollama pull llama3.2:3b`)
        """
    )
    
    with gr.Row():
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(height=450, label="Conversation")
            
            with gr.Row():
                audio_input = gr.Audio(
                    sources=["microphone"],
                    type="numpy",
                    label="Speak"
                )
                text_input = gr.Textbox(
                    placeholder="Or type here...",
                    label="Text input",
                    lines=2
                )
            
            with gr.Row():
                submit_btn = gr.Button("Send", variant="primary")
                clear_btn = gr.Button("Clear")
        
        with gr.Column(scale=1):
            voice_dropdown = gr.Dropdown(
                choices=list(AVAILABLE_VOICES.keys()),
                value="Lessac (Neutral, clear – Medium)",
                label="Voice (Gender / Tone)"
            )
            
            model_input = gr.Textbox(
                value=OLLAMA_MODEL,
                label="Ollama Model Name"
            )
            
            system_prompt = gr.Textbox(
                value=SYSTEM_PROMPT,
                label="System Prompt (Personality)",
                lines=6
            )
            
            audio_output = gr.Audio(label="AI Voice Reply", autoplay=True)
    
    state = gr.State([])
    
    submit_btn.click(
        fn=process,
        inputs=[audio_input, text_input, state, system_prompt, model_input, voice_dropdown],
        outputs=[chatbot, audio_output, state, text_input]
    )
    
    text_input.submit(
        fn=process,
        inputs=[audio_input, text_input, state, system_prompt, model_input, voice_dropdown],
        outputs=[chatbot, audio_output, state, text_input]
    )
    
    clear_btn.click(lambda: ([], None, [], ""), outputs=[chatbot, audio_output, state, text_input])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
