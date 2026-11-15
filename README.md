# Local Voice Chat

Fully offline AI voice + text chat that **you** control.

- Speak with your microphone **or** type text
- Local LLM via [Ollama](https://ollama.com) (use any model, fine-tune it, change personality)
- High-quality offline TTS with [Piper](https://github.com/rhasspy/piper) — choose different genders, tones and accents
- Speech-to-text with [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
- Clean Gradio interface

Perfect for people who want a private, local assistant that can talk and listen.

## Features

- Microphone input + text input
- Multiple voice options (male / female / British / American / different styles)
- Editable system prompt (full control over personality)
- Switch Ollama models on the fly
- Conversation history
- 100% local after setup

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com) installed and running
- Microphone

## Quick Start

```bash
# 1. Clone
git clone https://github.com/Mohamadx2/local-voice-chat.git
cd local-voice-chat

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Pull a model (example)
ollama pull llama3.2:3b

# 5. Run
python app.py
```

Then open **http://localhost:7860**

## Available Voices

You can choose between different genders and tones in the UI:

| Voice | Style |
|-------|-------|
| Lessac | Neutral, clear |
| Amy | Female, warm |
| Joe | Male, natural |
| Kristin | Female, calm |
| Bryce | Male, deeper |
| HFC Female / Male | More expressive |
| Cori | British Female (high quality) |
| Alan | British Male |

More Piper voices can be added easily.

## Customize

- Change the **System Prompt** in the sidebar to shape the personality
- Change the Ollama model name to use your own fine-tuned model
- Add more voices by editing the `AVAILABLE_VOICES` dictionary in `app.py`

## License

MIT
