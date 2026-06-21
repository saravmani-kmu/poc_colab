import os
import sys
import torch
import soundfile as sf
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


MODEL_NAME = "ai4bharat/indic-parler-tts"
OUTPUT_DIR = "output"
SPEAKER = "Jaya"
DESCRIPTION = f"{SPEAKER}'s voice is clear and natural with a warm, conversational tone. The recording is of very high quality with no background noise."


def generate_tamil_speech(text, output_filename="tamil_speech.wav"):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, output_filename)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("Loading model...")
    hf_token = os.environ.get("HF_TOKEN")
    model = ParlerTTSForConditionalGeneration.from_pretrained(MODEL_NAME, token=hf_token).to(device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=hf_token)
    description_tokenizer = AutoTokenizer.from_pretrained(
        model.config.text_encoder._name_or_path
    )

    print(f"Generating speech for: {text}")
    description_input_ids = description_tokenizer(DESCRIPTION, return_tensors="pt").to(device)
    prompt_input_ids = tokenizer(text, return_tensors="pt").to(device)

    generation = model.generate(
        input_ids=description_input_ids.input_ids,
        attention_mask=description_input_ids.attention_mask,
        prompt_input_ids=prompt_input_ids.input_ids,
        prompt_attention_mask=prompt_input_ids.attention_mask,
    )

    audio_arr = generation.cpu().numpy().squeeze()
    sf.write(output_path, audio_arr, model.config.sampling_rate)
    print(f"Audio saved: {output_path} ({os.path.getsize(output_path)} bytes)")
    return output_path


def run():
    samples = [
        ("வணக்கம், நீங்கள் எப்படி இருக்கிறீர்கள்?", "greeting.wav"),
        ("இன்று வானிலை மிகவும் அழகாக இருக்கிறது.", "weather.wav"),
        ("நன்றி, மீண்டும் சந்திப்போம்.", "farewell.wav"),
    ]

    generated_files = []
    for text, filename in samples:
        path = generate_tamil_speech(text, filename)
        generated_files.append(path)

    print(f"\nGenerated {len(generated_files)} audio files in '{OUTPUT_DIR}/'")
    return generated_files


if __name__ == "__main__":
    run()
