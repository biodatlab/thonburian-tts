'''
Demonstration of F5-TTS Engine for Thai language
'''
import logging
from pathlib import Path
import torch
from cached_path import cached_path
from flowtts.inference import FlowTTSPipeline, ModelConfig, AudioConfig
from transformers import pipeline

logging.basicConfig(level=logging.INFO)

def thonburian_whisper(audio_file, model_name = "biodatlab/whisper-th-medium-combined", lang="th"):
    '''
    ASR for transcribing the reference audio file.
    audio_file (str): reference audio file
    model_name (str): Whsiper ASR model
    lang (str): language code

    Return: Transcribed text of "audio_file"
    '''
    device = 0 if torch.cuda.is_available() else "cpu"
    pipe = pipeline(
        task="automatic-speech-recognition",
        model=model_name,
        chunk_length_s=30,
        device=device,
    )
    return pipe(str(audio_file), generate_kwargs={"language":f"<|{lang}|>", "task":"transcribe"}, batch_size=16)["text"]

def main():
    '''
    F5-TTS Demonstration for Text-to-Speech Generation
    '''
    # Configure model settings for F5
    model_config = ModelConfig(
        language="th", 
        model_type="F5",
        checkpoint="hf://ThuraAung1601/E2-F5-TTS/F5_Thai/mega_f5_last.safetensors",
        vocab_file="hf://ThuraAung1601/E2-F5-TTS/F5_Thai/mega_vocab.txt",
        ode_method="euler",
        use_ema=True,
        vocoder="vocos",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # Update audio config with F5-specific parameters
    audio_config = AudioConfig(
        silence_threshold=-45,
        max_audio_length=20000,
        cfg_strength=2.5, 
        nfe_step=32,
        target_rms=0.1,
        cross_fade_duration=0.15,
        speed=1.0,
        min_silence_len=500,
        keep_silence=200,
        seek_step=10
    )

    # Initialize pipeline
    pipeline = FlowTTSPipeline(
        model_config=model_config,
        audio_config=audio_config,
        temp_dir="temp_f5"
    )

    # Text for speech generation
    test_text = "ยินดีที่ได้รู้จักคุณวันนี้อากาศดีมาก"
    
    # Create output directory
    output_dir = Path("outputs_f5")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Reference data for Flow Matching based models to micmic
    ref_voice = str(cached_path("hf://ThuraAung1601/E2-F5-TTS/ref_samples/ref_sample.wav"))
    ref_text = thonburian_whisper(ref_voice)

    # Generate speech
    try:
        output_path = pipeline(
            text=test_text,
            ref_voice=ref_voice,
            ref_text=ref_text,
            output_file=str(output_dir / "f5_output.wav"),
            speed=1.0,
            check_duration=True
        )
        print(f"Generated F5 audio saved to: {output_path}")

    except Exception as e:
        logging.error(f"Error during speech synthesis: {e}")

if __name__ == "__main__":
    main()