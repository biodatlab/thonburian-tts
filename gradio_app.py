import gradio as gr
import torch
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os
import soundfile as sf
from flowtts.inference import FlowTTSPipeline, ModelConfig, AudioConfig

# Determine device
if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

# Set base path for relative files
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
logo_path = os.path.join(SCRIPT_DIR, "assets/ThonburianTTSLogo.png")
default_audio_path = os.path.join(SCRIPT_DIR, "assets/000000.wav")

def mel_generator(wav_path, mel_spectrogram_file):
    y, sr = librosa.load(wav_path, sr=None)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    S_dB = librosa.power_to_db(S, ref=np.max)

    os.makedirs(os.path.dirname(mel_spectrogram_file), exist_ok=True)

    plt.figure(figsize=(16, 8))
    librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', fmax=8000)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel-frequency spectrogram')
    plt.tight_layout()
    plt.savefig(mel_spectrogram_file, dpi=200)
    plt.close()

def inference(ref_audio, ref_text, gen_text, checkpoint, vocab_file, nfe_step):
    model_config = ModelConfig(
        language="th",
        model_type="F5",
        checkpoint=checkpoint,
        vocab_file=vocab_file,
        ode_method="euler",
        use_ema=True,
        vocoder="vocos",
        device=device
    )

    audio_config = AudioConfig(
        silence_threshold=-45,
        max_audio_length=20000,
        cfg_strength=2.5,
        nfe_step=nfe_step,
        target_rms=0.1,
        cross_fade_duration=0.15,
        speed=1.0,
        min_silence_len=500,
        keep_silence=200,
        seek_step=10
    )

    pipeline = FlowTTSPipeline(
        model_config=model_config,
        audio_config=audio_config,
        temp_dir="temp_f5"
    )

    output_file = "outputs_f5/generated.wav"
    mel_spectrogram_file = "outputs_f5/generated_mel.png"

    pipeline(
        text=gen_text,
        ref_voice=ref_audio,
        ref_text=ref_text,
        output_file=output_file,
        speed=1.0,
        check_duration=True
    )
    mel_generator(output_file, mel_spectrogram_file)
    return output_file, mel_spectrogram_file

with gr.Blocks() as demo:
    gr.Image(value=logo_path, show_label=False, container=False, width=400)

    gr.Markdown("## ThonburianTTS Demo 🇹🇭\nGenerate speech from Thai text with reference voice and visualize the Mel spectrogram.")

    with gr.Tab("TTS"):
        checkpoint_input = gr.Textbox(label="Checkpoint Path", value="hf://ThuraAung1601/E2-F5-TTS/F5_Thai/model_last_prune.safetensors")
        vocab_input = gr.Textbox(label="Vocab File", value="hf://ThuraAung1601/E2-F5-TTS/F5_Thai/vocab.txt")
        nfe_input = gr.Slider(label="NFE Value", minimum=4, maximum=64, value=32, step=2)
        ref_audio = gr.Audio(label="Reference Audio", type="filepath", value=default_audio_path)
        ref_text = gr.Textbox(label="Reference Text", value="ใครเป็นผู้รับ")
        gen_text = gr.Textbox(label="Text to Generate")
        output_audio = gr.Audio(label="Generated Audio")
        output_spectrogram = gr.Image(label="Mel-Spectrogram", type="filepath", container=True, width="200")
        generate_btn = gr.Button("Generate Speech")

        generate_btn.click(
            inference,
            inputs=[ref_audio, ref_text, gen_text, checkpoint_input, vocab_input, nfe_input],
            outputs=[output_audio, output_spectrogram]
        )

    with gr.Tab("Multi-Speaker"):
      gr.Markdown("### Multi-Speaker TTS\nProvide speaker references and use `{Speaker1}` or `{Speaker2}` in your script.")

      checkpoint_input = gr.Textbox(label="Checkpoint Path", value="hf://ThuraAung1601/E2-F5-TTS/F5_Thai/model_last_prune.safetensors")
      vocab_input = gr.Textbox(label="Vocab File", value="hf://ThuraAung1601/E2-F5-TTS/F5_Thai/vocab.txt")
      nfe_input = gr.Slider(label="NFE Value", minimum=4, maximum=64, value=32, step=2)

      speaker_labels = [gr.Textbox(label=f"Speaker {i+1} Label", value=f"Speaker{i+1}") for i in range(2)]
      speaker_audios = [gr.Audio(label=f"Speaker {i+1} Reference Audio", type="filepath") for i in range(2)]
      speaker_texts = [gr.Textbox(label=f"Speaker {i+1} Reference Text") for i in range(2)]

      gen_text_multi = gr.Textbox(label="Script (use {Speaker1} and {Speaker2} to indicate speakers)")
      output_audio_multi = gr.Audio(label="Generated Multi-Speaker Audio")
      generate_multi_btn = gr.Button("Generate Multi-Speaker Speech")

      def multi_speaker_inference(
          gen_text,
          speaker1_label, speaker2_label,
          speaker1_audio, speaker2_audio,
          speaker1_text, speaker2_text,
          checkpoint, vocab_file, nfe_step
      ):
          import json
          import numpy as np
          import librosa
          import soundfile as sf

          speakers = {
              speaker1_label.strip(): (speaker1_audio, speaker1_text),
              speaker2_label.strip(): (speaker2_audio, speaker2_text)
          }

          audio_segments = []
          sr = 22050  # fallback sampling rate

          lines = gen_text.strip().splitlines()

          for line in lines:
              # Split JSON part and text part
              try:
                  # Find the first closing brace "}" to separate JSON metadata from text
                  json_end_idx = line.index('}') + 1
                  json_str = line[:json_end_idx]
                  text = line[json_end_idx:].strip()

                  speaker_meta = json.loads(json_str)
                  speaker_name = speaker_meta.get("name", "").strip()

                  if speaker_name not in speakers:
                      print(f"⚠️ Unknown speaker: {speaker_name}, skipping line.")
                      continue

                  ref_audio, ref_text = speakers[speaker_name]

                  if not ref_audio:
                      print(f"⚠️ Missing reference audio for {speaker_name}, skipping line.")
                      continue

                  if not text:
                      print("⚠️ Empty text after speaker metadata, skipping line.")
                      continue

                  # Call your TTS inference
                  audio_file, _ = inference(ref_audio, ref_text, text, checkpoint, vocab_file, nfe_step)

                  # Load generated audio and append
                  y, sr = librosa.load(audio_file, sr=None)
                  audio_segments.append(y)

              except (ValueError, json.JSONDecodeError) as e:
                  print(f"⚠️ Error parsing line: {line}\nError: {e}")
                  continue

          if audio_segments:
              final_audio = np.concatenate(audio_segments)
              temp_out = "outputs_f5/generated_multi.wav"
              sf.write(temp_out, final_audio, sr)
              return temp_out
          else:
              return None

      generate_multi_btn.click(
          fn=multi_speaker_inference,
          inputs=[
              gen_text_multi,
              speaker_labels[0], speaker_labels[1],
              speaker_audios[0], speaker_audios[1],
              speaker_texts[0], speaker_texts[1],
              checkpoint_input, vocab_input, nfe_input
          ],
          outputs=[output_audio_multi]
      )

demo.launch(share=True)