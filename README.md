<p align="center">
  <img src="assets/ThonburianTTSLogo.png" width="400"/>
</p>

<p align="center">
  <img src="assets/looloo-logo.png" width="200" />
</p>

[🔊 Model Checkpoint](https://huggingface.co/ThuraAung1601/E2-F5-TTS) | [🤗 Gradio App Demo](https://github.com/biodatlab/thonburian-tts/blob/main/gradio_app.py) | [📄 Related Whisper Paper]()

---

## **Thonburian F5-TTS**
**Thonburian F5-TTS** is a **Thai Text-to-Speech (TTS)** engine built on top of the [F5-TTS](https://github.com/SWivid/F5-TTS).  
It generates **natural and expressive Thai speech** by leveraging **Flow-Matching diffusion techniques** and can **mimic reference voices** from short audio samples.

The system supports:
- **Thai language generation** (`language="th"`)
- **Reference-based voice cloning** using short audio clips
- High-quality synthesis with controllable speed and silence trimming
- Hugging Face integration for **easy deployment and hosting**

---

## **Quick Usage**

Below is a minimal example for generating **Thai speech** with **voice cloning** using a reference sample.

```python
from flowtts.inference import FlowTTSPipeline, ModelConfig, AudioConfig
import torch

# Configure F5-TTS model
model_config = ModelConfig(
    language="th",
    model_type="F5",
    checkpoint="hf://ThuraAung1601/E2-F5-TTS/F5_Thai/mega_f5_last.safetensors",
    vocab_file="hf://ThuraAung1601/E2-F5-TTS/F5_Thai/mega_vocab.txt",
    vocoder="vocos",
    device="cuda" if torch.cuda.is_available() else "cpu"
)

# Basic audio settings
audio_config = AudioConfig(
    silence_threshold=-45,
    cfg_strength=2.5,
    speed=1.0
)

pipeline = FlowTTSPipeline(model_config, audio_config)

# Input text and reference voice
text = "ยินดีที่ได้รู้จักคุณวันนี้อากาศดีมาก"
ref_voice = "ref_samples/ref_sample.wav"
ref_text = "ยินดีที่ได้รู้จัก"  # Manual transcript of the reference clip

# Generate speech
output_path = pipeline(
    text=text,
    ref_voice=ref_voice,
    ref_text=ref_text,
    output_file="f5_output.wav"
)
print(f"Generated F5 audio saved to: {output_path}")
```

---

## **Installation**

Install dependencies:

```bash
pip install torch cached-path librosa transformers f5-tts
sudo apt install ffmpeg
```

---

## **Model Checkpoints**

| Model Component        | Description                        | URL                                                                          |
| ---------------------- | ---------------------------------- | ---------------------------------------------------------------------------- |
| **F5-TTS Thai**        | Flow Matching-based Thai TTS models | [Link](https://huggingface.co/ThuraAung1601/E2-F5-TTS/F5_Thai)               |
| **F5-TTS IPA**         | Flow Matching-based Thai-IPA TTS models | [Link](https://huggingface.co/ThuraAung1601/E2-F5-TTS/F5_IPA)            |

## **Pipeline Overview**

The end-to-end **Thai voice cloning workflow**:

```
  Reference Voice (Thai Speech)
            ↓
Optional: Thonburian Whisper (ASR)
            ↓
 Transcribed Reference Text
            ↓
         F5-TTS
            ↓
  Generated Thai Speech
```

This workflow enables:
- **High-quality Thai speech generation** from text
- Voice cloning with **style and tone preservation**
- Full ASR-TTS integration for interactive voice applications

---

## **Example Outputs**

<table>
  <tr>
    <td align="center">
      <a href="https://youtu.be/rvmNgh0-jws">
        <img src="https://img.youtube.com/vi/rvmNgh0-jws/0.jpg" width="320"><br>
        🎵 Sample 1 – Single-speaker Thai Normal Text
      </a>
    </td>
    <td align="center">
      <a href="https://youtu.be/jVz3EpRTn1U">
        <img src="https://img.youtube.com/vi/jVz3EpRTn1U/0.jpg" width="320"><br>
        🎵 Sample 2 – Single-Speaker Thai Code-mixed Text
      </a>
    </td>
    <td align="center">
      <a href="https://youtu.be/sbaOdMhz3Z4">
        <img src="https://img.youtube.com/vi/sbaOdMhz3Z4/0.jpg" width="320"><br>
        🎵 Sample 3 – Multi-Speaker Conversational Speech
      </a>
    </td>
  </tr>
</table>

---

## **Developers**

* [Biomedical and Data Lab, Mahidol University](https://biodatlab.github.io/)
* [Looloo Technology](https://loolootech.com/)

<p align="center">
  <img width="150px" src="assets/looloo-logo.png" />
</p>

---

## **Citation**

If you use **ThonburianTTS** in your research, please cite:

```

```

Or cite as a written reference:

> 

---

## **License**

- **Codes:** Released under the [MIT License](LICENSE-MIT).
- **Models:** Released under the [Creative Commons Attribution Non-Commercial ShareAlike 4.0 License (CC BY-NC-SA 4.0)](LICENSE-CC-BY-NC-SA).

---
## Reference
- [F5-TTS Original Repository](https://github.com/SWivid/F5-TTS)
