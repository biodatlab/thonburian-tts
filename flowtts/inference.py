import torch
from pydub import silence
import os
from cached_path import cached_path
from pydub import AudioSegment
import logging
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
from flowtts.load_flowtts import FlowTTS

import re
from typing import List
from pydub import AudioSegment

def convert_to_wav(input_path: str, output_path: str) -> None:
    """
    Convert any audio format to WAV using pydub.
    
    Args:
        input_path: Path to input audio file
        output_path: Path to save converted WAV file
    """
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_channels(1)  # Convert to mono
    audio = audio.set_frame_rate(24000)  # Set to F5-TTS expected sample rate
    audio.export(output_path, format='wav')

def split_text_into_sentences(text: str, language: str = "en") -> List[str]:
    """
    Split text into sentences based on language.
    
    Args:
        text: Input text to split
        language: Language code ('en' for English, 'th' for Thai)
    
    Returns:
        List of sentences
        
    Raises:
        ValueError: If language is not supported
    """
    if language == "en":
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
    elif language == "th":
        from pythainlp.tokenize import sent_tokenize
        sentences = sent_tokenize(text)
    else:
        raise ValueError(f"Language '{language}' not supported. Currently supported languages: en, th")
    return sentences

def detect_leading_silence(audio: AudioSegment, 
                         silence_threshold: float = -42, 
                         chunk_size: int = 10) -> int:
    """
    Detect silence at the beginning of the audio.
    
    Args:
        audio: Audio segment to process
        silence_threshold: Threshold in dB to consider as silence
        chunk_size: Size of chunks to process
    
    Returns:
        Position in milliseconds where silence ends
    """
    trim_ms = 0
    while audio[trim_ms:trim_ms + chunk_size].dBFS < silence_threshold and trim_ms < len(audio):
        trim_ms += chunk_size
    return trim_ms

def remove_silence_edges(audio: AudioSegment, 
                        silence_threshold: float = -42) -> AudioSegment:
    """
    Remove silence from the beginning and end of the audio.
    
    Args:
        audio: Audio segment to process
        silence_threshold: Threshold in dB to consider as silence
    
    Returns:
        Audio segment with silence removed from edges
    """
    start_trim = detect_leading_silence(audio, silence_threshold)
    end_trim = detect_leading_silence(audio.reverse(), silence_threshold)
    duration = len(audio)
    return audio[start_trim:duration - end_trim]

# Configuration class for model settings
@dataclass
class ModelConfig:
    """Configuration settings for the TTS model."""
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"
    model_type: str = "F5"
    language: str = "th"
    vocoder: str = "vocos"
    ode_method: str = "euler"
    use_ema: bool = True
    vocab_file: str = ""
    checkpoint: str = ""
    seed: int = -1  # Added seed parameter
    
    def __post_init__(self):
        if not self.checkpoint and not self.vocab_file:
            if self.model_type == "E2":
                self.checkpoint = "hf://SWivid/E2-TTS/E2TTS_Base/model_1200000.safetensors"
                self.vocab_file = "hf://SWivid/E2-TTS/E2TTS_Base/vocab.txt"
                self.vocoder = "vocos"  # E2 works better with vocos
            elif self.model_type == "F5":
                if self.language == "th":
                    self.checkpoint = "hf://VIZINTZOR/F5-TTS-THAI/model_600000_FP16.pt"
                    self.vocab_file = "hf://VIZINTZOR/F5-TTS-THAI/vocab.txt"
                else:
                    self.checkpoint = "hf://SWivid/F5-TTS/F5TTS_Base/model_1200000.safetensors"
                    self.vocab_file = "hf://SWivid/F5-TTS/F5TTS_Base/vocab.txt"

@dataclass
class AudioConfig:
    """Audio processing configuration"""
    silence_threshold: int = -45
    min_audio_length: int = 1000
    max_audio_length: int = 20000
    cfg_strength: float = 2.0  # Changed from 2.5 to 2.0 to match Test Model
    nfe_step: int = 32
    target_rms: float = 0.1
    cross_fade_duration: float = 0.15
    speed: float = 1.0  # Added speed parameter
    use_truth_duration: bool = False  # Added use_truth_duration parameter
    no_ref_audio: bool = False  # Added no_ref_audio parameter
    silence_padding: int = 200  # Added silence padding parameter
    min_silence_len: int = 500  # Added min silence length parameter
    keep_silence: int = 200  # Added keep silence parameter
    seek_step: int = 10  # Added seek step parameter
    
    def __post_init__(self):
        if self.max_audio_length < self.min_audio_length:
            raise ValueError("max_audio_length must be greater than min_audio_length")

class FlowTTSPipeline:
    """
    Pipeline for text-to-speech synthesis and voice conversion using FlowTTS.
    
    Handles audio preprocessing, model inference, and post-processing.
    """
    
    def __init__(
        self,
        model: Optional[FlowTTS] = None,
        model_config: Optional[ModelConfig] = None,
        audio_config: Optional[AudioConfig] = None,
        temp_dir: str = "temp"
    ):
        """
        Initialize the TTS pipeline.
        
        Args:
            model: Optional pre-initialized FlowTTS model
            model_config: Model configuration settings
            audio_config: Audio processing configuration
            temp_dir: Directory for temporary files
        """
        self.model_config = model_config or ModelConfig()
        self.audio_config = audio_config or AudioConfig()
        
        if model:
            self.model = model
        else:
            # print(f"\nInitializing {self.model_config.checkpoint} model for {self.model_config.language} language...")
            self.model = FlowTTS(
                device=self.model_config.device,
                model_type=self.model_config.model_type,
                language=self.model_config.language,
                vocoder_name=self.model_config.vocoder,
                vocab_file=str(cached_path(self.model_config.vocab_file)),
                ode_method=self.model_config.ode_method,
                use_ema=self.model_config.use_ema,
                checkpoint=str(cached_path(self.model_config.checkpoint))
            )
        
        # Change output_dir to temp_dir for temporary files
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    def __call__(self, text: str, ref_voice: str, output_file: str, ref_text: Optional[str] = None, 
                 speed: float = 1.0, check_duration: bool = False) -> str:
        """
        Generate speech from text using a reference voice.
        
        Args:
            text: Text to synthesize
            ref_voice: Path to reference voice audio file
            output_file: Path to save generated audio
            ref_text: Optional text content of reference audio
            speed: Speech speed multiplier
            check_duration: Whether to print generation duration
            
        Returns:
            Path to generated audio file
        """
        start_time = time.time()
        
        # Convert reference audio to WAV if needed
        reference_file = ref_voice
        if not ref_voice.endswith('.wav'):
            ref_wav_path = self.temp_dir / "ref_converted.wav"
            convert_to_wav(ref_voice, str(ref_wav_path))
            reference_file = str(ref_wav_path)

        # Process reference audio
        temp_short_ref = self.temp_dir / "temp_short_ref.wav"
        aseg = AudioSegment.from_file(reference_file)

        # Process with configurable silence detection
        non_silent_wave = self._process_audio_silence(aseg)
        
        # Export processed audio
        aseg = remove_silence_edges(non_silent_wave) + AudioSegment.silent(duration=self.audio_config.silence_padding)
        aseg.export(str(temp_short_ref), format='wav')
        
        # Use provided reference text or transcribe automatically
        if ref_text is None:
            ref_text = self.model.transcribe(str(temp_short_ref))
            logging.info(f'Reference text transcribed: {ref_text}')
        
        # Use the provided output_file path
        save_path = Path(output_file)
        # Create output directory if it doesn't exist
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Update infer call with all necessary parameters from infer_cli.py
        wav, sr, _ = self.model.infer(
            ref_file=str(temp_short_ref),
            ref_text=ref_text,
            gen_text=text,
            target_rms=self.audio_config.target_rms,
            cross_fade_duration=self.audio_config.cross_fade_duration,
            nfe_step=self.audio_config.nfe_step,
            cfg_strength=self.audio_config.cfg_strength,
            sway_sampling_coef=0.0,  # Add this from infer_cli.py
            speed=speed,
            fix_duration=None,  # Add this from infer_cli.py
            file_wave=str(save_path),
            seed=self.model_config.seed  # Add seed parameter
        )

        if check_duration:
            elapsed_time = time.time() - start_time
            print(f"Time taken is {elapsed_time:.2f} seconds")
        
        return str(save_path)

    def _process_audio_silence(self, audio_segment: AudioSegment) -> AudioSegment:
        """
        Process audio by removing silence with adaptive thresholds.
        
        Uses two-pass silence detection with different thresholds to handle
        varying audio conditions. Falls back to force clipping if needed.
        
        Args:
            audio_segment: Input audio segment
            
        Returns:
            Processed audio segment with silence removed
        """
        for threshold, min_silence_len in [
            (self.audio_config.silence_threshold, self.audio_config.min_silence_len),
            (self.audio_config.silence_threshold + 10, self.audio_config.min_silence_len // 10)
        ]:
            non_silent_segs = silence.split_on_silence(
                audio_segment,
                min_silence_len=min_silence_len,
                silence_thresh=threshold,
                keep_silence=self.audio_config.keep_silence,
                seek_step=self.audio_config.seek_step
            )
            
            non_silent_wave = AudioSegment.silent(duration=0)
            for seg in non_silent_segs:
                if (len(non_silent_wave) > self.audio_config.min_audio_length and 
                    len(non_silent_wave + seg) > self.audio_config.max_audio_length):
                    logging.info(f"Audio exceeds {self.audio_config.max_audio_length}ms, clipping")
                    break
                non_silent_wave += seg
                
            if len(non_silent_wave) <= self.audio_config.max_audio_length:
                return non_silent_wave
        
        # If no proper silence found, force clip
        return audio_segment[:self.audio_config.max_audio_length]

    def voice_conversion(self, reference_file: str, file_to_convert, output_file: str) -> str:
        """
        Convert voice characteristics of input audio to match reference voice.
        
        Args:
            reference_file: Path to reference voice audio file
            file_to_convert: File-like object containing audio to convert
            output_file: Path to save converted audio
            
        Returns:
            Path to converted audio file
            
        Notes:
            Maintains the content of the input audio while adopting the
            voice characteristics of the reference audio.
        """
        input_path = self.temp_dir / "input_audio.wav"
        with open(input_path, 'wb') as f:
            f.write(file_to_convert.read())
        
        try:
            if not reference_file.endswith('.wav'):
                ref_wav_path = self.temp_dir / "ref_converted.wav"
                convert_to_wav(reference_file, str(ref_wav_path))
                reference_file = str(ref_wav_path)
            
            text = self.model.transcribe(str(input_path))
            save_path = Path(output_file)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            self.model.infer(
                ref_file=reference_file,
                ref_text=text,
                gen_text=text,
                file_wave=str(save_path)
            )
            
            return str(save_path)
        finally:
            if input_path.exists():
                input_path.unlink()
