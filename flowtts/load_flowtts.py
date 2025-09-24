import random
import sys
from importlib.resources import files

import soundfile as sf
import tqdm
from cached_path import cached_path
from hydra.utils import get_class
from omegaconf import OmegaConf

# Change from relative import to absolute import
from flowtts.infer.utils_infer import (
    load_model,
    load_vocoder,
    transcribe,
    preprocess_ref_audio_text,
    infer_process,
    remove_silence_for_generated_wav,
    save_spectrogram,
    transcribe,
    target_sample_rate,
    hop_length
)
from f5_tts.model.utils import seed_everything
from f5_tts.model import DiT, UNetT

from typing import Optional, Tuple, Union, Any
import torch
import numpy as np
from numpy.typing import NDArray

class FlowTTS:
    """
    FlowTTS model for text-to-speech synthesis and voice conversion.
    
    Supports both English and Thai languages with F5 and E2 model architectures.
    Uses either vocos or bigvgan as the vocoder.
    
    Attributes:
        model_type (str): Type of model architecture ('F5' or 'E2')
        checkpoint (str): Path to model checkpoint
        vocab_file (str): Path to vocabulary file
        ode_method (str): ODE solver method
        use_ema (str): Whether to use EMA weights
        vocoder_name (str): Name of vocoder ('vocos' or 'bigvgan')
        language (str): Language code ('en' or 'th')
        device (str): Device to run model on ('cuda', 'mps', or 'cpu')
    """
    
    def __init__(
            self,
            model_type: str = "F5",
            checkpoint: str = "",
            vocab_file: str = "",
            ode_method: str = "euler",
            use_ema: str = "True",
            vocoder_name: str = "vocos",
            local_path: Optional[str] = None,
            device: Optional[str] = None,
            hf_cache_dir: Optional[str] = None,
            language: str = "en"
    ) -> None:
        # Set model type and language first
        self.model_type = model_type
        self.language = language
        
        # Initialize device
        if device is not None and device in ['cuda', 'mps', 'cpu']:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

        # Set mel spec type before loading models
        self.mel_spec_type = vocoder_name
        
        # Initialize other parameters
        self.checkpoint = checkpoint
        self.vocab_file = vocab_file
        self.vocoder_name = vocoder_name
        self.hf_cache_dir = hf_cache_dir
        self.ode_method = ode_method
        self.use_ema = use_ema
        self.local_path = local_path
        
        # Initialize inference parameters
        self.final_wave = None
        self.target_sample_rate = target_sample_rate 
        self.hop_length = hop_length
        self.seed = -1

        # Load models after all parameters are set
        self.load_vocoder_model(
            self.vocoder_name, local_path=self.local_path, hf_cache_dir=self.hf_cache_dir
        )
        self.load_ema_model(
            self.model_type, self.checkpoint, self.vocoder_name, self.vocab_file, 
            self.ode_method, self.use_ema, hf_cache_dir=self.hf_cache_dir
        )

    def load_vocoder_model(self, vocoder_name: str, local_path: Optional[str] = None, 
                          hf_cache_dir: Optional[str] = None) -> None:
        """
        Load vocoder model for waveform generation.
        
        Args:
            vocoder_name: Name of vocoder to use
            local_path: Optional path to local vocoder model
            hf_cache_dir: Optional huggingface cache directory
        """
        self.vocoder = load_vocoder(vocoder_name, local_path is not None, local_path, 
                                  self.device, hf_cache_dir)

    def load_ema_model(self, model_type: str, checkpoint: str, vocoder_name: str, 
                      vocab_file: str, ode_method: str, use_ema: str, 
                      hf_cache_dir: Optional[str] = None) -> None:
        """
        Load TTS model with specified configuration.
        
        Args:
            model_type: Type of model architecture
            checkpoint: Path to model checkpoint
            vocoder_name: Name of vocoder
            vocab_file: Path to vocabulary file
            ode_method: ODE solver method
            use_ema: Whether to use EMA weights
            hf_cache_dir: Optional huggingface cache directory
        
        Raises:
            ValueError: If language or model type not supported
        """
        hf_cache_dir = hf_cache_dir or self.hf_cache_dir
    
        if model_type == "F5":
            if not checkpoint:
                if self.mel_spec_type == "vocos":
                    if self.language == "en":
                        checkpoint = str(
                            cached_path("hf://SWivid/F5-TTS/F5TTS_Base/model_1200000.safetensors", cache_dir=hf_cache_dir)
                        )
                        if not vocab_file:
                            vocab_file = str(
                                cached_path("hf://SWivid/F5-TTS/F5TTS_Base/vocab.txt", cache_dir=hf_cache_dir)
                            )
                    elif self.language == "th":
                        checkpoint = str(
                            cached_path("hf://VIZINTZOR/F5-TTS-THAI/model_600000_FP16.pt", cache_dir=hf_cache_dir)
                        )
                        if not vocab_file:
                            vocab_file = str(
                                cached_path("hf://VIZINTZOR/F5-TTS-THAI/vocab.txt", cache_dir=hf_cache_dir)
                            )
                    else:
                        raise ValueError(f"Language {self.language} not supported! Currently supported languages: en, th")
                elif self.mel_spec_type == "bigvgan":
                    if self.language == "en":
                        checkpoint = str(
                            cached_path("hf://SWivid/F5-TTS/F5TTS_Base_bigvgan/model_1250000.safetensors", cache_dir=hf_cache_dir)
                        )
                        if not vocab_file:
                            vocab_file = str(
                                cached_path("hf://SWivid/F5-TTS/F5TTS_Base_bigvgan/vocab.txt", cache_dir=hf_cache_dir)
                            )
                    else:
                        raise ValueError(f"Language {self.language} not supported for BigVGAN!")
            model_cfg = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
            model_cls = DiT

        elif model_type == "E2":
            if not checkpoint:
                if self.language == "en":
                    checkpoint = str(
                        cached_path("hf://SWivid/E2-TTS/E2TTS_Base/model_1200000.safetensors", cache_dir=hf_cache_dir)
                    )
                    if not vocab_file:
                        vocab_file = str(
                            cached_path("hf://SWivid/E2-TTS/E2TTS_Base/vocab.txt", cache_dir=hf_cache_dir)
                        )
                else:
                    raise ValueError(f"Language {self.language} not supported for E2!")
            model_cfg = dict(dim=1024, depth=24, heads=16, ff_mult=4, text_mask_padding=False, pe_attn_head=1)
            model_cls = UNetT

        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        self.ema_model = load_model(
            model_cls=model_cls,
            model_cfg=model_cfg,
            ckpt_path=checkpoint,
            mel_spec_type=self.mel_spec_type,
            vocab_file=vocab_file,
            ode_method=ode_method,
            use_ema=use_ema,
            device=self.device
        )

    def transcribe(self, ref_audio: str, language: Optional[str] = None) -> str:
        """
        Transcribe audio file to text.
        
        Args:
            ref_audio: Path to audio file
            language: Optional language code
        
        Returns:
            Transcribed text
        """
        return transcribe(ref_audio, language)

    def export_wav(self, wav: NDArray[np.float32], file_wave: str, 
                  remove_silence: bool = False) -> None:
        """
        Export waveform to WAV file.
        
        Args:
            wav: Audio waveform array
            file_wave: Output file path
            remove_silence: Whether to remove silence from edges
        """
        sf.write(file_wave, wav, self.target_sample_rate)
        if remove_silence:
            remove_silence_for_generated_wav(file_wave)

    def export_spectrogram(self, spect: torch.Tensor, file_spect: str) -> None:
        """
        Export mel spectrogram to image file.
        
        Args:
            spect: Mel spectrogram tensor
            file_spect: Output file path
        """
        save_spectrogram(spect, file_spect)

    def infer(
        self,
        ref_file: str,
        ref_text: str,
        gen_text: str,
        show_info: Any = print,
        progress: Any = tqdm,
        target_rms: float = 0.1,
        cross_fade_duration: float = 0.15,
        sway_sampling_coef: float = -1,
        cfg_strength: float = 2,
        nfe_step: int = 32,
        speed: float = 1.0,
        fix_duration: Optional[float] = None,
        remove_silence: bool = False,
        file_wave: Optional[str] = None,
        file_spect: Optional[str] = None,
        seed: int = -1,
    ) -> Tuple[NDArray[np.float32], int, torch.Tensor]:
        """
        Generate speech from text using reference audio.
        
        Args:
            ref_file: Path to reference audio file
            ref_text: Text content of reference audio
            gen_text: Text to synthesize
            show_info: Function for displaying info
            progress: Progress bar class
            target_rms: Target RMS value for audio
            cross_fade_duration: Duration of cross fade
            sway_sampling_coef: Coefficient for sway sampling
            cfg_strength: Classifier-free guidance strength
            nfe_step: Number of function evaluations
            speed: Speech speed multiplier
            fix_duration: Optional fixed duration
            remove_silence: Whether to remove silence
            file_wave: Optional path to save WAV
            file_spect: Optional path to save spectrogram
            seed: Random seed (-1 for random)
        
        Returns:
            Tuple of (waveform, sample rate, spectrogram)
        """
        # Set random seed
        if seed == -1:
            seed = random.randrange(4294967295)  # Maximum value allowed for PYTHONHASHSEED
        seed_everything(seed)
        self.seed = seed

        # Preprocess reference audio and text
        ref_file, ref_text = preprocess_ref_audio_text(ref_file, ref_text)

        # Generate speech
        wav, sr, spect = infer_process(
            ref_file,
            ref_text,
            gen_text,
            self.ema_model,
            self.vocoder,
            self.mel_spec_type,
            show_info=show_info,
            progress=progress,
            target_rms=target_rms,
            cross_fade_duration=cross_fade_duration,
            nfe_step=nfe_step,
            cfg_strength=cfg_strength,
            sway_sampling_coef=sway_sampling_coef,
            speed=speed,
            fix_duration=fix_duration,
            device=self.device,
        )

        # Export results if paths provided
        if file_wave is not None:
            self.export_wav(wav, file_wave, remove_silence)
        if file_spect is not None:
            self.export_spectrogram(spect, file_spect)

        return wav, sr, spect