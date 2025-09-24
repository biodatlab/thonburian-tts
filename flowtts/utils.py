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
