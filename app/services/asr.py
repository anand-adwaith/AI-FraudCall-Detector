import os
import logging
import torch
import torchaudio
import soundfile as sf
from transformers import AutoModel
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("asr-service")

"""
load the indic-conformer-600m-multilingual model and perform ASR
input: 
  audio_file (str) :path to the audio file, 
  lang_id (str) :language ID for ASR. Currently supported languages and their IDs are:
      Assamese: "as"      Bengali: "bn"       Bodo: "brx"
      Dogri: "doi"        Gujarati: "gu"      Hindi: "hi"
      Kannada: "kn"       Konkani: "kok"      Kashmiri: "ks"
      Maithili: "mai"     Malayalam: "ml"     Manipuri: "mni"
      Marathi: "mr"       Nepali: "ne"        Odia: "or"
      Punjabi: "pa"       Sanskrit: "sa"      Santali: "sat"
      Sindhi: "sd"        Tamil: "ta"         Telugu: "te"
      Urdu: "ur"
output: transcription of the audio file in the specified language
"""

# Global model cache
_ASR_MODEL = None

def initialize_asr_model():
    """
    Initialize the ASR model and cache it globally.
    Returns the initialized model.
    """
    global _ASR_MODEL
    
    if _ASR_MODEL is not None:
        logger.info("Using cached ASR model")
        return _ASR_MODEL
    
    logger.info("Initializing ASR model...")
    try:
        # Check if GPU is available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")
        
        # Load the model
        _ASR_MODEL = AutoModel.from_pretrained(
            "ai4bharat/indic-conformer-600m-multilingual", 
            trust_remote_code=True
        )
        
        # Move to proper device
        _ASR_MODEL = _ASR_MODEL.to(device)
        
        # Optimize if GPU
        if device == "cuda":
            _ASR_MODEL.half()  # Use half precision for GPU
            
        # Set to evaluation mode
        _ASR_MODEL.eval()
        
        logger.info("ASR model initialization successful")
        return _ASR_MODEL
    except Exception as e:
        logger.error(f"ASR model initialization failed: {str(e)}")
        raise

def load_audio_file(file_path: str) -> Tuple[torch.Tensor, int]:
    """
    Load audio file using multiple backends for enhanced reliability.
    
    Args:
        file_path: Path to the audio file
        
    Returns:
        Tuple of (waveform tensor, sample_rate)
        
    Raises:
        FileNotFoundError: If the audio file doesn't exist
        RuntimeError: If the audio file can't be loaded
    """
    logger.info(f"Loading audio file: {file_path}")
    
    if not os.path.exists(file_path):
        logger.error(f"Audio file not found: {file_path}")
        raise FileNotFoundError(f"Audio file not found: {file_path}")
    
    # Try multiple loading methods
    try:
        # Method 1: Try torchaudio
        try:
            waveform, sample_rate = torchaudio.load(file_path)
            logger.info(f"Successfully loaded audio with torchaudio, shape: {waveform.shape}, sr: {sample_rate}")
            return waveform, sample_rate
        except Exception as e:
            logger.warning(f"torchaudio failed to load {file_path}: {str(e)}")
        
        # Method 2: Try soundfile + torch conversion
        try:
            data, sample_rate = sf.read(file_path)
            # Convert to torch tensor
            if len(data.shape) == 1:  # Mono
                waveform = torch.FloatTensor(data).unsqueeze(0)
            else:  # Multi-channel
                waveform = torch.FloatTensor(data.T)
            logger.info(f"Successfully loaded audio with soundfile, shape: {waveform.shape}, sr: {sample_rate}")
            return waveform, sample_rate
        except Exception as e:
            logger.warning(f"soundfile failed to load {file_path}: {str(e)}")
        
        # If all methods fail, raise error
        raise RuntimeError(f"All loading methods failed for {file_path}")
        
    except Exception as e:
        logger.error(f"Error loading audio file: {str(e)}")
        raise

async def asr_transcribe(model, audio_file: str, lang_id: str) -> str:
    """
    Transcribe an audio file using the ASR model.
    
    Args:
        model: The ASR model
        audio_file: Path to the audio file
        lang_id: Language ID for transcription
        
    Returns:
        Transcribed text
    """
    try:
        logger.info(f"Transcribing audio file: {audio_file} with language: {lang_id}")
        
        # Load audio file with enhanced error handling
        wav, sr = load_audio_file(audio_file)
        
        # Convert to mono if stereo
        if wav.shape[0] > 1:
            wav = torch.mean(wav, dim=0, keepdim=True)
        
        # Resample if needed
        target_sample_rate = 16000  # Expected sample rate
        if sr != target_sample_rate:
            logger.info(f"Resampling from {sr}Hz to {target_sample_rate}Hz")
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sample_rate)
            wav = resampler(wav)
        
        # Perform ASR with RNNT decoding
        transcription = model(wav, lang_id, "rnnt")
        logger.info(f"Transcription complete: {transcription}")
        
        return transcription
    except Exception as e:
        logger.error(f"Transcription failed: {str(e)}")
        raise

if __name__ == "__main__":
    # Example usage
    model = initialize_asr_model()
    audio_file = r"Dataset/sample_audio_infer_ready.wav"  # Replace with your audio file path
    lang_id = "hi"  # Replace with desired language ID
    transcription = asr_transcribe(model, audio_file, lang_id)
    print("Transcription:", transcription)