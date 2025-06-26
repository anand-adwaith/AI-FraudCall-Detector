import os
import logging
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from IndicTransToolkit.processor import IndicProcessor
from pathlib import Path
from typing import Dict, Tuple, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("translate-service")

# Global configuration
BATCH_SIZE = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_MODEL_NAME = "ai4bharat/indictrans2-indic-en-dist-200M"

# Global model cache
_TRANSLATE_MODEL = None
_TRANSLATE_TOKENIZER = None
_INDIC_PROCESSOR = None

def initialize_translate_tokenizer_and_model(ckpt_dir=DEFAULT_MODEL_NAME) -> Tuple[AutoTokenizer, AutoModelForSeq2SeqLM]:
    """
    Initialize and cache translation model and tokenizer.
    
    Args:
        ckpt_dir: Model checkpoint directory or HuggingFace model name
        
    Returns:
        Tuple of (tokenizer, model)
    """
    global _TRANSLATE_MODEL, _TRANSLATE_TOKENIZER
    
    if _TRANSLATE_TOKENIZER is not None and _TRANSLATE_MODEL is not None:
        logger.info("Using cached translation model and tokenizer")
        return _TRANSLATE_TOKENIZER, _TRANSLATE_MODEL
    
    logger.info(f"Initializing translation model and tokenizer from {ckpt_dir}")
    
    try:
        # Initialize tokenizer
        _TRANSLATE_TOKENIZER = AutoTokenizer.from_pretrained(ckpt_dir, trust_remote_code=True)
        
        # Initialize model
        _TRANSLATE_MODEL = AutoModelForSeq2SeqLM.from_pretrained(
            ckpt_dir,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            quantization_config=None,
        )

        # Move model to device and optimize
        _TRANSLATE_MODEL = _TRANSLATE_MODEL.to(DEVICE)
        if DEVICE == "cuda":
            _TRANSLATE_MODEL.half()

        _TRANSLATE_MODEL.eval()
        
        logger.info("Translation model and tokenizer initialization successful")
        return _TRANSLATE_TOKENIZER, _TRANSLATE_MODEL
        
    except Exception as e:
        logger.error(f"Translation model initialization failed: {str(e)}")
        raise

def get_indic_processor():
    """
    Get or initialize the IndicProcessor.
    
    Returns:
        IndicProcessor instance
    """
    global _INDIC_PROCESSOR
    
    if _INDIC_PROCESSOR is not None:
        return _INDIC_PROCESSOR
    
    try:
        logger.info("Initializing IndicProcessor")
        _INDIC_PROCESSOR = IndicProcessor(inference=True)
        return _INDIC_PROCESSOR
    except Exception as e:
        logger.error(f"IndicProcessor initialization failed: {str(e)}")
        raise

# Language code mapping
LANG_CODE_MAP = {
    "as": "asm_Beng",
    "bn": "ben_Beng",
    "brx": "brx_Deva",
    "doi": "doi_Deva",
    "gu": "guj_Gujr",
    "hi": "hin_Deva",
    "kn": "kan_Knda",
    "kok": "gom_Deva",
    "ks": "kas_Deva",
    "mai": "mai_Deva",
    "ml": "mal_Mlym",
    "mni": "mni_Mtei",
    "mr": "mar_Deva",
    "ne": "npi_Deva",
    "or": "ory_Orya",
    "pa": "pan_Guru",
    "sa": "san_Deva",
    "sat": "sat_Olck",
    "sd": "snd_Deva",
    "ta": "tam_Taml",
    "te": "tel_Telu",
    "ur": "urd_Arab",
    "en": "eng_Latn"
}

"""
Translate Indic languages to English using the IndicTrans2 model.
This function preprocesses the input text, tokenizes it, generates 
the translation, and postprocesses the output.

Inputs:
    input_text (str): The text to be translated from an Indic language to English.
    src_lang_id (str): The source language code of the input text (e.g., "hi" for Hindi).
        Assamese: "as"      Bengali: "bn"       Bodo: "brx"
        Dogri: "doi"        Gujarati: "gu"      Hindi: "hi"
        Kannada: "kn"       Konkani: "kok"      Kashmiri: "ks"
        Maithili: "mai"     Malayalam: "ml"     Manipuri: "mni"
        Marathi: "mr"       Nepali: "ne"        Odia: "or"
        Punjabi: "pa"       Sanskrit: "sa"      Santali: "sat"
        Sindhi: "sd"        Tamil: "ta"         Telugu: "te"
        Urdu: "ur"
    tokenizer: The translation tokenizer
    model: The translation model
    
Outputs:
    translation (str): The translated text in English.
"""
async def translate_indic_to_english(input_text: str, src_lang_id: str, tokenizer=None, model=None):
    try:
        logger.info(f"Translating text from {src_lang_id} to English: {input_text[:50]}...")
        
        # Get tokenizer and model if not provided
        if tokenizer is None or model is None:
            tokenizer, model = initialize_translate_tokenizer_and_model()
        
        # Get IndicProcessor
        ip = get_indic_processor()
        
        # Map the source language ID to internal format
        src_lang_id_internal = LANG_CODE_MAP.get(src_lang_id)
        if not src_lang_id_internal:
            raise ValueError(f"Unsupported source language: {src_lang_id}")
        
        # Preprocess the input text
        input_batch = ip.preprocess_batch([input_text], src_lang=src_lang_id_internal, tgt_lang="eng_Latn")
        
        # Tokenize the input text
        inputs = tokenizer(
            input_batch,
            truncation=True,
            padding="longest",
            return_tensors="pt",
            return_attention_mask=True,
        ).to(DEVICE)
        
        # Generate the translation
        with torch.no_grad():
            generated_tokens = model.generate(
                **inputs,
                use_cache=True,
                min_length=0,
                max_length=256,
                num_beams=5,
                num_return_sequences=1,
            )        # Decode the generated tokens into text
        try:
            # Handle the generated tokens safely
            if isinstance(generated_tokens, list):
                tokens_to_decode = generated_tokens[0]
            else:
                tokens_to_decode = generated_tokens[0]
                
            translation = tokenizer.decode(
                tokens_to_decode,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
            
            # Postprocess the translation
            translation = ip.postprocess_batch([translation], lang="eng_Latn")[0]
            
            logger.info(f"Translation successful: {translation[:50]}...")
            return translation
            
        except Exception as e:
            logger.error(f"Error during token decoding: {str(e)}")
            raise
    
    except Exception as e:
        logger.error(f"Translation failed: {str(e)}")
        raise
        
async def transcribe_and_translate(audio_file: str, lang_id: str):
    """
    Transcribe an audio file and then translate the transcription to English.
    
    Args:
        audio_file: Path to the audio file
        lang_id: Language ID for transcription
        
    Returns:
        Dict containing both transcription and translation
    """
    # Import here to avoid circular imports
    from app.services.asr import initialize_asr_model, asr_transcribe
    
    try:
        logger.info(f"Starting transcription and translation for {audio_file}")
        
        # Initialize models
        asr_model = initialize_asr_model()
        translation_tokenizer, translation_model = initialize_translate_tokenizer_and_model()
        
        # Transcribe
        transcription = await asr_transcribe(asr_model, audio_file, lang_id)
        
        # Translate
        translation = await translate_indic_to_english(
            transcription, 
            lang_id, 
            translation_tokenizer,
            translation_model
        )
        
        return {
            "transcription": transcription,
            "translation": translation,
            "source_language": lang_id
        }
        
    except Exception as e:
        logger.error(f"Transcribe and translate failed: {str(e)}")
        raise

if __name__ == "__main__":
    import asyncio
    
    async def test():
        # Example usage
        from app.services.asr import initialize_asr_model, asr_transcribe
        
        # Initialize models
        asr_model = initialize_asr_model()
        translation_tokenizer, translation_model = initialize_translate_tokenizer_and_model()
        
        # Transcribe
        audio_file = r"Dataset/sample_audio_infer_ready.wav"
        lang_id = "hi"
        transcription = await asr_transcribe(asr_model, audio_file, lang_id)
        print(f"Transcription: {transcription}")
        
        # Translate
        translation = await translate_indic_to_english(
            transcription, 
            lang_id,
            translation_tokenizer,
            translation_model
        )
        print(f"Translation: {translation}")
    
    # Run the test
    asyncio.run(test())