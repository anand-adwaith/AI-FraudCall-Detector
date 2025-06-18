import torch
from transformers import AutoModelForSeq2SeqLM, BitsAndBytesConfig, AutoTokenizer
from IndicTransToolkit.processor import IndicProcessor

BATCH_SIZE = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def initialize_model_and_tokenizer(ckpt_dir):
    tokenizer = AutoTokenizer.from_pretrained(ckpt_dir, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        ckpt_dir,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        quantization_config=None,
    )

    model = model.to(DEVICE)
    if DEVICE == "cuda":
        model.half()

    model.eval()

    return tokenizer, model

"""
Translate Indic languages to English using the IndicTrans2 model.
This function initializes the model and tokenizer, preprocesses the input text,
tokenizes it, generates the translation, and postprocesses the output.
inputs:
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
outputs:
    translation (str): The translated text in English.
"""
def translate_indic_to_english(input_text, src_lang_id):
    indic_en_ckpt_dir = "ai4bharat/indictrans2-indic-en-dist-200M"
    indic_en_tokenizer, indic_en_model = initialize_model_and_tokenizer(indic_en_ckpt_dir)

    ip = IndicProcessor(inference=True)

    lang_code_map = {
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

    src_lang_id_internal = lang_code_map.get(src_lang_id)

    # Preprocess the input text
    input_batch = ip.preprocess_batch([input_text], src_lang=src_lang_id_internal, tgt_lang="eng_Latn")

    # Tokenize the input text
    inputs = indic_en_tokenizer(
        input_batch,
        truncation=True,
        padding="longest",
        return_tensors="pt",
        return_attention_mask=True,
    ).to(DEVICE)

    # Generate the translation
    with torch.no_grad():
        generated_tokens = indic_en_model.generate(
            **inputs,
            use_cache=True,
            min_length=0,
            max_length=256,
            num_beams=5,
            num_return_sequences=1,
        )

    # Decode the generated tokens into text
    translation = indic_en_tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )

    # Postprocess the translation
    translation = ip.postprocess_batch([translation], lang="eng_Latn")[0]

    del indic_en_tokenizer, indic_en_model

    print(f"{src_lang_id}: {input_text}")
    print(f"{"en"}: {translation}")

    return translation
