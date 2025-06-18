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
    src_lang_id (str): The source language code of the input text (e.g., "hin_Deva" for Hindi).
        Assamese: "asm_Beng"                Bengali: "ben_Beng"             Bodo: "brx_Deva"
        Dogri: "doi_Deva"                   Gujarati: "guj_Gujr"            Hindi: "hin_Deva"
        Kannada: "kan_Knda"                 Konkani: "gom_Deva"             Kashmiri (Arabic): "kas_Arab"
        Kashmiri (Devanagari): "kas_Deva"   Maithili: "mai_Deva"            Malayalam: "mal_Mlym"
        Manipuri (Bengali): "mni_Beng"      Manipuri (Meitei): "mni_Mtei"   Marathi: "mar_Deva"
        Nepali: "npi_Deva"                  Odia: "ory_Orya"                Punjabi: "pan_Guru"
        Sanskrit: "san_Deva"                Santali: "sat_Olck"             Sindhi (Arabic): "snd_Arab"
        Sindhi (Devanagari): "snd_Deva"     Tamil: "tam_Taml"               Telugu: "tel_Telu"
        Urdu: "urd_Arab"                    English: "eng_Latn"
outputs:
    translation (str): The translated text in English.
"""
def translate_indic_to_english(input_text, src_lang_id):
    indic_en_ckpt_dir = "ai4bharat/indictrans2-indic-en-dist-200M"
    indic_en_tokenizer, indic_en_model = initialize_model_and_tokenizer(indic_en_ckpt_dir)

    ip = IndicProcessor(inference=True)

    # Preprocess the input text
    input_batch = ip.preprocess_batch([input_text], src_lang=src_lang_id, tgt_lang="eng_Latn")

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
    print(f"{"eng_Latn"}: {translation}")

    return translation
