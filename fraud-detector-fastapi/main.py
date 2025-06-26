from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
import audio_processing.asr as asr
import audio_processing.translate as translation

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TextInput(BaseModel):
    text: str

@app.post("/predict")
def predict(input: TextInput):
    text = input.text.lower()
    if any(keyword in text for keyword in ["kyc", "otp", "urgent", "money", "account"]):
        return {"prediction": "Scam"}
    else:
        return {"prediction": "Not a Scam"}

@app.post("/transcribe")
def transcribe_audio(file: UploadFile = File(...),
                     lang_id: str = Form(...)):
    if not file.filename.endswith(('.wav', '.mp3', '.flac')):
        return {"error": "Invalid audio file format. Please upload a .wav, .mp3, or .flac file."}

    # Save the uploaded file temporarily
    with open(file.filename, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Check if the file exists and is not empty
    if not os.path.exists(file.filename):
        return {"error": "Audio file not found. Please upload a valid audio file."}
    if os.path.getsize(file.filename) == 0:
        return {"error": "Audio file is empty. Please upload a valid audio file."}
    if not lang_id:
        return {"error": "Language ID is required. Please provide a valid language ID."}
    if lang_id not in ["as", "bn", "brx", "doi", "gu", "hi", "kn", "kok", "ks", "mai", "ml", "mni", "mr", "ne", "or", "pa", "sa", "sat", "sd", "ta", "te", "ur"]:
        return {"error": f"Unsupported language ID: {lang_id}"}

    # Perform ASR transciption
    transcription = asr.asr_transcribe(file.filename, lang_id)
    if not transcription:
        return {"error": "Transcription failed. Please check the audio file and language ID."}

    # Remove the temporary file
    os.remove(file.filename)

    return {"transcription": transcription}

@app.post("/translate")
def translate_text(input: TextInput,
                     lang_id: str = Form(...)):
    if not input.text:
        return {"error": "Input text is required for translation."}
    if lang_id not in ["as", "bn", "brx", "doi", "gu", "hi", "kn", "kok", "ks", "mai", "ml", "mni", "mr", "ne", "or", "pa", "sa", "sat", "sd", "ta", "te", "ur"]:
        return {"error": f"Unsupported language ID: {lang_id}"}

    # Perform translation
    translated_text = translation.translate_indic_to_english(input.text, lang_id)

    if not translated_text:
        return {"error": "Translation failed. Please check the input text and language ID."}

    return {"translated_text": translated_text}

@app.post("/transcribe_translate")
def transcribe_translate_audio(file: UploadFile = File(...),
                     lang_id: str = Form(...)):
    if not file.filename.endswith(('.wav', '.mp3', '.flac')):
        return {"error": "Invalid audio file format. Please upload a .wav, .mp3, or .flac file."}

    # Save the uploaded file temporarily
    with open(file.filename, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Check if the file exists and is not empty
    if not os.path.exists(file.filename):
        return {"error": "Audio file not found. Please upload a valid audio file."}
    if os.path.getsize(file.filename) == 0:
        return {"error": "Audio file is empty. Please upload a valid audio file."}
    if not lang_id:
        return {"error": "Language ID is required. Please provide a valid language ID."}
    if lang_id not in ["as", "bn", "brx", "doi", "gu", "hi", "kn", "kok", "ks", "mai", "ml", "mni", "mr", "ne", "or", "pa", "sa", "sat", "sd", "ta", "te", "ur"]:
        return {"error": f"Unsupported language ID: {lang_id}"}

    # Perform ASR transciption
    transcription = asr.asr_transcribe(file.filename, lang_id)
    if not transcription:
        return {"error": "Transcription failed. Please check the audio file and language ID."}

    # Remove the temporary file
    os.remove(file.filename)

    # Perform translation
    translated_transcription = translation.translate_indic_to_english(transcription, lang_id)
    if not translated_transcription:
        # return error and trascription
        return {
            "error": "Translation failed. Please check the transcription and language ID.",
            "transcription": transcription
        }

    return {"translated_transcription": translated_transcription}
