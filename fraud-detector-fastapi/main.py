from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

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
def transcribe_audio(file: UploadFile = File(...)):
    fake_transcript = "Hello, your bank account is blocked. Please share OTP."
    return {"text": fake_transcript}
