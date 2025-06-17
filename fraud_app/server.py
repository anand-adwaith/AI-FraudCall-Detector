from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Enable CORS for local testing with Streamlit
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this in prod!
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------- For Text Classification -----------

class TextInput(BaseModel):
    text: str

@app.post("/predict")
def predict(input: TextInput):
    text = input.text.lower()
    if any(keyword in text for keyword in ["kyc", "otp", "urgent", "money", "account"]):
        return {"prediction": "Scam"}
    else:
        return {"prediction": "Not a Scam"}

# ----------- For Audio Upload (Optional) -----------

@app.post("/transcribe")
def transcribe_audio(file: UploadFile = File(...)):
    # Dummy transcriber for demo
    fake_transcript = "Hello, your bank account is blocked. Please share OTP."
    return {"text": fake_transcript}
