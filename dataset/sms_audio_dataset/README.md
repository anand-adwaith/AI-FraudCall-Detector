# SMS Dataset, Audio file Generator

This project generates synthetic SMS and call transcript datasets for scam and normal scenarios in the Indian context, along with corresponding audio files in multiple Indian languages. The datasets are designed for training machine learning models to detect fraudulent communications. The project uses APIs from LLM for text generation and Sarvam AI for translation and text-to-speech (TTS) functionalities.

## Project Structure

- **generate_scam_sms_dataset.py**: Generates synthetic SMS datasets for scam and normal messages, categorized by type (e.g., Phishing, Social). Outputs JSON and CSV files.
- **generate_scam_call_transcript.py**: Creates synthetic call transcript datasets for scam and normal conversations, with detailed speaker and gender information. Outputs JSON files.
- **audio_file_generator.py**: Converts call transcripts into audio files in multiple Indian languages (e.g., Hindi, Tamil) using Sarvam AI's TTS API.
- **sarvam_wrapper.py**: Provides wrapper functions for interacting with Sarvam AI APIs (translation, TTS, STT, chat).
- **transcript_extracter.py**: Extracts and formats conversations from call transcript datasets based on type, category, and speaker filters.
- **misc_utils.py**: Contains utility functions for file handling, JSON/CSV processing, and audio file manipulation.
- **requirements.txt**: Lists Python dependencies required for the project.

## Prerequisites

- **Python**: Version 3.8 or higher.
- **API Keys**:
  - **xAI API Key**: For Grok model access (text generation).
  - **Sarvam AI API Key**: For translation and TTS functionalities.
  - Store keys in `config/api_key.txt` with format:
    ```
    GROK_API_KEY:your_xai_api_key
    SARVAM_API_KEY:your_sarvam_api_key
    ```
- **Dependencies**: Install required packages using:
  ```bash
  pip install -r requirements.txt
  ```

## Directory Structure

```
project_root/
│
├── config/
│   └── api_key.txt               # API keys for xAI and Sarvam AI
├── sms_dataset_input/            # Input SMS datasets (TXT, JSON)
├── sms_dataset_output/           # Generated SMS datasets (JSON, CSV)
├── call_transcript_dataset_output/ # Generated call transcript datasets (JSON)
├── audio_output/                 # Generated audio files (WAV)
├── generate_scam_sms_dataset.py
├── generate_scam_call_transcript.py
├── audio_file_generator.py
├── sarvam_wrapper.py
├── transcript_extracter.py
├── misc_utils.py
├── requirements.txt
└── README.md
```

## Usage

### 1. Generate SMS Dataset
Run `generate_scam_sms_dataset.py` to create synthetic SMS datasets for scam and normal messages.
```bash
python generate_scam_sms_dataset.py
```
- **Input**: Existing datasets in `sms_dataset_input/` (TXT, JSON).
- **Output**: JSON and CSV files in `sms_dataset_output/` (e.g., `generated_sms_dataset_YYYYMMDD_HHMMSS.csv`).
- **Categories**:
  - Scam: Emotional Manipulation, Phishing, etc.
  - Normal: Social, Delivery Update, etc.
- **Features**:
  - Generates messages under 160 characters.
  - Includes Indian-specific references (e.g., SBI, Diwali).
  - Supports multiple datasets with unique messages.

### 2. Generate Call Transcript Dataset
Run `generate_scam_call_transcript.py` to create synthetic call transcript datasets.
```bash
python generate_scam_call_transcript.py
```
- **Output**: JSON files in `call_transcript_dataset_output/` (e.g., `generated_transcript_dataset_0.json`).
- **Features**:
  - Generates 10-20 turn conversations with alternating speakers.
  - Includes scam (e.g., Financial Fraud) and normal (e.g., Family) categories.
  - Uses Indian-specific references and realistic dialogue.

### 3. Generate Audio Files
Run `audio_file_generator.py` to convert call transcripts into audio files.
```bash
python audio_file_generator.py
```
- **Input**: Call transcript datasets in `call_transcript_dataset_output/`.
- **Output**: WAV files in `audio_output/` (e.g., `audio_fraud_Phishing_hi-IN_male_1.wav`).
- **Features**:
  - Supports multiple languages: English (en-IN), Hindi (hi-IN), Tamil (ta-IN), Kannada (kn-IN).
  - Uses Sarvam AI's TTS API with male/female speaker options.
  - Translates transcripts to target languages before audio generation.

## Configuration

- **API Keys**: Ensure `config/api_key.txt` contains valid xAI and Sarvam AI API keys.
- **Directory Paths**:
  - Update `DATASET_INPUT_DIR`, `DATASET_OUTPUT_DIR`, `AUDIO_OUTPUT_DIR`, etc., in respective scripts if needed.
- **Dataset Parameters**:
  - Adjust `num_datasets` and `num_entries_per_category` in `generate_scam_sms_dataset.py` and `generate_scam_call_transcript.py` for dataset size.
  - Modify `languages` in `audio_file_generator.py` to add/remove supported languages.

## Dependencies

Listed in `requirements.txt`:
- `backoff`: For retrying failed API calls.
- `openai`: For API interactions.
- `requests`: For HTTP requests to Sarvam AI APIs.
- `langchain_openai`: For Grok model integration.

Install with:
```bash
pip install -r requirements.txt
```

## Notes

- **Ethical Use**: The generated datasets are synthetic and intended for research or training purposes to combat fraud. Do not use for malicious activities.
- **API Limits**: Ensure sufficient API quotas for xAI and Sarvam AI, as text and audio generation can be resource-intensive.
- **Data Quality**: Generated messages/transcripts may require manual review for realism and accuracy.
- **Extensibility**: Add new categories or languages by updating `scam_categories`, `normal_categories`, or `languages` in the scripts.

## Example Output

- **SMS Dataset (CSV)**:
  ```csv
  Type,Category,Message
  fraud,Phishing,"Your SBI account is locked! Verify now at http://sbi-secure.in or lose access."
  normal,Social,"Hey, movie night at my place this Sat? Bring snacks!"
  ```
- **Call Transcript (JSON)**:
  ```json
  [
    {
      "type": "fraud",
      "category": "Financial Fraud",
      "transcript": [
        {"speaker": "scammer", "gender": "male", "message": "Sir, this is Vikram from SBI. Your account is at risk!"},
        {"speaker": "person 1", "gender": "female", "message": "What’s wrong? I didn’t get any alerts."},
        ...
      ]
    }
  ]
  ```
- **Audio File**: `audio_fraud_Financial Fraud_hi-IN_male_1.wav` (Hindi audio of a scam call transcript).

## Troubleshooting

- **API Errors**: Verify API keys and network connectivity. Check Sarvam AI/xAI documentation for error codes.
- **Empty Outputs**: Ensure input directories contain valid datasets and `num_entries_per_category` is set appropriately.
- **Audio Issues**: Confirm WAV file compatibility and Sarvam AI TTS API response format.

## License

This project is licensed under the MIT License. See `LICENSE` file for details (not included in this repository).