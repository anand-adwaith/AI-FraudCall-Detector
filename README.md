# AI-FraudCall-Detector

![Fraud Call Detector Banner](https://img.shields.io/badge/AI-Fraud%20Detection-orange)

A comprehensive fraud detection system that uses artificial intelligence to identify scam calls and text messages in multiple Indian languages.

**Live Demo**: [https://frauddetectionapp.publicvm.com/](https://frauddetectionapp.publicvm.com/)

**Demo Video**: [Watch on Google Drive](https://drive.google.com/file/d/1IXspU9Jvv78TztyarFU5Apk1D78hK6dI/view?usp=share_link)

## Project Overview

The AI-FraudCall-Detector is an intelligent system designed to identify potential fraud in phone calls and SMS messages. Using various AI models including LLMs, BERT, and fine-tuned Gemma, the system analyzes text and audio inputs to determine if they exhibit characteristics of fraudulent communication. The project supports multiple Indian languages and provides both text and audio-based analysis.

## Features

- **Multi-modal Detection**: Text and audio-based fraud detection
- **Real-time Analysis**: Live recording and instant analysis capabilities
- **Multi-language Support**: Works with English, Hindi, Tamil, Kannada, and other Indian languages
- **Multiple Detection Methods**: Utilizes various AI approaches for robust detection
- **User-friendly Interface**: Easy-to-use Streamlit UI for interaction
- **Synthetic Data Generation**: Tools to create training datasets for different fraud categories

## Project Structure

```
AI-FraudCall-Detector/
│
├── app/                           # Main application backend
│   ├── __init__.py
│   ├── .env                       # Environment variables
│   ├── config.py                  # Configuration settings
│   ├── main.py                    # FastAPI main application
│   ├── utils.py                   # Utility functions
│   ├── Evaluation/                # Model evaluation tools
│   ├── IndicTransToolkit/         # Indian language support tools
│   ├── models/                    # Model definitions
│   ├── routers/                   # FastAPI routers
│   └── services/                  # Business logic services
│
├── Dataset/                       # Dataset files
│   ├── data_call.csv              # Call transcript data
│   ├── data.csv                   # Main dataset
│   ├── merged_call_data.csv       # Combined call data
│   ├── merged_call_text.csv       # Text extracted from calls
│   └── sample_audio_infer_ready.wav # Sample audio file
│
├── dataset/                       # Dataset generation tools
│   ├── call_dataset/              # Tools for call dataset generation
│   └── sms_audio_dataset/         # SMS and audio dataset generation
│       ├── audio_file_generator.py     # Generates audio from text
│       ├── generate_scam_call_transcript.py  # Creates call transcripts
│       ├── generate_scam_sms_dataset.py      # Creates SMS datasets
│       ├── sarvam_wrapper.py      # API wrapper for Sarvam AI
│       ├── requirements.txt       # Dependencies
│       └── README.md              # Documentation
│
├── fraud_app/                     # Streamlit UI application
│   ├── utils.py                   # UI utility functions
│   ├── server.py                  # FastAPI server for UI
│   ├── main.py                    # Streamlit main app
│   ├── requirements.txt           # UI dependencies
│   └── pages/                     # Streamlit pages
│       ├── 1_💬_Text_Input.py       # Text analysis page
│       ├── 2_📁_Upload_Audio.py      # Audio upload page
│       └── 3_🎤_Live_Recording.py    # Live recording page
│
├── Ingestion/                     # Data ingestion pipeline
│   ├── ingest.py                  # Data ingestion script
│   └── preprocess.py              # Data preprocessing
│
├── fraud-detector-fastapi/        # FastAPI backend service
│   └── main.py                    # API endpoint definitions
│
├── results/                       # Evaluation results
│   └── evaluation_results.csv     # Performance metrics
│
├── Temp/                          # Experimental code
│   ├── gemma_test.py              # Gemma model testing
│   ├── llama_v1.py                # LLaMA model implementation
│   └── meta_llama.py              # Meta's LLaMA testing
│
├── report/                        # Project documentation
│   └── Team_4__AI_Enabled_Scam_Detection.pdf  # Detailed project report
│
├── requirements.txt               # Project dependencies
├── README.md                      # Project overview
├── app.py                         # Combined app entry point
└── eval_result_object.joblib      # Saved evaluation results
```

## Model Approaches

The system implements four distinct AI approaches for fraud detection, each with different trade-offs between accuracy, efficiency, and deployment considerations:

### 1. BERT-Based Supervised Classification (Approach 1)
- **Model**: Fine-tuned bert-base-multilingual-cased (110M parameters)
- **Performance**: F1-score of 91.26%, perfect precision (1.0), recall of 83.93%
- **Inference Time**: 3.97 seconds per sample
- **Model Size**: ~682 MB
- **Edge Deployment Suitability**: High (4/5)
- **Benefits**: Lightweight, excellent balance between accuracy and efficiency, suitable for resource-constrained environments

### 2. Gemma 2B with LoRA Fine-tuning (Approach 2)
- **Model**: Gemma 2B decoder-only model using Low-Rank Adaptation
- **Implementation**: LoRA fine-tunes only 0.07% of parameters (1.8M out of 2.5B)
- **Performance**: 94.55% recall for detecting harmful messages
- **Size**: Base model 9.5 GB, fine-tuned adapter only 56 MB
- **Edge Deployment Suitability**: Moderate (2/5)
- **Benefits**: High recall with efficient parameter adaptation

### 3. Few-Shot + Chain of Thought Prompting (Approach 3)
- **Model**: Gemma 3 1B instruction-tuned model
- **Technique**: Rich prompts with few-shot examples and chain-of-thought reasoning
- **Response Format**: Structured JSON outputs with classification, reasoning, and follow-up questions
- **Context Window**: 32k tokens
- **Performance**: F1 score of 0.59
- **Edge Deployment Suitability**: Medium (3/5)
- **Benefits**: No fine-tuning required, adaptable to new scam types

### 4. Retrieval-Augmented Generation (RAG) with Gemma (Approach 4)
- **Process**: Two-phase approach: ingestion (dataset embedding) and retrieval
- **Components**: BGE embedding model + Gemma 1B LLM + Vector Database
- **Operation**: Retrieves top-5 similar examples as context for the LLM
- **Performance**: F1 score of 0.48
- **Edge Deployment Suitability**: Low (2/5)
- **Benefits**: Leverages example database for better contextual understanding

The project also tested GPT-based approaches (Few-Shot + CoT and RAG with GPT), which achieved the highest F1 scores (0.98) but are least suitable for edge deployment due to their cloud-based nature and massive parameter count (1.8T).

## Dataset Information

The project uses multiple types of datasets:

- **Real-world Data**: Collected from actual fraud cases and normal communications
- **Synthetic Data**: Generated using LLMs to cover a wide range of fraud categories
- **Multi-modal Data**: Includes text, audio, and transcribed content

### Fraud Categories

- Emotional Manipulation
- Fake Delivery Scam
- Financial Fraud
- Identity Theft
- Impersonation
- Investment Scams
- Job Offer Scam
- Loan Scam
- Lottery Scam
- Phishing
- Service Fraud
- Subscription Scam
- Tech Support Scam

### Normal Categories

- Delivery Update
- Social
- Service Inquiry
- Entertainment
- Work Update
- Family
- Sports
- Recreation
- Education
- Travel

## Architecture

The system follows a multimodal architecture with three main components:

1. **Input Interface**: Streamlit-based web application supporting:
   - Text message input
   - Pre-recorded audio file upload
   - Live voice recording

2. **Preprocessing and Translation**:
   - **ASR Module**: IndicConformer-600M for multilingual speech recognition
   - **Translation Module**: IndicTrans2-Distilled (200M) for Indic-to-English translation

3. **Classification and Inference**:
   - Four independent approaches (BERT, Gemma-LoRA, Few-Shot+CoT, RAG)
   - Each returns classification, reasoning explanation, and suggested follow-up questions

## Installation & Setup

### Prerequisites

- Python 3.8+
- Docker
- API keys for OpenAI/xAI and Sarvam AI (for dataset generation)

### Database Setup

```bash
# Run Qdrant vector database
sudo docker run -d --name qdrant -d -p 6333:6333 qdrant/qdrant

# Create database collections
python db_creation.py
```

### Backend Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/AI-FraudCall-Detector.git
cd AI-FraudCall-Detector

# Install dependencies
pip install -r requirements.txt

# Start the FastAPI backend
python app/main.py
```

### Frontend Setup

```bash
# Navigate to the fraud_app directory
cd fraud_app

# Install Streamlit UI dependencies
pip install -r requirements.txt

# Start the Streamlit UI
streamlit run Home.py
```

## Usage

### Text Analysis

1. Navigate to the "Text Input" tab
2. Enter the suspicious message or call transcript
3. Click "Analyze" to get results

### Audio Analysis

1. Go to the "Upload Audio" tab
2. Upload an audio file (MP3/WAV)
3. The system will transcribe and analyze the content

### Live Recording

1. Select the "Live Recording" tab
2. Click "Start Recording" to record 5 seconds of audio
3. The system will transcribe and analyze in real-time

## Evaluation Results

Based on model evaluations and trade-offs analyzed in the report:

- BERT-based classifier offers the most balanced approach for edge deployment with high accuracy (91.26% F1-score)
- GPT-based approaches provide the highest accuracy (98% F1-score) but require cloud connectivity
- Gemma with LoRA fine-tuning shows excellent recall (94.55%) with modest parameter updates
- RAG and Few-shot approaches offer flexibility but at lower accuracy on the test dataset

For detailed performance metrics and comparative analysis, see the evaluation results in the project report.

## Future Work

- Complete edge-based deployment of the full pipeline (ASR, translation, and classification) on smartphones
- Support for additional Indian languages and dialects
- Integration with telecom systems for real-time call screening
- Mobile application development
- Enhanced explainability features

## Team

This project was developed by Team 4 for the AI-Enabled Scam Detection challenge. For details, see the project report in the report folder.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgements

- Indian Institute of Science for project support
- OpenAI for GPT models
- Google for Gemma models
- Sarvam AI for Indian language TTS/STT capabilities
- xAI for Grok model access
