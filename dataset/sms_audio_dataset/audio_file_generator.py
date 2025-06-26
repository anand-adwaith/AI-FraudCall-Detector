import glob

import transcript_extracter as extracter
import sarvam_wrapper as sarvam
import misc_utils as misc

AUDIO_OUTPUT_DIR = "audio_output"
AUDIO_OUTPUT_BASE_NAME = "audio"
CONFIG_DIR = "config"
API_KEY_FILE_NAME = "api_key.txt"
DATASET_DIR = "call_transcript_dataset_output"
CALL_TRANSCRIPT_FILE_PATTERN = "generated_transcript_dataset*.json"

scam_categories = [
    "Emotional Manipulation", "Fake Delivery Scam", "Financial Fraud",
    "Identity Theft", "Impersonation", "Investment Scams", "Job Offer Scam",
    "Lottery Scam", "Loan Scam", "Phishing", "Service Fraud",
    "Subscription Scam", "Tech Support Scam"
]

normal_categories = [
    "Delivery Update", "Social", "Service Inquiry", "Entertainment",
    "Work Update", "Family", "Sports", "Recreation", "Education", "Travel"
]

types = ["fraud", "normal"]

languages = ["en-IN", "hi-IN", "ta-IN", "kn-IN"]


def generate_audio_file(api_key, text, language, gender, file_name):
    audio_fragments = sarvam.tts(api_key, text, language, gender)
    wave_data, params = misc.combine_wav_fragments(audio_fragments)
    misc.create_wave_file(file_name, wave_data, params)


if __name__ == "__main__":
    # load the API key from the config file
    api_key = misc.get_api_key(f"{CONFIG_DIR}/{API_KEY_FILE_NAME}",
                               "SARVAM_API_KEY")

    dataset_files = glob.glob(f"{DATASET_DIR}/{CALL_TRANSCRIPT_FILE_PATTERN}")
    for file in dataset_files:
        num_samples = 2

        data = misc.load_json_file(file)
        # iterate the type, category, and speaker
        for type in types:
            for category in (scam_categories
                             if type == "fraud" else normal_categories):
                speaker = "scammer" if type == "fraud" else "person 1"

                extracted_conversations = extracter.extract_conversations(
                    data, type, category, speaker)
                if not extracted_conversations:
                    print(
                        f"No conversations found for type: {type}, category: {category}, speaker: {speaker}"
                    )
                    continue

                # sample the first num_samples entries from extracted_conversations
                for i in range(num_samples):
                    conversation = extracted_conversations[i].get(
                        'conversation')
                    gender = extracted_conversations[i].get('gender')

                    # iterate through the languages
                    for language in languages:
                        if language == "en-IN":
                            # donot translate if the language is English
                            translated_text = conversation
                        else:
                            translated_text = sarvam.translate(
                                api_key, conversation, "en-IN", language)

                        file_name = f"{AUDIO_OUTPUT_DIR}/{AUDIO_OUTPUT_BASE_NAME}_{type}_{category}_{language}_{gender}_{i+1}.wav"
                        generate_audio_file(api_key, translated_text, language,
                                            gender, file_name)
