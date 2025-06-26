import json
import csv
import glob
import re
import base64
import wave
import io
import os


# Function to load a JSON file
def load_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    return data


# Function to save a JSON file
def save_json_file(file_path, data):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)


# Function to extract numbers from filenames for numerical sorting
def extract_number(filename):
    match = re.search(r'generated_dataset_(\d+)\.json', filename)
    return int(match.group(1)) if match else 0


def combine_json_files(input_pattern, output_file):
    files = sorted(glob.glob(input_pattern), key=extract_number)

    # Initialize an empty list to hold combined data
    combined_data = []

    # Process each file
    for file in files:
        with open(file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # Check if the data is a list
            if isinstance(data, list):
                combined_data.extend(data)
            else:
                print(f"Warning: {file} does not contain a list. Skipping.")

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(combined_data, f, indent=4)


def move_json_files(input_pattern, output_dir):
    files = sorted(glob.glob(input_pattern), key=extract_number)

    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # move each file to the output directory
    for file in files:
        filename = os.path.basename(file)
        output_file = os.path.join(output_dir, filename)
        os.rename(file, output_file)


def json_to_csv(input_pattern, output_path):
    # Initialize list to store all data
    all_data = []

    # Find all JSON files matching the pattern
    json_files = glob.glob(input_pattern)

    if not json_files:
        print("No JSON files found matching the pattern")
        return

    # Load data from each JSON file
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                all_data.extend(data)
            print(f"Processed {json_file}")
        except Exception as e:
            print(f"Error processing {json_file}: {str(e)}")

    if not all_data:
        print("No data to write to CSV")
        return

    # Write combined data to CSV
    with open(output_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f, delimiter=',', quoting=csv.QUOTE_ALL)
        # Write header
        writer.writerow(["Type", "Category", "Message"])
        # Write data rows
        writer.writerows(all_data)

    print(f"Combined {len(json_files)} JSON files into {output_path}")


def combine_wav_fragments(audio_base64_list):
    combined_audio_data = b""
    wave_params = None
    target_nframes = 0

    for i, fragment in enumerate(audio_base64_list):
        decoded_fragment = base64.b64decode(fragment)
        with io.BytesIO(decoded_fragment) as wav_io:
            with wave.open(wav_io, 'rb') as wav_file:
                params = wav_file.getparams()
                format_params = (params.nchannels, params.sampwidth,
                                 params.framerate, params.comptype,
                                 params.compname)
                if i == 0:
                    wave_params = params
                    format_reference = format_params
                    audio_frames = wav_file.readframes(wav_file.getnframes())
                    combined_audio_data = audio_frames
                    target_nframes = params.nframes
                elif format_params != format_reference:
                    continue
                else:
                    audio_frames = wav_file.readframes(wav_file.getnframes())
                    combined_audio_data += audio_frames
                    target_nframes += params.nframes

    if not combined_audio_data:
        return None, None

    updated_params = wave_params._replace(nframes=target_nframes)

    return combined_audio_data, updated_params


def create_wave_file(output_file, audio_data, params):
    with wave.open(output_file, 'wb') as wav_out:
        wav_out.setparams(params)
        wav_out.writeframes(audio_data)


# Helper function to fetch the API key from a file
def get_api_key(file_path, name):
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith(name):
                return line.split(':', 1)[1].strip()
    return None
