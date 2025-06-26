import random
import json
import csv
import re
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import time
import backoff
import misc_utils as misc

CONFIG_DIR = "config"
API_KEY_FILE_NAME = "api_key.txt"
DATASET_INPUT_DIR = "sms_dataset_input"
DATASET_OUTPUT_DIR = "sms_dataset_output"
DATASET_OUTPUT_BASE_NAME = "generated_sms_dataset_"


# Function to load messages from a .txt file (tab-separated)
def load_txt_messages(filename):
    messages = {"fraud": {}, "normal": {}}
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 3:
                type, category, message = parts
                if type in messages:
                    if category not in messages[type]:
                        messages[type][category] = set()
                    messages[type][category].add(message)
    count = sum(len(msgs) for cat in messages.values() for msgs in cat.values())
    print(f"Loaded {count} messages from {filename}")
    return messages


# Function to load messages from a .json file
def load_json_messages(filename):
    messages = {"fraud": {}, "normal": {}}
    with open(filename, "r", encoding="utf-8") as f:
        dataset = json.load(f)
        for entry in dataset:
            if len(entry) == 3:
                type, category, message = entry
                if type in messages:
                    if category not in messages[type]:
                        messages[type][category] = set()
                        messages[type][category].add(message)
        count = sum(
            len(msgs) for cat in messages.values() for msgs in cat.values())
        print(f"Loaded {count} messages from {filename}")
    return messages


# Function to load all datasets (text, csv and json files) from input and out folders
def load_all_datasets(input_dir, output_dir):
    global_messages = {"fraud": {}, "normal": {}}
    input_path = Path(input_dir).absolute()
    output_path = Path(output_dir).absolute()

    def merge_messages(source_messages):
        for type in source_messages:
            for category in source_messages[type]:
                if category not in global_messages[type]:
                    global_messages[type][category] = set()
                global_messages[type][category].update(
                    source_messages[type][category])

    if input_path.exists():
        for txt_file in input_path.glob("*.txt"):
            merge_messages(load_txt_messages(txt_file))
        for json_file in input_path.glob("*.json"):
            merge_messages(load_json_messages(json_file))
    else:
        print(f"Input directory {input_path} does not exist")

    total_messages = sum(
        len(msgs) for cat in global_messages.values() for msgs in cat.values())
    print(f"Total loaded messages: {total_messages}")
    return global_messages


# Function to create a batch prompt for SMS messages
def create_batch_prompt(type,
                        category,
                        messages_dict,
                        num_examples=5,
                        batch_size=20):
    if type not in messages_dict or category not in messages_dict[
            type] or not messages_dict[type][category]:
        print(
            f"No messages found for {type}/{category}. Using default examples.")
        default_category = next(iter(messages_dict[type]), None)
        if default_category:
            selected = random.sample(
                list(messages_dict[type][default_category]),
                min(num_examples, len(messages_dict[type][default_category])))
        else:
            selected = ["Default example message: Please adjust input data."
                       ] * num_examples
    else:
        selected = random.sample(
            list(messages_dict[type][category]),
            min(num_examples, len(messages_dict[type][category])))
    if type == "normal":
        prompt = (
            f"Generate {batch_size} short, normal, non-scam SMS messages (each under 160 characters) for {category}. "
            f"Use casual, concise language typical of SMS, like updates or friendly texts. "
            f"Include Indian-specific references. "
            f"Messages should reflect a range of tones, polite or casual styles. "
            f"Ensure each message is unique and varied. Return the messages as a numbered list. Base them on these examples:\n"
        )
    else:
        prompt = (
            f"Generate {batch_size} short fraud SMS messages (each under 160 characters) for {category}. "
            f"Include urgent, suspicious elements like fake URLs (e.g., http://secure.sbi-verify.com), "
            f"phone numbers, or emotional triggers. Use unique, fictitious URLs. "
            f"Include Indian-specific references. "
            f"Messages should reflect a range of tones, including aggressive, threatening, or manipulative styles. "
            f"Ensure each message is unique and varied. Return the messages as a numbered list. Base them on these examples:\n"
        )
    for ex in selected:
        prompt += f"- {ex}\n"
    prompt += "New SMS messages (numbered list):"
    return prompt


# Function to parse batch response
def parse_batch_response(response, batch_size):
    if isinstance(response, dict):
        text = response.get("content", "").strip()
    else:
        text = response.content.strip()
    messages = []
    for line in text.split("\n"):
        line = line.strip()
        if re.match(r"^\d+[.:]\s*", line):
            message = re.sub(r"^\d+[.:]\s*", "", line).strip()
            if message and len(message) <= 160:
                messages.append(message)
    token_usage = response.response_metadata.get(
        "token_usage", {}) if isinstance(response, dict) else {}
    if len(messages) < batch_size:
        print(
            f"Generated {len(messages)} of {batch_size} messages. Token usage: {token_usage}. Consider adjusting max_tokens or prompt."
        )
    return messages, token_usage


# Function to generate messages with backoff
@backoff.on_exception(backoff.expo, Exception, max_tries=3)
def generate_batch_messages(api_key, prompt, batch_size):
    llm = ChatOpenAI(api_key=api_key,
                     base_url="https://api.x.ai/v1",
                     model="grok-3",
                     temperature=0.9,
                     max_tokens=1200)
    prompt_template = ChatPromptTemplate.from_messages([("human", "{prompt}")])
    llm_chain = prompt_template | llm
    response = llm_chain.invoke({"prompt": prompt})
    return parse_batch_response(response, batch_size)


# Helper function to generate unique messages for a category
def generate_category_messages(type,
                               category,
                               messages_dict,
                               num_required,
                               global_generated_set,
                               api_key,
                               batch_size=20):
    generated_set = set()
    max_attempts = 5
    attempts = 0
    total_tokens = 0
    while len(generated_set) < num_required and attempts < max_attempts:
        prompt = create_batch_prompt(type,
                                     category,
                                     messages_dict,
                                     num_examples=5,
                                     batch_size=batch_size)
        try:
            messages, token_usage = generate_batch_messages(
                api_key, prompt, batch_size)
            for message in messages:
                if len(
                        generated_set
                ) < num_required and message not in generated_set and message not in global_generated_set:
                    generated_set.add(message)
                    global_generated_set.add(message)
            attempts += 1
            total_tokens += token_usage.get("total_tokens", 0)
            if len(messages) == 0:
                print(
                    f"No valid messages generated for {type}/{category} on attempt {attempts}"
                )
        except Exception as e:
            print(f"Error generating messages for {type}/{category}: {e}")
            time.sleep(1)
            attempts += 1
    if len(generated_set) < num_required:
        print(
            f"Only generated {len(generated_set)} of {num_required} messages for {type}/{category}. Total tokens: {total_tokens}"
        )
    return [(type, category, msg) for msg in generated_set]


# Function to generate a single dataset
def generate_dataset(messages_dict, num_entries_per_category, api_key,
                     global_generated_set):
    dataset = []

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

    categories = [
        (type, cat)
        for type in ["fraud", "normal"]
        for cat in (scam_categories if type == "fraud" else normal_categories)
    ]
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(generate_category_messages, type, cat,
                            messages_dict, num_entries_per_category,
                            global_generated_set, api_key)
            for type, cat in categories
        ]
        for future in futures:
            dataset.extend(future.result())

    random.shuffle(dataset)
    return dataset


if __name__ == "__main__":
    api_key = misc.get_api_key(f"{CONFIG_DIR}/{API_KEY_FILE_NAME}",
                               "GROK_API_KEY")

    global_messages = load_all_datasets(DATASET_INPUT_DIR, DATASET_OUTPUT_DIR)
    global_generated_set = set()
    for type in global_messages:
        for category in global_messages[type]:
            global_generated_set.update(global_messages[type][category])

    num_datasets = 1
    for i in range(num_datasets):
        start_time = time.time()
        dataset = generate_dataset(global_messages, 10, api_key,
                                   global_generated_set)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_file = f"{DATASET_OUTPUT_DIR}/{DATASET_OUTPUT_BASE_NAME}{timestamp}.json"
        misc.save_json_file(output_file, dataset)

    # combine all generated datasets into a single csv file
    input_file_pattern = f"{DATASET_OUTPUT_DIR}/{DATASET_OUTPUT_BASE_NAME}*.json"
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_csv_file = f"{DATASET_OUTPUT_DIR}/{DATASET_OUTPUT_BASE_NAME}{timestamp}.csv"
    misc.json_to_csv(input_file_pattern, output_csv_file)
    # move the generated json files to the input directory
    misc.move_json_files(input_file_pattern, DATASET_INPUT_DIR)

    print(f"Generated dataset saved to {output_csv_file}")
