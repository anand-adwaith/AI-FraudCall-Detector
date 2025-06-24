import pandas as pd
import csv
import time
import random
from llmfarminf import llmfarminf

def load_examples_from_excel(filepath):
    df = pd.read_excel(filepath)
    examples = []
    for _, row in df.iterrows():
        transcript = str(row.iloc[0]).strip()
        category = str(row.iloc[1]).strip() if len(row) > 1 else ""
        if transcript:
            examples.append((transcript, category))
    return examples

def generate_samples(llm, label, examples, n_samples, sysprompt):
    samples = set()
    tries = 0
    while len(samples) < n_samples and tries < n_samples * 4:
        # Shuffle examples for more diversity in each prompt
        random.shuffle(examples)
        prompt = (
            f"Here are some examples of {label} call transcripts with their categories:\n"
            + "\n".join(
                f"{label} ({cat}): {ex}" if cat else f"{label}: {ex}"
                for ex, cat in examples[:30]
            )
            + f"\n\nGenerate a new, short, diverse Indian {label} call transcript in the same style belonging to one of the categories. "
              "Each example should be unique, use different topics, wording, and structure. "
              "Avoid repeating phrases or patterns. Also provide a suitable category for the call. Format: transcript || category."
           )
        try:
            text = llm._completion(prompt, sysprompt).strip()
            if "||" in text:
                transcript, category = map(str.strip, text.split("||", 1))
            else:
                transcript, category = text, ""
            if len(transcript) > 10 and (transcript, category) not in samples:
                samples.add((transcript, category))
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(1)
        tries += 1
        if tries % 20 == 0:
            print(f"Generated {len(samples)}/{n_samples} {label} samples...")
    return list(samples)

def main():
    llm = llmfarminf()
    sysprompt = (
        "You are an expert at generating short, realistic, and highly diverse Indian phone call transcripts. "
        "Each transcript should be a single, brief message or exchange, not a long conversation. "
        "Vary the topics, wording, and structure. Assign a suitable category to each transcript."
        "Try to avoid repeating phrases or patterns, and ensure each example is unique."
        "The tone of the prompts should be casual and conversational, reflecting typical Indian phone call interactions."
        "From a sentiment point of view have a diverse range of emotions in the transcripts, including positive, negative, and neutral tones."
        "The fraud Categories should be one 'only' of following 13 categories 'Emotional Manipulation', 'Fake Delivery Scam', 'Financial Fraud', 'Identity Theft', 'Impersonation', 'Investment Scams', 'Job Offer Scam', 'Loan Scam', 'Lottery Scam', 'Phishing', 'Service Fraud', 'Subscription Scam', 'Tech Support Scam'."
        "The normal Categories cannot be one 'only' of the following 10 categories 'Delivery Update' , 'Social' , 'Service Inquiry' , 'Entertainment' , 'Work Update' , 'Family' , 'Sports' , 'Recreation' , 'Education' , 'Travel'."      
        "Under no circumstance must there be an overlap between the categories of fraud and normal calls."
        "In the cateogry column only keep the category name. Accidentally I see even conversations are being generated with category 'None' or 'No Category'."
    )

    n_each = 1000

    fraud_examples = load_examples_from_excel("Fraud_DB.xlsx")
    normal_examples = load_examples_from_excel("Normal_DB.xlsx")

    fraud_calls = []
    normal_calls = []

    try:
        print("Generating fraud samples...")
        fraud_calls = generate_samples(llm, "fraud", fraud_examples, n_each, sysprompt)
        print("Generating normal samples...")
        normal_calls = generate_samples(llm, "normal", normal_examples, n_each, sysprompt)
    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Saving progress...")
    except Exception as e:
        print(f"\nError occurred: {e}. Saving progress...")

    # Write to SyntheticNumber_Test1000.txt
    with open("SyntheticNumber_Test1000.txt", "w", encoding="utf-8") as txt_file:
        for call, category in fraud_calls:
            txt_file.write(f"fraud\t{call}\t{category}\n")
        for call, category in normal_calls:
            txt_file.write(f"normal\t{call}\t{category}\n")

    # Write to SyntheticNumber_Test1000.csv
    with open("SyntheticNumber_Test1000.csv", "w", newline='', encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Call_Transcript", "Label", "Category"])
        for call, category in fraud_calls:
            writer.writerow([call, "fraud", category])
        for call, category in normal_calls:
            writer.writerow([call, "normal", category])

    print("Progress saved to SyntheticNumber_Test1000.txt and SyntheticNumber_Test1000.csv.")

if __name__ == "__main__":
    main()