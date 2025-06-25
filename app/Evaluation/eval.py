import sys
import os
import time
import pandas as pd
import json
import logging
from typing import Dict, Any, List
from pathlib import Path
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, classification_report
import numpy as np
import csv
import datetime
import pickle

# Add project root to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Import necessary modules
from app.utils import ModelManager
from app.services.rag_service import run_rag_query, run_few_shot_query, get_llm_for_few_shot

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    filename="evaluation.log",
    filemode="w"
)
logger = logging.getLogger("evaluation")

# Define modified schemas for simpler outputs
classify_eval_schema = {
    "name": "classify_scam",
    "description": "Used by AI to classify if a given text is a Scam / Not Scam / Suspicious",
    "parameters": {
        "type": "object",
        "properties": {
            "classification": {
                "type": "string",
                "enum": ["Scam", "Not Scam", "Suspicious"],
                "description": "One of: Scam, Not Scam, Suspicious"
            }
        },
        "required": ["classification"]
    }
}

# Define modified system prompts for evaluation
EVAL_SYSTEM_PROMPT_RAG = """
You are an AI Fraud/Scam Detector.

You will be given a message or transcript to analyze. 
Your task is to analyze this message using context documents retrieved from a database.
These retrieved messages include previous scam and non-scam examples, with labels and scam categories.

Your ONLY task is to classify the message as ONE of the following:
- "Scam"
- "Not Scam" 
- "Suspicious"

You must ONLY respond with ONE of these three options. Do not provide explanations, reasoning, or any additional text.
"""

EVAL_SYSTEM_PROMPT_FEW_SHOT = """
You are an AI Fraud/Scam Detector.

You will be given a message or transcript to analyze.
Your task is to classify this message based on examples you've seen.

Your ONLY task is to classify the message as ONE of the following:
- "Scam"
- "Not Scam" 
- "Suspicious"

You must ONLY respond with ONE of these three options. Do not provide explanations, reasoning, or any additional text.
"""

class EvaluationResults:
    """Class to store and display evaluation results."""
    
    def __init__(self, output_dir="results"):
        self.results = {
            "gpt_rag": {
                "predictions": [],
                "times": []
            },
            "gpt_few_shot": {
                "predictions": [],
                "times": []
            },
            "gemma_rag": {
                "predictions": [],
                "times": []
            },
            "gemma_few_shot": {
                "predictions": [],
                "times": []
            }
        }
        self.true_labels = []
        self.texts = []
        self.types = []
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def add_result(self, model_key, prediction, time_taken):
        """Add a single prediction result."""
        self.results[model_key]["predictions"].append(prediction)
        self.results[model_key]["times"].append(time_taken)
    
    def add_ground_truth(self, true_label, text, type_):
        """Add ground truth data."""
        self.true_labels.append(true_label)
        self.texts.append(text)
        self.types.append(type_)
    
    def _convert_label(self, label):
        """Convert label to standardized format."""
        label = label.lower()
        if "not" in label or "normal" in label:
            return "not_scam"
        elif "fraud" in label or "scam" in label:
            return "scam"
        else:
            return "suspicious"
    
    def calculate_metrics(self, model_key):
        """Calculate evaluation metrics for a specific model."""
        y_true = [self._convert_label(label) for label in self.true_labels]
        y_pred = [self._convert_label(pred) for pred in self.results[model_key]["predictions"]]
        
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='weighted')
        avg_time = sum(self.results[model_key]["times"]) / len(self.results[model_key]["times"])
        
        return {
            "precision": precision,
            "recall": recall,
            "accuracy": accuracy,
            "f1_score": f1,
            "avg_response_time": avg_time
        }

    def save_results_to_csv(self, sample_size):
        """Save detailed results to CSV file."""
        # filename = f"{self.output_dir}/evaluation_results_{sample_size}_{self.timestamp}.csv"
        
        # with open(filename, 'w', newline='') as csvfile:
        #     fieldnames = ['text', 'type', 'true_label', 'gpt_rag_pred', 'gpt_rag_time', 
        #                  'gpt_few_shot_pred', 'gpt_few_shot_time', 'gemma_rag_pred', 
        #                  'gemma_rag_time', 'gemma_few_shot_pred', 'gemma_few_shot_time']
        #     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
        #     writer.writeheader()
        #     for i in range(len(self.true_labels)):
        #         writer.writerow({
        #             'text': self.texts[i],
        #             'type': self.types[i],
        #             'true_label': self.true_labels[i],
        #             'gpt_rag_pred': self.results['gpt_rag']['predictions'][i],
        #             'gpt_rag_time': self.results['gpt_rag']['times'][i],
        #             'gpt_few_shot_pred': self.results['gpt_few_shot']['predictions'][i],
        #             'gpt_few_shot_time': self.results['gpt_few_shot']['times'][i],
        #             'gemma_rag_pred': self.results['gemma_rag']['predictions'][i],
        #             'gemma_rag_time': self.results['gemma_rag']['times'][i],
        #             'gemma_few_shot_pred': self.results['gemma_few_shot']['predictions'][i],
        #             'gemma_few_shot_time': self.results['gemma_few_shot']['times'][i]
        #         })
        
        # Save summary metrics
        summary_filename = f"{self.output_dir}/summary_metrics_{sample_size}_{self.timestamp}.csv"
        with open(summary_filename, 'w', newline='') as csvfile:
            fieldnames = ['model', 'precision', 'recall', 'accuracy', 'f1_score', 'avg_response_time']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for model_key in self.results.keys():
                metrics = self.calculate_metrics(model_key)
                metrics['model'] = model_key
                writer.writerow(metrics)
        
        print(f"Results saved to{summary_filename}")
        return summary_filename

    def print_summary(self):
        """Print summary of evaluation results."""
        print("\n===== EVALUATION SUMMARY =====")
        
        for model_key in self.results.keys():
            metrics = self.calculate_metrics(model_key)
            print(f"\n--- {model_key.upper()} ---")
            print(f"Accuracy: {metrics['accuracy']:.4f}")
            print(f"Precision: {metrics['precision']:.4f}")
            print(f"Recall: {metrics['recall']:.4f}")
            print(f"F1 Score: {metrics['f1_score']:.4f}")
            print(f"Average Response Time: {metrics['avg_response_time']:.2f} seconds")

def extract_classification(response):
    """Extract classification from model response."""
    try:
        # Extract from model output
        answer = response.get('answer', {})
        classification = answer.get('classification', '')
        
        # Clean up and normalize response
        if not classification:
            return "Error"
        
        classification = classification.lower()
        print(f"Extracted classification: {classification}")
        if "not" in classification:
            return "Not Scam"
        elif "scam" in classification or "fraud" in classification:
            return "Scam"
        elif "suspicious" in classification:
            return "Scam"
        else:
            return "Error"
    except Exception as e:
        logger.error(f"Error extracting classification: {e}")
        return "Error"

def run_evaluation(dataset_path, sample_size=10, top_k=5):
    """
    Run evaluation on multiple models and modes.
    
    Args:
        dataset_path: Path to the test dataset CSV
        sample_size: Number of samples to evaluate
        top_k: Number of documents to retrieve in RAG mode
    """
    # Initialize evaluation results
    results = EvaluationResults()
    
    # Load test dataset
    try:
        df = pd.read_csv(dataset_path)
        logger.info(f"Loaded dataset with {len(df)} rows")
        
        if sample_size > len(df):
            sample_size = len(df)
            logger.warning(f"Sample size larger than dataset. Using full dataset ({sample_size} rows)")
        
        # Use a subset for evaluation
        eval_df = df.head(sample_size)
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        print(f"Error loading dataset: {e}")
        return
    
    # Initialize models
    logger.info("Initializing models...")
    try:
        ModelManager.initialize(top_k=top_k)
        ModelManager.initialize_gemma()
    except Exception as e:
        logger.error(f"Error initializing models: {e}")
        print(f"Error initializing models: {e}")
        return
    
    # Get components for different modes
    retriever_call, llm_call = ModelManager.get_components(message_type="call", mode="rag")
    retriever_text, llm_text = ModelManager.get_components(message_type="text", mode="rag")
    _, few_shot_call = ModelManager.get_components(message_type="call", mode="few_shot")
    _, few_shot_text = ModelManager.get_components(message_type="text", mode="few_shot")
    
    # Process each sample
    for index, row in eval_df.iterrows():
        text = row['Text']
        label = row['Label']
        type_ = row['Type']
        
        logger.info(f"Processing sample {index + 1}/{sample_size}: {text[:50]}...")
        
        # Add ground truth
        results.add_ground_truth(label, text, type_)
        
        # Message type determination (assuming all are text messages for now)
        message_type = "text"  # Default to text type
        
        # Evaluate GPT with RAG
        try:
            logger.info("Evaluating GPT with RAG...")
            start_time = time.time()
            gpt_rag_response = run_rag_query(
                query=text,
                retriever=retriever_text if message_type == "text" else retriever_call,
                llm=llm_text if message_type == "text" else llm_call,
                mode=message_type,
                model_type="GPT"
            )
            elapsed_time = time.time() - start_time
            gpt_rag_classification = extract_classification(gpt_rag_response)
            results.add_result("gpt_rag", gpt_rag_classification, elapsed_time)
            logger.info(f"GPT RAG classification: {gpt_rag_classification}, time: {elapsed_time:.2f}s")
        except Exception as e:
            logger.error(f"Error with GPT RAG: {e}")
            results.add_result("gpt_rag", "Error", 0)
        
        # Evaluate GPT with Few-Shot
        try:
            logger.info("Evaluating GPT with Few-Shot...")
            start_time = time.time()
            gpt_few_shot_response = run_few_shot_query(
                query=text,
                llm=few_shot_text if message_type == "text" else few_shot_call,
                mode=message_type,
                model_type="GPT"
            )
            elapsed_time = time.time() - start_time
            gpt_few_shot_classification = extract_classification(gpt_few_shot_response)
            results.add_result("gpt_few_shot", gpt_few_shot_classification, elapsed_time)
            logger.info(f"GPT Few-Shot classification: {gpt_few_shot_classification}, time: {elapsed_time:.2f}s")
        except Exception as e:
            logger.error(f"Error with GPT Few-Shot: {e}")
            results.add_result("gpt_few_shot", "Error", 0)
        
        # Evaluate Gemma with RAG
        try:
            logger.info("Evaluating Gemma with RAG...")
            start_time = time.time()
            gemma_rag_response = run_rag_query(
                query=text,
                retriever=retriever_text if message_type == "text" else retriever_call,
                llm=llm_text if message_type == "text" else llm_call,
                mode=message_type,
                model_type="gemma"
            )
            elapsed_time = time.time() - start_time
            gemma_rag_classification = extract_classification(gemma_rag_response)
            results.add_result("gemma_rag", gemma_rag_classification, elapsed_time)
            logger.info(f"Gemma RAG classification: {gemma_rag_classification}, time: {elapsed_time:.2f}s")
        except Exception as e:
            logger.error(f"Error with Gemma RAG: {e}")
            results.add_result("gemma_rag", "Error", 0)
        
        # Evaluate Gemma with Few-Shot
        try:
            logger.info("Evaluating Gemma with Few-Shot...")
            start_time = time.time()
            gemma_few_shot_response = run_few_shot_query(
                query=text,
                llm=few_shot_text if message_type == "text" else few_shot_call,
                mode=message_type,
                model_type="gemma"
            )
            elapsed_time = time.time() - start_time
            gemma_few_shot_classification = extract_classification(gemma_few_shot_response)
            results.add_result("gemma_few_shot", gemma_few_shot_classification, elapsed_time)
            logger.info(f"Gemma Few-Shot classification: {gemma_few_shot_classification}, time: {elapsed_time:.2f}s")
        except Exception as e:
            logger.error(f"Error with Gemma Few-Shot: {e}")
            results.add_result("gemma_few_shot", "Error", 0)
        
        print(f"Completed sample {index + 1}/{sample_size}")
    
    # Save and print results
    # save results as pickle file
    with open(f"{results.output_dir}/evaluation_results_{sample_size}_{results.timestamp}.pkl", 'wb') as f:
        pickle.dump(results, f)
    logger.info("Evaluation completed. Saving results...")
    print("Evaluation completed. Saving results...")
    results.print_summary()
    # load pickle file
    with open(f"{results.output_dir}/evaluation_results_{sample_size}_{results.timestamp}.pkl", 'rb') as f:
        results = pickle.load(f)
    # save results to csv
    results.save_results_to_csv(sample_size)
    
    
    return results

if __name__ == "__main__":
    # Path to test dataset
    test_dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../Dataset/test.csv"))
    
    # First run with 10 samples
    # print("\n===== RUNNING EVALUATION WITH 10 SAMPLES =====")
    # results_10 = run_evaluation(test_dataset_path, sample_size=10)
    
    # Then run with all samples (100)
    print("\n===== RUNNING EVALUATION WITH ALL SAMPLES =====")
    results_all = run_evaluation(test_dataset_path, sample_size=100)
    
    print("\nEvaluation completed!")
