# import pickle
# from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, classification_report
# def calculate_metrics(self, model_key):
#         """Calculate evaluation metrics for a specific model."""
#         y_true = [self._convert_label(label) for label in self.true_labels]
#         y_pred = [self._convert_label(pred) for pred in self.results[model_key]["predictions"]]
        
#         precision = precision_score(y_true, y_pred, average='weighted')
#         recall = recall_score(y_true, y_pred, average='weighted')
#         accuracy = accuracy_score(y_true, y_pred)
#         f1 = f1_score(y_true, y_pred, average='weighted')
#         avg_time = sum(self.results[model_key]["times"]) / len(self.results[model_key]["times"])
        
#         return {
#             "precision": precision,
#             "recall": recall,
#             "accuracy": accuracy,
#             "f1_score": f1,
#             "avg_response_time": avg_time
#         }

# with open(f"C:/Users/Harshavardhan A/Downloads/IISC/AI-FraudCall-Detector/results/evaluation_results_10_20250624_121914.pkl", 'rb') as f:
#         results = pickle.load(f)
# print("\n===== EVALUATION SUMMARY =====")
        
# for model_key in results.keys():
#     metrics = calculate_metrics(model_key)
#     print(f"\n--- {model_key.upper()} ---")
#     print(f"Accuracy: {metrics['accuracy']:.4f}")
#     print(f"Precision: {metrics['precision']:.4f}")
#     print(f"Recall: {metrics['recall']:.4f}")
#     print(f"F1 Score: {metrics['f1_score']:.4f}")
#     print(f"Average Response Time: {metrics['avg_response_time']:.2f} seconds")
# print(f"Evaluation results loaded")

      
import joblib
result = joblib.load(r'C:\Users\Harshavardhan A\Downloads\IISC\AI-FraudCall-Detector\eval_result_object.joblib')
print(result.keys())