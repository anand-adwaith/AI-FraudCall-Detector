import pandas as pd

data = pd.read_csv("Dataset/merged_call_text.txt", sep="\t", names=["Label", "Text", "Type"])
data.to_csv("Dataset/merged_call_text.csv", index=False)