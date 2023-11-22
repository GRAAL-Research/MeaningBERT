import os.path

from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_path = os.path.join("datastore", "V1")
model = AutoModelForSequenceClassification.from_pretrained(model_path)

model.push_to_hub("davebulaval/MeaningBERT")

tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.push_to_hub("davebulaval/MeaningBERT")
