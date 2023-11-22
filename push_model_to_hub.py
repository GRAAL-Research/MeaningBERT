import os.path

from transformers import AutoModelForSequenceClassification

model_path = os.path.join("datastore", "V1")
model = AutoModelForSequenceClassification.from_pretrained(model_path)

model.push_to_hub("davebulaval/MeaningBERT")
