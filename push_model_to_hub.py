import os.path

import huggingface_hub
from dotenv import dotenv_values
from transformers import AutoModelForSequenceClassification, AutoTokenizer

secrets = dotenv_values(".env")

huggingface_hub.login(token=secrets["huggingface_token"])

model_path = os.path.join(".", "datastore", "V2")
model = AutoModelForSequenceClassification.from_pretrained(model_path)

model.push_to_hub("davebulaval/MeaningBERT")

tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.push_to_hub("davebulaval/MeaningBERT")
