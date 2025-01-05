from transformers import (
BertForSequenceClassification, 
BertTokenizer
)
# Load pre-trained model and tokenizer
model_name = "textattack/bert-base-uncased-yelp-polarity"
tokenizer = BertTokenizer.from_pretrained(model_name)      
model = BertForSequenceClassification.from_pretrained(model_name)