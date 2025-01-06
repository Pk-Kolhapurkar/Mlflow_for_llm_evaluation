import torch
import mlflow
import mlflow.pytorch
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from transformers import BertForSequenceClassification, BertTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
import dagshub

# Initialize Dagshub and MLflow
dagshub.init(repo_owner='prathamesh.khade20', repo_name='Mlflow_for_llm_evaluation', mlflow=True)
mlflow.set_experiment("Sentiment_Analysis_Experiment")
mlflow.set_tracking_uri("https://dagshub.com/prathamesh.khade20/Mlflow_for_llm_evaluation.mlflow")


# Load pre-trained model and tokenizer
model_name = "textattack/bert-base-uncased-yelp-polarity"
tokenizer = BertTokenizer.from_pretrained(model_name)      
model = BertForSequenceClassification.from_pretrained(model_name)

# Load the IMDb dataset
dataset = load_dataset("imdb")

# Preprocess the dataset
def preprocess_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True)
tokenized_datasets = dataset.map(preprocess_function, batched=True)

# Function to generate predictions
def get_predictions(model, dataset):
    model.eval()  # Set model to evaluation mode
    predictions = []
    with torch.no_grad():  # No need to compute gradients for inference
        for item in dataset:
            inputs = tokenizer(item['text'], return_tensors="pt", padding=True, truncation=True, max_length=512)
            outputs = model(**inputs)
            logits = outputs.logits
            predictions.append(torch.argmax(logits, dim=-1).item())  # Get predicted class index
    return predictions

# Get predictions
y_pred = get_predictions(model, dataset['test'])

# Assuming y_true are the true labels from the dataset
y_true = dataset['test']['label']

# Log experiment metadata with MLflow
with mlflow.start_run() as run:
    mlflow.log_param("model_name", "bert")
    mlflow.log_param("model_version", "v1.0")
    mlflow.log_param("evaluation_task", "sentiment_analysis")

# Log dataset size
with mlflow.start_run() as run:
    mlflow.log_param("dataset_size", len(dataset['test']))

# Log metrics
from sklearn.metrics import accuracy_score, f1_score
accuracy = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred, average='weighted')
with mlflow.start_run() as run:
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("f1_score", f1)

# Log the model and model weights
with mlflow.start_run() as run:
    mlflow.pytorch.save_model(model, "model")
    joblib.dump(model, "model_weights.pkl") 
    mlflow.log_artifact("model", artifact_path="model")

# Log confusion matrix as an image
conf_matrix = confusion_matrix(y_true, y_pred)
cm_display = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
cm_display.plot(cmap='Blues')
plt.savefig("confusion_matrix.png") 
mlflow.log_artifact("confusion_matrix.png")

# Log generated outputs as a text file
outputs = dataset['test']
outputs['prediction'] = y_pred
with open("generated_outputs.txt", "w") as f:
    for output in outputs:
        f.write(str(output) + "\n")
mlflow.log_artifact("generated_outputs.txt")

# Define training arguments
def train_and_log_model(model, lr, bs):
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=1,
        per_device_train_batch_size=bs,
        per_device_eval_batch_size=bs,
        learning_rate=lr,
        evaluation_strategy="epoch"
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test']
    )

    # Log hyperparameters and train the model
    with mlflow.start_run() as run:
        mlflow.log_param("learning_rate", lr)
        mlflow.log_param("batch_size", bs)  
        trainer.train()
        eval_result = trainer.evaluate()      
        mlflow.log_metric("eval_accuracy", eval_result['eval_accuracy'])

# Hyperparameter grid for training
hyperparameter_grid = [
    {"learning_rate": 5e-5, "batch_size": 8},
    {"learning_rate": 3e-5, "batch_size": 16}
]

# Train and log models for different hyperparameters
for params in hyperparameter_grid:
    train_and_log_model(params["learning_rate"], params["batch_size"])

# Set MLflow experiment for comparison
mlflow.set_experiment("sentiment_analysis_comparison")
model_names = ["bert-base-cased", "bert-base-uncased"]
run_ids = []
artifact_paths = []

# Log models and evaluations
for model_name in model_names:
    with mlflow.start_run(run_name=f"log_model_{model_name}"):
        artifact_path = f"models/{model_name}"
        mlflow.pyfunc.log_model(
            artifact_path=artifact_path,
            python_model=model,
        )
        run_ids.append(mlflow.active_run().info.run_id)
        artifact_paths.append(artifact_path)

# Evaluate models
for i in range(len(model_names)):
    with mlflow.start_run(run_id=run_ids[i]):
        evaluation_results = mlflow.evaluate(
            model=f"runs:/{run_ids[i]}/{artifact_paths[i]}",
            model_type="text",
            data=dataset['test'],
        )

# Start MLflow server
#mlflow.server --backend-store-uri sqlite:///mlruns.db --default-artifact-root ./mlruns
