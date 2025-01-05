import mlflow
import mlflow.pytorch
import dagshub
dagshub.init(repo_owner='prathamesh.khade20', repo_name='Mlflow_for_llm_evaluation', mlflow=True)
mlflow.set_experiment("LLM_Evaluation")
mlflow.set_tracking_uri("https://dagshub.com/prathamesh.khade20/Watre_test.mlflow")  # URL to track the experiment
  


with mlflow.start_run() as run:
    # Log experiment metadata
    mlflow.log_param("model_name", "bert")
    mlflow.log_param("model_version", "v1.0")
    mlflow.log_param("evaluation_task", "sentiment_analysis")

with mlflow.start_run() as run:
	  mlflow.log_param("dataset_size",  len(dataset['test']))
         

         
from sklearn.metrics import accuracy_score, f1_score
# Assuming y_true and y_pred are true labels and model predictions
accuracy = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred, average='weighted')
with mlflow.start_run() as run:
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("f1_score", f1)