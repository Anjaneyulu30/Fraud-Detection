import mlflow
import mlflow.xgboost
mlflow.set_tracking_uri("http://127.0.0.1:5000")
MODEL_NAME="Fixed_Model"
MODEL_ALIAS="challenger"
model_uri=f"models:/{MODEL_NAME}@challenger"
model=mlflow.xgboost.load_model(model_uri)
client=mlflow.tracking.MlflowClient()
model_version=client.get_model_version_by_alias(MODEL_NAME,MODEL_ALIAS)
run=client.get_run(model_version.run_id)
business_threshold=float(run.data.params["threshold"])
print("model loaded succesfully")
print("Business_threshold",business_threshold)