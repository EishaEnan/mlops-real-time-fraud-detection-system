import os, tempfile, re
import mlflow
from mlflow.tracking import MlflowClient
import boto3

BUCKET_ROOT = os.getenv("ARTIFACTS_URI", "").rstrip("/")
assert BUCKET_ROOT.startswith("s3://"), "ARTIFACTS_URI must be s3://... (no trailing slash)"
bucket = BUCKET_ROOT.split("://",1)[1].split("/",1)[0]
prefix_base = "/".join(BUCKET_ROOT.split("/",3)[3:])  # '' or existing prefix
if prefix_base: prefix_base += "/"

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI","http://mlflow:5000"))
name = os.getenv("MODEL_NAME","fraud_xgb"); alias = os.getenv("MODEL_ALIAS","staging")
c = MlflowClient()
mv = c.get_model_version_by_alias(name, alias)
print(f"Resolved {name}@{alias} -> v{mv.version}, run_id={mv.run_id}")
print("mv.source:", mv.source)

region = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION")
s3 = boto3.client("s3", region_name=region)

# find the registry directory that contains this version’s model
# we look under artifacts/fraud_train/models/**/artifacts/ and pick the newest folder that has MLmodel
scan_prefix = f"{prefix_base}artifacts/fraud_train/models/"
resp = s3.list_objects_v2(Bucket=bucket, Prefix=scan_prefix, Delimiter="/")
# paginate through subfolders
folders = set()
while True:
    for cp in resp.get("CommonPrefixes", []):
        folders.add(cp["Prefix"])
    if resp.get("IsTruncated"):
        resp = s3.list_objects_v2(Bucket=bucket, Prefix=scan_prefix, Delimiter="/",
                                  ContinuationToken=resp["NextContinuationToken"])
    else:
        break

candidates = []
for f in sorted(folders):
    # look for MLmodel under <folder>/artifacts/
    mlmodel_key = f + "artifacts/MLmodel"
    head = s3.list_objects_v2(Bucket=bucket, Prefix=mlmodel_key, MaxKeys=1)
    if head.get("KeyCount"):
        candidates.append(f)

if not candidates:
    raise SystemExit("No registry folders with MLmodel found under " + scan_prefix)

chosen = sorted(candidates)[-1]  # pick the newest by path (works if m-ids are time-based)
s3_model_dir = "s3://"+bucket+"/"+chosen+"artifacts"
print("Chosen registry path:", s3_model_dir)

with tempfile.TemporaryDirectory() as td:
    local_dir = mlflow.artifacts.download_artifacts(s3_model_dir, dst_path=td)
    import os as _os
    print("Downloaded files:", _os.listdir(local_dir))
    try:
        from mlflow import xgboost as mlf_xgb
        _ = mlf_xgb.load_model(local_dir)
        print("✅ Loaded XGBoost flavor")
    except Exception as e1:
        print("XGB failed:", e1, "→ trying PyFunc")
        _ = mlflow.pyfunc.load_model(local_dir)
        print("✅ Loaded PyFunc flavor")

print("SMOKE REGISTRY DIRECT: OK")
