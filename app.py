from flask import Flask, request, jsonify
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from azure.storage.blob import BlobServiceClient
import os

# Configuration Azure Blob Storage
AZURE_STORAGE_CONNECTION_STRING = "DefaultEndpointsProtocol=https;AccountName=monstockagebert;AccountKey=TON_CLEF_AZURE;BlobEndpoint=https://monstockagebert.blob.core.windows.net/"
CONTAINER_NAME = "models"
MODEL_BLOB_NAME = "model.safetensors"

# Télécharger le modèle depuis Azure Blob Storage
def download_model():
    blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
    blob_client = blob_service_client.get_blob_client(container=CONTAINER_NAME, blob=MODEL_BLOB_NAME)

    local_model_path = "model.safetensors"
    with open(local_model_path, "wb") as download_file:
        download_file.write(blob_client.download_blob().readall())

    return local_model_path

# Charger le modèle
print("Téléchargement du modèle en cours...")
model_path = download_model()
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", state_dict=torch.load(model_path))

# Création de l'API Flask
app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    if "text" not in data:
        return jsonify({"error": "Texte manquant"}), 400

    inputs = tokenizer(data["text"], return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

    return jsonify({"sentiment": predictions.tolist()})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
