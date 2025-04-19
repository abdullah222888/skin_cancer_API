# main.py

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = FastAPI()

# Load model once when the server starts
model = tf.keras.models.load_model("flexible_skin_cancer_model.h5")

# Class mapping (update if needed)
index_to_class = {
    0: 'akiec',
    1: 'bcc',
    2: 'bkl',
    3: 'df',
    4: 'mel',
    5: 'nv',
    6: 'vasc'
}

# Preprocess image
def preprocess_image(file) -> np.ndarray:
    img = Image.open(io.BytesIO(file)).convert("RGB")
    img = img.resize((150, 150))
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

# Root endpoint
@app.get("/")
def read_root():
    return {"message": "Skin Cancer Classifier API is running 🚀"}

# Predict endpoint
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    img_array = preprocess_image(contents)

    prediction = model.predict(img_array)
    predicted_index = np.argmax(prediction)
    predicted_label = index_to_class[predicted_index]
    confidence = float(prediction[0][predicted_index])

    return JSONResponse({
        "predicted_class": predicted_label,
        "confidence": f"{confidence*100:.2f}%"
    })
