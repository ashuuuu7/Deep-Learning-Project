from fastapi import FastAPI
import numpy as np
import tensorflow as tf

app = FastAPI()

model = tf.keras.models.load_model("cats_dogs_classifier.keras")

@app.get("/")
def home():
    return {"message": "ML API is running"}

@app.post("/predict")
def predict(data: list):
    input_data = np.array(data).reshape(1, -1)
    prediction = model.predict(input_data)
    return {"prediction": prediction.tolist()}