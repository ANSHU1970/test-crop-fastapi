import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from io import BytesIO
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Allow all origins for CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load all models and set class names
models = {
    'corn': {
        'model': load_model('corn_model.keras'),
        'class_names': ['Blight', 'Common Rust', 'Gray Leaf Spot', 'Healthy'],
        'confidence_threshold': 36,
        'img_size': (256, 256)
    },
    'pea': {
        'model': load_model('pea_model.keras'),
        'class_names': ['Downy Mildew', 'Healthy', 'Leafminner', 'Powder Mildew'],
        'confidence_threshold': 50,
        'img_size': (256, 256)
    },
    'potato': {
        'model': load_model('potato_model.keras'),
        'class_names': ['Early Blight', 'Late Blight', 'Healthy'],
        'confidence_threshold': 50,
        'img_size': (256, 256)
    },
    'rice': {
        'model': load_model('rice_model.keras'),
        'class_names': [
            'bacterial leaf blight', 'brown spot', 'healthy',
            'leaf blast', 'leaf scald', 'narrow brown spot'
        ],
        'confidence_threshold': 50,
        'img_size': (256, 256)
    },
    'tomato': {
        'model': load_model('tomato_model.keras'),
        'class_names': [
            'Tomato Bacterial spot', 'Tomato Early blight', 'Tomato Late blight',
            'Tomato Leaf Mold', 'Tomato Septoria leaf spot',
            'Tomato Spider mites Two spotted_spider_mite',
            'Tomato Target Spot', 'Tomato YellowLeaf Curl Virus',
            'Tomato mosaic virus', 'Tomato healthy'
        ],
        'confidence_threshold': 50,
        'img_size': (128, 128)
    },
    'wheat': {
        'model': load_model('wheat_model.keras'),
        'class_names': ['Healthy', 'septoria', 'stripe rust'],
        'confidence_threshold': 50,
        'img_size': (256, 256)
    },
}

def predict_disease(image: Image.Image, crop: str):
    try:
        config = models[crop]
        if image.mode != "RGB":
            image = image.convert("RGB")
        image = image.resize(config['img_size'])
        img_array = tf.keras.preprocessing.image.img_to_array(image)
        img_array = np.expand_dims(img_array, 0)
        predictions = config['model'].predict(img_array)
        result = config['class_names'][np.argmax(predictions)]
        confidence = round(100 * (np.max(predictions)), 2)
        if confidence < config['confidence_threshold']:
            return {"disease": "can't say for sure", "confidence": f"{confidence}%"}
        else:
            return {"disease": result, "confidence": f"{confidence}%"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict/corn")
async def predict_corn(file: UploadFile = File(...)):
    try:
        image = Image.open(BytesIO(await file.read()))
        prediction = predict_disease(image, 'corn')
        return JSONResponse(content=prediction)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")

@app.post("/predict/pea")
async def predict_pea(file: UploadFile = File(...)):
    try:
        image = Image.open(BytesIO(await file.read()))
        prediction = predict_disease(image, 'pea')
        return JSONResponse(content=prediction)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")

@app.post("/predict/potato")
async def predict_potato(file: UploadFile = File(...)):
    try:
        image = Image.open(BytesIO(await file.read()))
        prediction = predict_disease(image, 'potato')
        return JSONResponse(content=prediction)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")

@app.post("/predict/rice")
async def predict_rice(file: UploadFile = File(...)):
    try:
        image = Image.open(BytesIO(await file.read()))
        prediction = predict_disease(image, 'rice')
        return JSONResponse(content=prediction)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")

@app.post("/predict/tomato")
async def predict_tomato(file: UploadFile = File(...)):
    try:
        image = Image.open(BytesIO(await file.read()))
        prediction = predict_disease(image, 'tomato')
        return JSONResponse(content=prediction)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")

@app.post("/predict/wheat")
async def predict_wheat(file: UploadFile = File(...)):
    try:
        image = Image.open(BytesIO(await file.read()))
        prediction = predict_disease(image, 'wheat')
        return JSONResponse(content=prediction)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")

# Run the API server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)
