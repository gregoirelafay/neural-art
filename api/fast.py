import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import joblib
from pydantic import BaseModel
import shutil
from tensorflow.keras import models


PATH_TO_LOCAL_MODEL = '/Users/axlav/code/Aximande/Neural-Art/model_test_webbapp'

app = FastAPI()

@app.get("/")
def home ():
    return {"greeting": "Hello world"}


@app.post("/uploadfile/")
async def root(file: UploadFile = File(...)):
    with open(f'{file.filename}',"wb") as buffer:
        shutil.copyfileobj(file.file,buffer)

    return {"file_name":file.filename}

@app.get("/predict")
def predict(img):
    pipeline = tensorflow.keras.models.load_model(PATH_TO_LOCAL_MODEL, custom_objects=None, compile=True, options=None)
