import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import joblib
from pydantic import BaseModel
import shutil
from tensorflow.keras import models
from neuralart.predict import *
import os

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PATH_TO_LOCAL_MODEL = os.path.join(root,"models","VGG16","20210831-135428-images_41472-unfreeze_2-batch_128")

model = models.load_model(PATH_TO_LOCAL_MODEL)

app = FastAPI()

@app.get("/")
def home ():
    return {"greeting": "Hello world"}


'''@app.post("/uploadfile/")
async def root(file: UploadFile = File(...)):
    with open(f'{file.filename}',"wb") as buffer:
        shutil.copyfileobj(file.file,buffer)

    return '''

@app.post("/uploadfile")
async def create_upload_file(file: bytes = File(...)):


    #print("\nreceived file:")
    #print(type(file))
    #print(file)

    #image_path = "image_api.png"
    predictor = Predict(file,model)


    # write file to disk
    #with open(image_path, "wb") as f:
        #f.write(file)

    predictor.decode_image(224,224) # for VGG specs!
    result = predictor.get_prediction()

    movements = {predictor.class_names[i]: result[0][i] for i in range(len(result[0]))}
    print(type(movements))
    # main_movement =  predictor.class_names[np.argmax(result[0])]
    # model -> pred
    # dict(pred=str(main_movement))


    return dict(pred=str(movements))
