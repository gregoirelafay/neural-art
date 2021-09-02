import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Response
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import joblib
from pydantic import BaseModel
import shutil
from tensorflow.keras import models
from neuralart.predict import *
from neuralart.fourier import *
from neuralart.baseline import *
import os

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#PATH_TO_LOCAL_MODEL = os.path.join(root,"models","VGG16","20210831-135428-images_41472-unfreeze_2-batch_128")
PATH_TO_BASE_DICT = os.path.join(root,'baseline','baseline.npy')

PATH_TO_IMG=os.path.join(root,'api','georges-seurat_sketch-with-many-figures-for-sunday-afternoon-on-grande-jatte-1884.jpg')
#model = models.load_model(PATH_TO_LOCAL_MODEL)


app = FastAPI()

@app.get("/")
def home ():
    img=plt.imread(PATH_TO_IMG)

    img_conv=baselines_single(img)

    base_dict=np.load('baseline/baseline.npy',allow_pickle=True)[()]

    #baselines_viz_single(img_conv,array=False)

    img_pred_avg=base_pred_avg(img_conv,base_dict)
    img_pred_dom=base_pred_dom(img_conv,base_dict)
    img_pred_fft=base_pred_fft(img_conv,base_dict)


    return Response(content=base_dict, media_type="application/xml")


@app.post("/upload")
async def create_upload_file(file: bytes = File(...)):

    # adapt import to imread method
    image_path = "myimage.png"
    with open(image_path, "wb") as disk_file:
        disk_file.write(file)

    img=plt.imread(image_path)

    # drop the alpha in the image RGBA
    clean_img = img[:,:, :3]

    img_conv=baselines_single(clean_img)

    base_dict=np.load('baseline/baseline.npy',allow_pickle=True)[()]

    #baselines_viz_single(img_conv,array=False)

    img_pred_avg=base_pred_avg(img_conv,base_dict)
    img_pred_dom=base_pred_dom(img_conv,base_dict)
    img_pred_fft=base_pred_fft(img_conv,base_dict)

    return {'Average Color':img_pred_avg,
            'Dominant Color':img_pred_dom,
            'Fourier Transform':img_pred_fft}
