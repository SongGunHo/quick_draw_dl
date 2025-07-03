import os
import sys
import keras
import numpy as np
from  PIL import Image


# 기준 경로 
base_path  = os.path.dirname(os.path.realpath(__file__))


#모델 불러오기
#model = keras.models.load_model(f"${base_path}/best-model.keras")
model = keras.models.load_model("C:/models/best-model.keras")
model.summary()