import os
import sys
import keras
import numpy as np
import json
from  PIL import Image


if len(sys.argv) <2 :
    sys.exit(1)


# 기준 경로 
base_path  = os.path.dirname(os.path.realpath(__file__))

# 이미지 경로 
image_path = sys.argv[1]


#모델 불러오기
#model = keras.models.load_model(f"${base_path}/best-model.keras")
model = keras.models.load_model("C:/models/best-model.keras")


# 이미지 전처리 -224 * 224 이미지 반전 
Image = np.invert(Image.open(image_path).resize((224, 224)))

# 분류 목록 조회 

categories = np.load(base_path +'/category.npy')

# 추론 하기 
predictions = model.predict(image[np.newaxis, :, :, :], verbose=0)[0]
indexes = np.argsort(predictions).tolist()
indexes.reverse()
indexes = indexes[:5]

items = np.column_stack((categories[indexes], np.round(predictions[indexes], decimals=3))).tolist()
print(json.dumps(items))






