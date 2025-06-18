from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image as keras_image
import numpy as np
import faiss
import json
from PIL import Image
import io

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def check():
    return {"message": "OK"}

model = ResNet50(weights="imagenet", include_top=False, pooling="avg")
features = np.load("features.npy")
with open("metadata.json", "r", encoding="utf-8") as f:
    metadata = json.load(f)

index = faiss.IndexFlatL2(features.shape[1])
index.add(features)

@app.post("/search")
async def search_image(image: UploadFile = File(...)):
    contents = await image.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB").resize((224, 224))
    x = keras_image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    vec = model.predict(x)[0]

    D, I = index.search(np.array([vec]), 5)

    results = []
    for idx, distance in zip(I[0], D[0]):
        item = metadata[idx]
        item["score"] = float(distance)
        results.append(item)

    return results
#thêm chức năng thử đồ ảo
@app.post("/tryon")
async def tryon(
    photo: UploadFile = File(...),
    product_image: UploadFile = File(...)
):
    # Đọc ảnh người dùng
    user_img = Image.open(photo.file).convert("RGBA")
    # Đọc ảnh sản phẩm (ví dụ: áo, kính...)
    prod_img = Image.open(product_image.file).convert("RGBA")

    # Resize sản phẩm cho phù hợp (ví dụ: overlay lên góc trái)
    prod_img = prod_img.resize((int(user_img.width/2), int(user_img.height/2)))
    user_img.paste(prod_img, (0, 0), prod_img)

    # Lưu vào buffer
    buf = io.BytesIO()
    user_img.convert("RGB").save(buf, format='JPEG')
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/jpeg")