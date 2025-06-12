import pymysql
import os
import json
import shutil
import numpy as np
from PIL import Image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model

# ========== CONFIG ==========
LARAVEL_IMG_PATH = "C:/xampp/htdocs/duantn/DREAMS-BE/public/img/" # Thư mục ảnh trong Laravel
AI_IMG_PATH = "images/"                               # Thư mục ảnh trong Python AI project
DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "",
    "database": "duantn"
}
# ============================

# Kết nối MySQL
conn = pymysql.connect(**DB_CONFIG)
cursor = conn.cursor(pymysql.cursors.DictCursor)

# Load model ResNet50
model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

# Lấy dữ liệu ảnh và sản phẩm
cursor.execute("""
    SELECT img.name, img.product_id, products.name AS product_name
    FROM img
    JOIN products ON img.product_id = products.id
""")
rows = cursor.fetchall()

# Đảm bảo thư mục ảnh tồn tại
os.makedirs(AI_IMG_PATH, exist_ok=True)

features = []
metadata = []

for row in rows:
    src_path = os.path.join(LARAVEL_IMG_PATH, row['name'])
    dest_path = os.path.join(AI_IMG_PATH, row['name'])

    if not os.path.exists(src_path):
        print(f"⚠️ Không tìm thấy ảnh: {src_path}")
        continue

    # Copy ảnh từ Laravel sang Python
    shutil.copyfile(src_path, dest_path)

    try:
        img = Image.open(dest_path).convert("RGB").resize((224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        vec = model.predict(x)[0]

        features.append(vec)
        metadata.append({
            "product_id": row["product_id"],
            "product_name": row["product_name"],
            "image_path": row["name"]
        })

    except Exception as e:
        print(f"❌ Lỗi xử lý ảnh {row['name']}: {e}")

# Lưu kết quả
np.save("features.npy", np.array(features))
with open("metadata.json", "w", encoding='utf-8') as f:
    json.dump(metadata, f, ensure_ascii=False, indent=2)

print("✅ Đã tạo xong features.npy và metadata.json!")
