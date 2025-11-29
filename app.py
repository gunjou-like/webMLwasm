from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import io
import time

app = FastAPI()

# CORS設定（念のため）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# サーバーサイド推論用のモデルロード (比較用: 遅いAPI)
model = torchvision.models.resnet18(pretrained=True)
model.eval()

# 画像前処理
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # クロップせずに強制リサイズ（JSに合わせる）
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

@app.post("/api/predict-server")
async def predict_server(file: UploadFile = File(...)):
    """サーバーサイドで推論を行うAPI (通信ラグあり)"""
    start_time = time.time()
    
    # 画像読み込み
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    
    # 前処理 & 推論
    input_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
    
    # 上位1件を取得 (デモ用なので簡易化)
    top1_prob, top1_id = torch.topk(probabilities, 1)
    
    end_time = time.time()
    
    return {
        "class_id": int(top1_id[0]),
        "probability": float(top1_prob[0]),
        "latency_ms": (end_time - start_time) * 1000,
        "mode": "Server-side (Python)"
    }

# 静的ファイル (HTML/JS/Model) の配信
# modelsディレクトリも配信して、ブラウザがfetchできるようにする
app.mount("/models", StaticFiles(directory="models"), name="models")
app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    # localhost:8000 で起動
    uvicorn.run(app, host="0.0.0.0", port=8000)