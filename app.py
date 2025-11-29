from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import io
import time
import logging

app = FastAPI()

# ログ設定
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')

# PyTorchスレッドを制限してスレッドオーバーヘッドを抑える
try:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    logging.info(f"torch threads set: intra={torch.get_num_threads()}, inter={torch.get_num_interop_threads()}")
except Exception:
    logging.warning("Could not set torch threads")

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

def _warmup_model(model, runs=5):
    """モデルのウォームアップを行い、初回レイテンシやキャッシュを温める"""
    logging.info("Starting model warm-up...")
    dummy = torch.randn(1, 3, 224, 224)
    # 軽めに数回実行
    with torch.no_grad():
        for i in range(runs):
            t0 = time.time()
            _ = model(dummy)
            t1 = time.time()
            logging.info(f" Warmup {i+1}/{runs}: {(t1-t0)*1000:.2f} ms")
    logging.info("Warm-up complete")

# 画像前処理
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # クロップせずに強制リサイズ（JSに合わせる）
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

@app.post("/api/predict-server")
async def predict_server(file: UploadFile = File(...)):
    """サーバーサイドで推論を行うAPI (通信ラグあり)"""
    req_start = time.time()

    # 画像読み込み
    t0 = time.time()
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    t1 = time.time()

    # 前処理
    p0 = time.time()
    input_tensor = transform(image).unsqueeze(0)
    p1 = time.time()

    # 推論
    inf0 = time.time()
    with torch.no_grad():
        output = model(input_tensor)
    inf1 = time.time()

    # 結果処理
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    top1_prob, top1_id = torch.topk(probabilities, 1)

    req_end = time.time()

    preprocess_ms = (p1 - p0) * 1000
    inference_ms = (inf1 - inf0) * 1000
    total_ms = (req_end - req_start) * 1000

    logging.info(f"Request processed: preprocess={preprocess_ms:.2f}ms inference={inference_ms:.2f}ms total={total_ms:.2f}ms")

    return {
        "class_id": int(top1_id[0]),
        "probability": float(top1_prob[0]),
        "latency_ms": total_ms,
        "preprocess_ms": preprocess_ms,
        "inference_ms": inference_ms,
        "mode": "Server-side (Python)"
    }

# 静的ファイル (HTML/JS/Model) の配信
# modelsディレクトリも配信して、ブラウザがfetchできるようにする
app.mount("/models", StaticFiles(directory="models"), name="models")
app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    # 起動前にウォームアップ
    try:
        _warmup_model(model, runs=5)
    except Exception as e:
        logging.warning(f"Warm-up failed: {e}")

    # localhost:8000 で起動
    uvicorn.run(app, host="0.0.0.0", port=8000)