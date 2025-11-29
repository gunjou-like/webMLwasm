import torch
import torchvision
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
from pathlib import Path

def main():
    print("1. PyTorchモデル(ResNet18)をダウンロード中...")
    model = torchvision.models.resnet18(pretrained=True)
    model.eval()

    # ダミー入力（モデルの入力サイズ定義用）
    dummy_input = torch.randn(1, 3, 224, 224)
    
    # Resolve paths relative to the project root (one level up from `scripts/`)
    project_root = Path(__file__).resolve().parent.parent
    models_dir = project_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    output_path = str(models_dir / "resnet18.onnx")
    quant_output_path = str(models_dir / "resnet18.quant.onnx")

    print("2. ONNXへ変換中...")
    # PyTorch 2.xの新しいエクスポーターは量子化と相性が悪いため、
    # opset_versionを明示して安定したONNXを生成
    torch.onnx.export(
        model, 
        dummy_input, 
        output_path, 
        export_params=True,
        opset_version=13,  # 安定版opset
        do_constant_folding=True,
        input_names=['input'], 
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print("   ✅ ONNX変換完了")

    print("3. 量子化(Quantization)を実行中... (サイズ削減)")
    try:
        quantize_dynamic(
            output_path,
            quant_output_path,
            weight_type=QuantType.QUInt8
        )
        print(f"✅ 量子化成功!")
    except Exception as e:
        print(f"⚠️ 量子化失敗: {e}")
        print(f"代替案: ブラウザ用に単一ファイルモデルを生成します...")
        
        # 外部データを含めて読み込み、単一ファイルとして保存
        model_onnx = onnx.load(output_path, load_external_data=True)
        onnx.save(model_onnx, quant_output_path, save_as_external_data=False)
        print(f"✅ 単一ファイル化完了 (量子化なし)")
    
    print(f"\n完了! モデルはこちらに保存されました:\n - オリジナル: {output_path}\n - ブラウザ用(推奨): {quant_output_path}")

if __name__ == "__main__":
    main()