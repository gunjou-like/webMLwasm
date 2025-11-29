"""
量子化エラーの原因を調査するスクリプト
"""
import onnx
from onnx import shape_inference
import tempfile
import os

def main():
    model_path = "models/resnet18.onnx"
    
    print("=" * 60)
    print("ONNX量子化エラー診断")
    print("=" * 60)
    
    # 1. モデル読み込み
    print("\n1. モデル読み込み中...")
    model = onnx.load(model_path)
    print(f"   ✅ 読み込み成功")
    
    # 2. Shape Inference テスト
    print("\n2. Shape Inference テスト...")
    try:
        inferred_model = shape_inference.infer_shapes(model)
        print(f"   ✅ Shape Inference 成功")
    except Exception as e:
        print(f"   ❌ Shape Inference 失敗: {e}")
        print("\n   原因: PyTorch 2.x の新しいONNXエクスポーターは")
        print("   一部の形状情報が不完全な場合があります。")
        return
    
    # 3. 外部データの確認
    print("\n3. 外部データファイル確認...")
    data_file = model_path + ".data"
    if os.path.exists(data_file):
        size_mb = os.path.getsize(data_file) / 1024 / 1024
        print(f"   ✅ {data_file} 存在 ({size_mb:.2f} MB)")
    else:
        print(f"   ⚠️ 外部データファイルなし")
    
    # 4. 量子化の代替手法を提案
    print("\n" + "=" * 60)
    print("推奨される解決策:")
    print("=" * 60)
    print("""
1. 【推奨】量子化をスキップして通常のONNXモデルを使用
   - 現在のスクリプトが自動でコピーを作成済み
   - ブラウザでも動作します（サイズは大きめ）

2. PyTorchの古いONNXエクスポーター（opset_version指定）を使用
   - torch.onnx.export に opset_version=13 を指定
   - より安定したONNX出力が得られる可能性

3. ONNXSimplifierで最適化後に量子化
   - onnx-simplifier で中間ノードを整理
   - その後に量子化を実行

現在の状態: resnet18.quant.onnx は通常モデルのコピーです（動作OK）
    """)

if __name__ == "__main__":
    main()
