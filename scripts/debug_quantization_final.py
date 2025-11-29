"""
quantize_dynamicの内部処理を再現して、どこで失敗するか特定
"""
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
from onnxruntime.quantization.quant_utils import save_and_reload_model_with_shape_infer
import tempfile
import os

def main():
    model_path = "models/resnet18.onnx"
    
    print("=" * 70)
    print("quantize_dynamic内部処理の再現")
    print("=" * 70)
    
    # Step 1: モデル読み込み
    print("\n1. モデル読み込み...")
    model = onnx.load(model_path)
    print(f"   ✅ 成功")
    
    # Step 2: save_and_reload_model_with_shape_infer を実行
    # これがquantize_dynamicの最初のステップで失敗している
    print("\n2. save_and_reload_model_with_shape_infer を実行...")
    try:
        # 一時ファイルに保存
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_model = os.path.join(tmpdir, "temp.onnx")
            onnx.save(model, temp_model)
            
            # quantize_dynamicが内部で呼び出す関数
            inferred_model = save_and_reload_model_with_shape_infer(temp_model)
            
            print(f"   ✅ 成功")
            print(f"   推論されたモデルのノード数: {len(inferred_model.graph.node)}")
            
    except Exception as e:
        print(f"   ❌ 失敗")
        print(f"   エラー: {e}")
        print("\n【原因】")
        print("   quantize_dynamicは内部でモデルを一旦保存し、")
        print("   shape inferenceを実行してから再読み込みします。")
        print("   この過程で外部データファイル(.onnx.data)の扱いに")
        print("   問題が発生している可能性があります。")
        
        # 外部データの状態を確認
        print("\n【外部データファイルの確認】")
        data_file = model_path + ".data"
        if os.path.exists(data_file):
            size = os.path.getsize(data_file)
            print(f"   ファイル存在: {data_file}")
            print(f"   サイズ: {size / 1024 / 1024:.2f} MB")
            print("\n   問題: quantize_dynamicは外部データを持つモデルの")
            print("   処理に対応していない可能性があります。")
        
        print("\n【推奨する回避策】")
        print("   1. 外部データを埋め込んだモデルを作成")
        print("   2. より小さなモデルサイズにする")
        print("   3. 量子化をスキップ（現在の対応）")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        print("\n" + "=" * 70)
        print("【結論】")
        print("=" * 70)
        print("""
量子化が失敗する根本原因:
- PyTorch 2.xのONNXエクスポーターは大きなモデルを外部データ形式で出力
- onnxruntime.quantization.quantize_dynamic()は外部データ形式のモデルを
  内部で再保存・再読み込みする際にshape inferenceでエラーを起こす
- エラーメッセージ: "Inferred shape and existing shape differ in dimension 0"

対応策:
✅ 現在の実装: 量子化をスキップしてオリジナルをコピー（動作OK）
- ブラウザでも問題なく動作します
- サイズは大きいですが、機能としては完全です
        """)
