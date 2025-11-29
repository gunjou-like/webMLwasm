"""
量子化時のShape Inferenceエラーを詳細調査
"""
import onnx
from onnx import shape_inference, numpy_helper
import tempfile
import os

def main():
    model_path = "models/resnet18.onnx"
    
    print("=" * 70)
    print("詳細なShape Inference調査")
    print("=" * 70)
    
    # モデル読み込み
    model = onnx.load(model_path)
    
    # すべてのテンソルの形状を確認
    print("\n【Value Info (中間テンソル)】")
    for i, vi in enumerate(model.graph.value_info[:5]):  # 最初の5つだけ表示
        print(f"  {i+1}. {vi.name}: {vi.type.tensor_type.shape}")
    
    if len(model.graph.value_info) > 5:
        print(f"  ... (残り {len(model.graph.value_info) - 5} 個)")
    
    # Initializerの確認
    print(f"\n【Initializer (重み)】")
    print(f"  総数: {len(model.graph.initializer)}")
    for init in model.graph.initializer[:3]:
        print(f"  - {init.name}: shape={init.dims}")
    
    # 問題のあるノードを探す
    print(f"\n【ノード解析】")
    print(f"  総ノード数: {len(model.graph.node)}")
    
    # Shape Inference を path経由で実行（quantize_dynamicが内部で行う処理）
    print(f"\n【Shape Inference (Path方式) - quantize_dynamicが使う方法】")
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_model = os.path.join(tmpdir, "temp.onnx")
            inferred_model = os.path.join(tmpdir, "inferred.onnx")
            
            # 一旦保存
            onnx.save(model, temp_model)
            
            # Path経由でShape Inference
            onnx.shape_inference.infer_shapes_path(temp_model, inferred_model)
            
            print("  ✅ Path方式のShape Inference成功")
            
            # 推論されたモデルを読み込んで確認
            inferred = onnx.load(inferred_model)
            print(f"  推論後のvalue_info数: {len(inferred.graph.value_info)}")
            
    except Exception as e:
        print(f"  ❌ Path方式のShape Inference失敗")
        print(f"  エラー: {e}")
        print("\n【原因分析】")
        print("  これが量子化失敗の直接の原因です。")
        print("  PyTorch 2.xの新しいONNXエクスポーターが生成するモデルには")
        print("  shape inferenceで推論できない形状情報が含まれています。")
        print("\n  具体的には:")
        print("  - 出力層の形状: (512) vs (1000) のミスマッチ")
        print("  - これはResNet18の最終層(fc層)の重みと出力の不整合")
        
        # より詳細なエラー情報を表示
        print("\n【解決策】")
        print("  オプション1: opset_versionを指定して古い方式でエクスポート")
        print("  オプション2: 量子化をスキップ（現在の対応）")
        print("  オプション3: onnxsimで最適化してから量子化")

if __name__ == "__main__":
    main()
