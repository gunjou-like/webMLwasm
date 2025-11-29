"""
resnet18.quant.onnxを単一ファイルとして再生成
"""
import onnx
import shutil
from pathlib import Path

def main():
    project_root = Path(__file__).resolve().parent.parent
    models_dir = project_root / "models"
    
    input_model = str(models_dir / "resnet18.onnx")
    output_model = str(models_dir / "resnet18.quant.onnx")
    
    print("=" * 60)
    print("単一ファイルモデルの生成")
    print("=" * 60)
    
    print("\n1. 元モデル読み込み...")
    # 外部データも含めて読み込む
    model = onnx.load(input_model, load_external_data=True)
    print(f"   ✅ 読み込み完了")
    
    # モデルのサイズを確認
    total_size = 0
    for init in model.graph.initializer:
        if init.data_type in [1, 10, 11]:  # float32, float16, double
            total_size += len(init.raw_data) if init.raw_data else 0
    
    print(f"   モデル重みサイズ: {total_size / 1024 / 1024:.2f} MB")
    
    print("\n2. 単一ファイルとして保存...")
    # 外部データなしで保存（すべて埋め込む）
    onnx.save(
        model, 
        output_model,
        save_as_external_data=False
    )
    print(f"   ✅ 保存完了: {output_model}")
    
    # ファイルサイズ確認
    import os
    output_size = os.path.getsize(output_model) / 1024 / 1024
    print(f"   ファイルサイズ: {output_size:.2f} MB")
    
    # 古い.dataファイルがあれば削除（混乱を避けるため）
    data_file = output_model + ".data"
    if os.path.exists(data_file):
        os.remove(data_file)
        print(f"   🗑️ 削除: {data_file}")
    
    print("\n" + "=" * 60)
    print("✅ 完了!")
    print("=" * 60)
    print(f"ブラウザで読み込み可能な単一ファイルモデル:")
    print(f"  {output_model} ({output_size:.2f} MB)")

if __name__ == "__main__":
    main()
