"""
ブラウザからのモデルロードをテストするスクリプト
"""
import onnx
import os

def check_model(model_path):
    print(f"\n{'='*60}")
    print(f"モデル: {model_path}")
    print(f"{'='*60}")
    
    if not os.path.exists(model_path):
        print("❌ ファイルが存在しません")
        return False
    
    # ファイルサイズ
    size_mb = os.path.getsize(model_path) / 1024 / 1024
    print(f"ファイルサイズ: {size_mb:.2f} MB")
    
    # 外部データの確認
    data_file = model_path + ".data"
    if os.path.exists(data_file):
        data_size_mb = os.path.getsize(data_file) / 1024 / 1024
        print(f"外部データ: {data_file} ({data_size_mb:.2f} MB)")
    else:
        print(f"外部データ: なし")
    
    # ONNXとして読み込めるか
    try:
        model = onnx.load(model_path)
        print(f"✅ ONNX読み込み: 成功")
        
        # 外部データの参照チェック
        has_external_data = any(
            init.external_data for init in model.graph.initializer 
            if hasattr(init, 'external_data') and init.external_data
        )
        
        if has_external_data:
            print(f"⚠️ 外部データ参照: あり")
            print(f"   → ブラウザから読み込む場合、.dataファイルも配信が必要")
        else:
            print(f"✅ 外部データ参照: なし（単一ファイル）")
            
        return True
        
    except Exception as e:
        print(f"❌ ONNX読み込み: 失敗")
        print(f"   エラー: {e}")
        return False

def main():
    print("=" * 60)
    print("モデルファイル診断")
    print("=" * 60)
    
    models = [
        "models/resnet18.onnx",
        "models/resnet18.quant.onnx"
    ]
    
    for model_path in models:
        success = check_model(model_path)
    
    print("\n" + "=" * 60)
    print("【結論】")
    print("=" * 60)
    print("""
問題: resnet18.quant.onnx は resnet18.onnx のコピーですが、
     外部データ(.onnx.data)を参照しています。
     
解決策:
1. resnet18.quant.onnx を単一ファイルに変換
2. main.js を修正して resnet18.onnx を直接使用
    """)

if __name__ == "__main__":
    main()
