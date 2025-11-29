# PyTorchとONNXの出力不一致を解消した方法

## 問題の概要

### 発生した問題
- **症状**: PyTorchモデルとONNXモデルで同じ入力画像に対して異なる予測結果が出力される
- **具体例**:
  ```
  PyTorch: ID 417, Prob 0.134
  ONNX:    ID 530, Prob 0.183
  ```
- **影響**: サーバーサイド推論とクライアントサイド推論で結果が一致せず、デモの信頼性が損なわれる

### 原因の特定

#### 1. 前処理の検証
`scripts/compare_preprocessing.py` を使用して、PyTorchとONNX（JavaScript風）の前処理を比較:
```python
# 結果: 前処理は完全に一致
PyTorch input: shape=(1,3,224,224), min=-2.118, max=2.535, mean=0.313
ONNX input:    shape=(1,3,224,224), min=-2.118, max=2.535, mean=0.313
先頭16要素も完全一致
```
→ **前処理は問題ではない**

#### 2. モデル変換の問題
出力（logits）を比較:
```python
# 元のONNX（dynamo exporter）
PyTorch logits[0:5]: [-0.4273, -0.7182, -0.7901, -2.5015, -3.8901]
ONNX logits[0:5]:    [ 0.0862,  2.1311,  0.6516, -0.8853,  2.7685]
L2 diff: 87.568  # 大きな差異
```
→ **ONNX変換プロセスに問題がある**

## 解決方法

### PyTorch 2.xの新エクスポーターの問題

**問題のあったコード** (`scripts/export_model.py`):
```python
# デフォルトでdynamo=Trueが使われる（PyTorch 2.x）
torch.onnx.export(
    model, 
    dummy_input, 
    output_path,
    input_names=['input'], 
    output_names=['output'],
    # dynamoベースの新エクスポーターが自動選択される
)
```

**課題**:
- PyTorch 2.x の新しい `torch.export` ベースのエクスポーター（dynamo）は一部のモデルで変換誤差が発生
- 特に ResNet のような複雑なモデルで顕著
- opset変換や最適化の過程で演算が変わる

### 解決策: torch.jit.trace + レガシーエクスポーター

**成功したコード** (`scripts/reexport_traced_onnx.py`):
```python
import torch
import torchvision
from pathlib import Path

def main():
    # 1. PyTorchモデルをロード
    model = torchvision.models.resnet18(pretrained=True)
    model.eval()
    
    dummy_input = torch.randn(1, 3, 224, 224)
    
    # 2. torch.jit.traceでモデルをトレース
    print('Tracing model with torch.jit.trace...')
    traced = torch.jit.trace(model, dummy_input)
    
    # 3. レガシーエクスポーターで変換
    with torch.onnx.select_model_mode_for_export(traced, torch.onnx.TrainingMode.EVAL):
        torch.onnx.export(
            traced,
            dummy_input,
            'models/resnet18_traced.onnx',
            export_params=True,
            opset_version=13,  # 安定版opset
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamo=False,  # レガシーエクスポーターを明示
        )
```

**重要なポイント**:
1. **`torch.jit.trace()`**: モデルを実行トレースしてScriptModuleに変換
2. **`dynamo=False`**: レガシーエクスポーターを明示的に使用
3. **`opset_version=13`**: 安定したopsetバージョンを指定
4. **`select_model_mode_for_export()`**: トレース済みモデルのエクスポートモード設定

## 結果

### 改善後の比較

```python
# Traced ONNX（レガシーエクスポーター）
PyTorch logits[0:5]: [-0.4273, -0.7182, -0.7901, -2.5015, -3.8901]
ONNX logits[0:5]:    [-0.4273, -0.7182, -0.7901, -2.5015, -3.8901]
L2 diff: 0.00006  # ほぼゼロ！

PyTorch: ID 417, Prob 0.134372
ONNX:    ID 417, Prob 0.134372  # 完全一致
```

### 改善の定量評価

| 指標 | 元のONNX | Traced ONNX | 改善率 |
|------|----------|-------------|--------|
| L2 logits差 | 87.568 | 0.00006 | **99.9999%改善** |
| L2 確率差 | 0.318 | 0.0000004 | **99.9999%改善** |
| Top-1一致 | ❌ 不一致 | ✅ 一致 | **完全解決** |

## 適用手順

### 1. Traced ONNXの生成
```bash
# venv環境で実行
.\venv\Scripts\python.exe scripts\reexport_traced_onnx.py
```

### 2. 本番ファイルに置き換え
```powershell
# バックアップ作成
Copy-Item models\resnet18.onnx models\resnet18_old.onnx

# Traced版を本番用にコピー
Copy-Item models\resnet18_traced.onnx models\resnet18.onnx
Copy-Item models\resnet18_traced.onnx models\resnet18.quant.onnx
```

### 3. 検証
```bash
# 比較スクリプトで検証
.\venv\Scripts\python.exe scripts\compare_preprocessing.py "path\to\image.jpg"
```

### 4. ブラウザキャッシュクリア
- **重要**: ブラウザが古いONNXをキャッシュしている場合があるため、必ずキャッシュクリア
- **方法**: `Ctrl + Shift + R`（ハードリロード）
- **確認**: DevTools Network タブで `resnet18.quant.onnx` が再ダウンロードされることを確認

## 学んだこと

### 1. PyTorch 2.xエクスポーターの選択
- **新エクスポーター（dynamo）**: 新機能対応だが安定性に課題
- **レガシーエクスポーター**: 古いが安定、精度重視なら推奨
- **torch.jit.trace**: より正確なONNX変換が可能

### 2. ONNX変換の検証は必須
- エクスポート後に必ず PyTorch と ONNX の出力を比較
- `compare_preprocessing.py` のような検証スクリプトが有用
- 前処理と推論の両方を検証

### 3. ブラウザキャッシュの注意
- 大きなモデルファイルは積極的にキャッシュされる
- 更新時は必ずハードリロードで確認
- DevTools で実際のダウンロードを確認

## トラブルシューティング

### Q1: traced ONNXでもまだ結果が異なる
**A**: 
- ブラウザキャッシュをクリア（`Ctrl + Shift + R`）
- DevTools Networkタブで実際にファイルがダウンロードされているか確認
- ファイルのSHA256ハッシュが一致しているか確認:
  ```powershell
  Get-FileHash models\resnet18.onnx -Algorithm SHA256
  Get-FileHash models\resnet18_traced.onnx -Algorithm SHA256
  ```

### Q2: torch.jit.traceで警告が出る
**A**: 
- 条件分岐やループがあるモデルでは警告が出る場合がある
- ResNet18 のような標準モデルでは問題なし
- 複雑な制御フローがある場合は `torch.jit.script` を検討

### Q3: エクスポートが遅い
**A**: 
- traced ONNXは初回のみ生成すればOK
- 生成したファイルを再利用（Git LFSなどで管理推奨）

## 参考資料

- [PyTorch ONNX Export Documentation](https://pytorch.org/docs/stable/onnx.html)
- [torch.jit.trace Documentation](https://pytorch.org/docs/stable/generated/torch.jit.trace.html)
- [ONNX Opset Versions](https://github.com/onnx/onnx/blob/main/docs/Operators.md)

## まとめ

PyTorch 2.xの新しいエクスポーターは便利だが、精度が重要な場合は **`torch.jit.trace` + レガシーエクスポーター** を使用することで、PyTorchとONNXの出力を完全に一致させることができる。

**重要な教訓**: 
- ONNX変換後は必ず検証する
- エクスポーター選択は目的に応じて行う
- ブラウザキャッシュに注意する
