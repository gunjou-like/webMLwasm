"""
Re-export ResNet18 to ONNX with fixed input size and opset_version=18.
Saves to `models/resnet18_reexport.onnx` and embeds initializers (no external data).
"""
from pathlib import Path
import torch
import torchvision


def main():
    project_root = Path(__file__).resolve().parent.parent
    models_dir = project_root / 'models'
    models_dir.mkdir(parents=True, exist_ok=True)

    output_path = models_dir / 'resnet18_reexport.onnx'

    print('Loading PyTorch ResNet18 pretrained...')
    model = torchvision.models.resnet18(pretrained=True)
    model.eval()

    dummy_input = torch.randn(1, 3, 224, 224)

    print(f'Exporting to ONNX (opset=18) -> {output_path}')
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        # fixed input; do not pass dynamic_axes
    )

    # Ensure model has no external data
    try:
        import onnx
        m = onnx.load(str(output_path))
        onnx.save_model(m, str(output_path), save_as_external_data=False)
        print('Saved ONNX with embedded tensors (no external data).')
    except Exception as e:
        print('Warning: could not re-save ONNX to embed tensors:', e)

    print('Done.')

if __name__ == '__main__':
    main()
