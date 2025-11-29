"""
Trace the ResNet18 with torch.jit.trace and export traced module to ONNX.
Saves to `models/resnet18_traced.onnx` and attempts to embed tensors.
"""
from pathlib import Path
import torch
import torchvision


def main():
    project_root = Path(__file__).resolve().parent.parent
    models_dir = project_root / 'models'
    models_dir.mkdir(parents=True, exist_ok=True)

    output_path = models_dir / 'resnet18_traced.onnx'

    print('Loading PyTorch ResNet18 pretrained...')
    model = torchvision.models.resnet18(pretrained=True)
    model.eval()

    dummy_input = torch.randn(1, 3, 224, 224)

    print('Tracing model with torch.jit.trace...')
    traced = torch.jit.trace(model, dummy_input)

    print(f'Exporting traced model to ONNX -> {output_path}')
    # Use legacy exporter for traced models (dynamo doesn't support ScriptModule)
    with torch.onnx.select_model_mode_for_export(traced, torch.onnx.TrainingMode.EVAL):
        torch.onnx.export(
            traced,
            dummy_input,
            str(output_path),
            export_params=True,
            opset_version=13,  # Use stable opset
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamo=False,  # Explicitly use legacy exporter
        )

    try:
        import onnx
        m = onnx.load(str(output_path))
        onnx.save_model(m, str(output_path), save_as_external_data=False)
        print('Saved traced ONNX with embedded tensors (no external data).')
    except Exception as e:
        print('Warning: could not re-save ONNX to embed tensors:', e)

    print('Done.')

if __name__ == '__main__':
    main()
