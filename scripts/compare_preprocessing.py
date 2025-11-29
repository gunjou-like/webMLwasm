"""
Compare preprocessing and model outputs between server (PyTorch) and ONNX Runtime.
Usage:
  python scripts/compare_preprocessing.py /path/to/image.jpg

This script:
 - Loads `models/resnet18.onnx` with onnxruntime
 - Loads PyTorch ResNet18 and uses the same `transform` as in `app.py`
 - Applies JS-style preprocessing (canvas-style) for ONNX path
 - Compares top-1 id/prob and prints L2 difference between output logits/probs
"""
import sys
import numpy as np
import onnxruntime as ort
import onnx
from PIL import Image
import torch
import torchvision.transforms as transforms
import torchvision
import os
from pathlib import Path
from PIL import Image
import numpy as np


def js_preprocess(img: Image.Image):
    # Resize to 224x224 (canvas drawImage forced resize)
    img = img.convert('RGB').resize((224, 224), Image.BILINEAR)
    arr = np.asarray(img).astype(np.float32) / 255.0  # H x W x C, values 0..1
    # Convert to C x H x W with ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    # arr shape H,W,3 -> transpose to 3,H,W
    arr = (arr - mean) / std
    arr = np.transpose(arr, (2,0,1)).astype(np.float32)
    return arr[np.newaxis, :]


def pytorch_preprocess(img: Image.Image):
    # Use same transform as app.py
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    t = transform(img)
    return t.unsqueeze(0)


def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/compare_preprocessing.py /path/to/image.jpg")
        return
    path = sys.argv[1]
    img = Image.open(path).convert('RGB')
    out_dir = Path('tmp')
    out_dir.mkdir(exist_ok=True)

    # PyTorch path
    pt_input = pytorch_preprocess(img)
    # Save PyTorch preprocessed image for inspection (C,H,W -> H,W,C)
    pt_np = pt_input[0].numpy()
    # denormalize for visualization
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3,1,1)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3,1,1)
    vis = (pt_np * std + mean)
    vis = np.clip(np.transpose(vis, (1,2,0)) * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(vis).save(out_dir / 'pt_preprocessed.png')
    print('\n[DEBUG] PyTorch input stats: shape=', pt_input.shape, 'min=', pt_input.min().item(), 'max=', pt_input.max().item(), 'mean=', pt_input.mean().item())
    print('[DEBUG] PyTorch input first elems:', pt_input.flatten()[0:16].numpy())
    model = torchvision.models.resnet18(pretrained=True)
    model.eval()
    with torch.no_grad():
        pt_out = model(pt_input)[0].numpy()
    pt_probs = softmax(pt_out)
    pt_top = int(np.argmax(pt_probs))

    # ONNX path using JS-style preprocessing
    onnx_sess = ort.InferenceSession('models/resnet18.onnx')
    onnx_input = js_preprocess(img)
    # Save ONNX (JS-style) preprocessed image for inspection
    onnx_vis = onnx_input[0]
    # onnx_input is C,H,W normalized -> denormalize
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3,1,1)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3,1,1)
    onnx_img = (onnx_vis * std + mean)
    onnx_img = np.clip(np.transpose(onnx_img, (1,2,0)) * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(onnx_img).save(out_dir / 'onnx_preprocessed.png')
    print('\n[DEBUG] ONNX(JS) input stats: shape=', onnx_input.shape, 'min=', onnx_input.min(), 'max=', onnx_input.max(), 'mean=', onnx_input.mean())
    print('[DEBUG] ONNX(JS) input first elems:', onnx_input.flatten()[0:16])
    input_name = onnx_sess.get_inputs()[0].name
    res = onnx_sess.run(None, {input_name: onnx_input})
    onnx_out = np.array(res[0][0])
    onnx_probs = softmax(onnx_out)
    onnx_top = int(np.argmax(onnx_probs))

    # Debug: show logits head and differences
    print('\n[DEBUG] PyTorch logits first 32:', np.round(pt_out[:32], 4))
    print('[DEBUG] ONNX logits first 32:   ', np.round(onnx_out[:32], 4))
    diff_logits = pt_out - onnx_out
    print('[DEBUG] logits diff first 32:  ', np.round(diff_logits[:32], 4))
    print('[DEBUG] L2 norm logits diff:', np.linalg.norm(diff_logits))

    print('PyTorch top1:', pt_top, f'prob={pt_probs[pt_top]:.6f}')
    print('ONNX   top1:', onnx_top, f'prob={onnx_probs[onnx_top]:.6f}')
    print('L2 diff between logits:', np.linalg.norm(pt_out - onnx_out))
    print('L2 diff between probs :', np.linalg.norm(pt_probs - onnx_probs))

if __name__ == '__main__':
    main()
