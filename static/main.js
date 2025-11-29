// ImageNetのクラスラベル (簡易版)
const IMAGENET_CLASSES = { 0: "Tench", 1: "Goldfish", /* ...省略、デモ用にIDだけ表示でもOK */ };

let wasmSession = null;

// 1. WASMセッションの初期化 (ページ読み込み時)
async function initWasm() {
    try {
        // Quantizedモデルを読み込む
        wasmSession = await ort.InferenceSession.create('/models/resnet18.quant.onnx', {
            executionProviders: ['wasm'] // WebAssembly指定
        });
        document.querySelector("#wasmResult").innerHTML = "✅ Model Loaded (Ready)";
    } catch (e) {
        console.error(e);
        document.querySelector("#wasmResult").innerHTML = "❌ Model Load Failed";
    }
}
initWasm();

// 画像アップロード処理
const inputElement = document.getElementById('uploadInput');
const previewElement = document.getElementById('preview');
const controlPanel = document.getElementById('controlPanel');

inputElement.addEventListener('change', (evt) => {
    const file = evt.target.files[0];
    if (file) {
        previewElement.src = URL.createObjectURL(file);
        previewElement.style.display = 'block';
        controlPanel.style.display = 'flex';
        // 結果クリア
        document.getElementById('serverResult').innerText = "";
        // WASMの方はモデルロード済みか確認して表示
        if(wasmSession) document.getElementById('wasmResult').innerText = "✅ Ready";
    }
});

// --- A. Server Side Inference ---
async function runServerInference() {
    const file = inputElement.files[0];
    const formData = new FormData();
    formData.append("file", file);

    const startTime = performance.now();
    const uiRes = document.getElementById('serverResult');
    uiRes.innerText = "Requesting...";

    try {
        const res = await fetch('/api/predict-server', {
            method: 'POST',
            body: formData
        });
        const data = await res.json();
        const endTime = performance.now();
        const totalLatency = (endTime - startTime).toFixed(2);

        uiRes.innerHTML = `
            ID: ${data.class_id}<br>
            Prob: ${data.probability.toFixed(4)}<br>
            <div class="latency">Total Latency: ${totalLatency} ms</div>
            <small>(Net: ${(totalLatency - data.latency_ms).toFixed(2)}ms + Inf: ${data.latency_ms.toFixed(2)}ms)</small>
        `;
    } catch (e) {
        uiRes.innerText = "Error";
    }
}

// --- B. Client Side (WASM) Inference ---
async function runWasmInference() {
    if (!wasmSession) { alert("Model not loaded yet"); return; }
    
    const uiRes = document.getElementById('wasmResult');
    uiRes.innerText = "Processing...";
    
    // 画像をCanvasに描画してピクセルデータ取得 -> Tensor変換
    // ※簡略化のため、画像リサイズ処理などの詳細はデモ用に最適化
    const tensor = await imageToTensor(previewElement);

    const startTime = performance.now();
    
    // 推論実行
    const feeds = { input: tensor }; // ONNXのinput nameに合わせる
    const results = await wasmSession.run(feeds);
    const output = results.output.data; // ONNXのoutput nameに合わせる

    const endTime = performance.now();
    const latency = (endTime - startTime).toFixed(2);

    // 最大値(argmax)を探す
    let maxProb = -1;
    let maxId = -1;
    for(let i=0; i<output.length; i++){
        if(output[i] > maxProb){
            maxProb = output[i];
            maxId = i;
        }
    }

    // Softmax簡易計算(デモ用)
    // 厳密にはここで行うが、スコア比較だけで十分なら省略可

    uiRes.innerHTML = `
        ID: ${maxId}<br>
        Score: ${maxProb.toFixed(4)}<br>
        <div class="latency">Latency: ${latency} ms</div>
        <small>(Network: 0 ms)</small>
    `;
}

// ユーティリティ: HTML Image -> ONNX Tensor (1, 3, 224, 224)
async function imageToTensor(imgElement) {
    const canvas = document.createElement('canvas');
    canvas.width = 224;
    canvas.height = 224;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(imgElement, 0, 0, 224, 224);
    
    const imgData = ctx.getImageData(0, 0, 224, 224).data;
    const float32Data = new Float32Array(1 * 3 * 224 * 224);
    
    // Normalization Constants (ImageNet)
    const mean = [0.485, 0.456, 0.406];
    const std = [0.229, 0.224, 0.225];

    for (let i = 0; i < 224 * 224; i++) {
        // R
        let r = imgData[i * 4 + 0] / 255.0;
        float32Data[i] = (r - mean[0]) / std[0];
        // G
        let g = imgData[i * 4 + 1] / 255.0;
        float32Data[i + 224 * 224] = (g - mean[1]) / std[1];
        // B
        let b = imgData[i * 4 + 2] / 255.0;
        float32Data[i + 224 * 224 * 2] = (b - mean[2]) / std[2];
    }
    
    return new ort.Tensor('float32', float32Data, [1, 3, 224, 224]);
}