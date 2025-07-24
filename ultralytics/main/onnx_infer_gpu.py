import onnxruntime as ort
import numpy as np
import os
from PIL import Image
import cv2

# 打印 onnxruntime 版本和 CUDA 是否可用
print('onnxruntime version:', ort.__version__)
print('CUDAExecutionProvider' in ort.get_available_providers() and 'CUDA 可用' or 'CUDA 不可用')
def check_cuda_dll():
    for p in os.environ['PATH'].split(';'):
        if os.path.isdir(p) and os.path.exists(p):
            try:
                if any('cudart' in f.lower() for f in os.listdir(p)):
                    return True
            except Exception:
                continue
    return False
print('CUDA DLL in PATH:', check_cuda_dll())

def check_cudnn_dll():
    for p in os.environ['PATH'].split(';'):
        if os.path.isdir(p) and os.path.exists(p):
            try:
                if any('cudnn' in f.lower() for f in os.listdir(p)):
                    return True
            except Exception:
                continue
    return False

print('cuDNN DLL in PATH:', check_cudnn_dll())

# 测试ONNX模型加载和推理
def test_onnx_model(model_path, img_path,image):
    import time
    try:
        print(f'加载模型: {model_path}')
        session = ort.InferenceSession(model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        print('模型加载成功')
        img = Image.open(img_path).convert('RGB').resize((640, 640))
        img_np = np.array(img).astype(np.float32) / 255.0
        img_np = np.transpose(img_np, (2, 0, 1))[None]
        input_name = session.get_inputs()[0].name
        print('连续推理10次，统计每次推理耗时:')
        times = []
        for i in range(1):
            start = time.time()
            outputs = session.run(None, {input_name: img_np})
            end = time.time()
            t = (end - start) * 1000  # ms
            times.append(t)
            print(f'第{i+1}次推理耗时: {t:.2f} ms, 输出shape: {[o.shape for o in outputs]}')
        print(f'平均推理耗时: {np.mean(times):.2f} ms')

        img = Image.open(image).convert('RGB').resize((640, 640))
        img_np = np.array(img).astype(np.float32) / 255.0
        img_np = np.transpose(img_np, (2, 0, 1))[None]
        input_name = session.get_inputs()[0].name
        print('连续推理10次，统计每次推理耗时:')
        times = []
        for i in range(20):
            start = time.time()
            outputs = session.run(None, {input_name: img_np})
            end = time.time()
            t = (end - start) * 1000  # ms
            times.append(t)
            print(f'第{i+1}次推理耗时: {t:.2f} ms, 输出shape: {[o.shape for o in outputs]}')
        print(f'平均推理耗时: {np.mean(times):.2f} ms')

    except Exception as e:
        print('模型加载或推理报错:', e)



if __name__ == '__main__':
    test_onnx_model(r'D:\train13\weights\best.onnx', r'D:\AI\Ai\demo711\cut\random_0_2e05b7d4.jpg', r'D:\AI\Ai\demo712\cut1\crop_0_0_b3e23b8d.jpg')