from ultralytics import YOLO
import torch
import tensorrt as trt

print(trt.__version__)  
model = YOLO(r"D:\train13\weights\best.pt")

model.export(
    format="engine",  # 导出为 TensorRT
    device=0,        # 使用 GPU 0
    half=True,       # FP16 量化 (提升速度)
    workspace=4,     # GPU 工作空间大小 (GB)
    simplify=True,   # 简化 ONNX 图
    opset=12         # ONNX 算子集版本
)


def validate_engine(engine_path):
    logger = trt.Logger(trt.Logger.WARNING)
    with open(engine_path, "rb") as f, trt.Runtime(logger) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
        print(f"引擎验证成功! 输入维度: {engine[0].shape}")

validate_engine("best.engine")