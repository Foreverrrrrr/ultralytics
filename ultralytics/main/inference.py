import onnx
from ultralytics import YOLO
import onnx
from onnxsim import simplify

# === 配置路径 ===
pt_path = r'D:\AI\code\ultralytics\runs\detect\train16\weights\best.pt'                 # 你的 .pt 权重路径
onnx_path = r'D:\AI\code\ultralytics\runs\detect\train16\weights\best.onnx'     # 导出 onnx 路径

model = YOLO(pt_path)

# === 导出为 ONNX（含动态 batch）===
model.export(
    format='onnx',
    dynamic=True,
    simplify=False,         
    opset=17,                 
    imgsz=(640, 640)
)

onnx_model = onnx.load(onnx_path)
for input_tensor in onnx_model.graph.input:
    shape = input_tensor.type.tensor_type.shape
    print(f"输入名: {input_tensor.name}")
    for i, dim in enumerate(shape.dim):
        if dim.HasField('dim_value'):
            print(f"  维度{i}: dim_value={dim.dim_value}")
        elif dim.HasField('dim_param'):
            print(f"  维度{i}: dim_param={dim.dim_param}")
        else:
            print(f"  维度{i}: 未设置dim_value或dim_param")

# 输出输出张量信息
for output_tensor in onnx_model.graph.output:
    shape = output_tensor.type.tensor_type.shape
    print(f"输出名: {output_tensor.name}")
    for i, dim in enumerate(shape.dim):
        if dim.HasField('dim_value'):
            print(f"  维度{i}: dim_value={dim.dim_value}")
        elif dim.HasField('dim_param'):
            print(f"  维度{i}: dim_param={dim.dim_param}")
        else:
            print(f"  维度{i}: 未设置dim_value或dim_param")
