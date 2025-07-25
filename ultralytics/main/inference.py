import onnx

from ultralytics import YOLO
import onnx
from onnxsim import simplify

# === 配置路径 ===
pt_path = r'D:/train13/weights/best.pt'                 # 你的 .pt 权重路径
onnx_path = r'D:/train13/weights/best.onnx'     # 导出 onnx 路径

model = YOLO(pt_path)

# === 导出为 ONNX（含动态 batch）===
model.export(
    format='onnx',
    dynamic=True,
    simplify=False,         
    opset=17,                 
    imgsz=(640, 640)
)
model = onnx.load(onnx_path)
input_tensor = model.graph.input[0]
dims = input_tensor.type.tensor_type.shape.dim
dims[0].dim_param = 'batch'  # 动态 batch
dims[2].dim_value = 640      # 固定 height
dims[3].dim_value = 640      # 固定 width
onnx.save(model, "best_dynamic_fix_hw.onnx")

for input_tensor in model.graph.input:
    shape = input_tensor.type.tensor_type.shape
    print(f"输入名: {input_tensor.name}")
    for i, dim in enumerate(shape.dim):
        if dim.HasField('dim_value'):
            print(f"  维度{i}: dim_value={dim.dim_value}")
        elif dim.HasField('dim_param'):
            print(f"  维度{i}: dim_param={dim.dim_param}")
        else:
            print(f"  维度{i}: 未设置dim_value或dim_param")