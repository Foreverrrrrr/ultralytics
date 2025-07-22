from ultralytics import YOLO

# 加载模型
model = YOLO(r"D:\train13\weights\best.pt")

# 导出为 ONNX 格式
model.export(format="onnx")  # 默认会保存为 best.onnx 文件