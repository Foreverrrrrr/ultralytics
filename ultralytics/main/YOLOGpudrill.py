import os
from ultralytics import YOLO
from PIL import Image
import torch

train_folder = r'D:\AI\Ai\yolo\dataset\images\train'
val_folder = r'D:\AI\Ai\yolo\dataset\images\val'
yaml_path = r'D:\AI\Ai\yolo\dataset\data.yaml'

def process_images(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            with Image.open(file_path) as img:
                if img.format == 'GIF':
                    print(f"发现 GIF 图像: {file_path}，正在转换...")
                    new_file_path = os.path.splitext(file_path)[0] + '.jpg'
                    img = img.convert('RGB')
                    img.save(new_file_path, 'JPG')
                    os.remove(file_path)
                    print(f"已将 {file_path} 转换为 {new_file_path}")
        except Exception as e:
            print(f"处理 {file_path} 时出现错误: {e}")

class GradChecker:
    def __init__(self, model, threshold=1e4):
        self.model = model
        self.threshold = threshold
        self.hooks = []
        self.register_hooks()

    def register_hooks(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                hook = param.register_hook(self._make_hook(name))
                self.hooks.append(hook)

    def _make_hook(self, name):
        def hook_fn(grad):
            if torch.isnan(grad).any():
                print(f"🚨 梯度检测NAN: {name}")
            if torch.isinf(grad).any():
                print(f"🚨 梯度检测INF: {name}")
            if grad.abs().max() > self.threshold:
                print(f"🚨 Gradient too large in: {name}, max grad: {grad.abs().max().item()}")
        return hook_fn
    
    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

def main():
    model = YOLO(r"ultralytics\cfg\models\11\FRFN.yaml").load("yolo11n.pt")
    if os.path.exists(yaml_path):
        print(f"文件 {yaml_path} 存在。")
    else:
        print(f"文件 {yaml_path} 不存在，请检查路径和文件是否正确。")
        return
    grad_checker = GradChecker(model.model, threshold=1e3) 
    results = model.train(
        data=yaml_path,
        multi_scale=False,
        epochs=1,
        imgsz=640,
        batch=8,
        lr0=0.001,  # 初始学习率
        lrf=0.00001,  # 最终学习率
        cos_lr=True,  # 余弦调度器
        augment=True,  # 数据增强
        degrees=120,  # 随机旋转
        mosaic=True,
        close_mosaic=100,
    )
    grad_checker.remove_hooks()
    model.save('custom_yolo11_model.pt')
    onnx_path = model.export(format='onnx')
    print(f"模型已保存为 ONNX 格式，路径为: {onnx_path}")
    trt_path = model.export(format='engine')
    print(f"模型已保存为 TensorRT 格式，路径为: {trt_path}")
    openvino_path = model.export(format='openvino')
    print(f"模型已保存为 OpenVINO 格式，路径为: {openvino_path}")

if __name__ == '__main__':
    process_images(train_folder)
    process_images(val_folder)
    main()
