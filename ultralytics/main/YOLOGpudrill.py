import os
from ultralytics import YOLO
from PIL import Image
import torch

train_folder = r'C:\Users\Forever\Desktop\data integration\metal\QRDATA\QRcode\images'
val_folder = r'C:\Users\Forever\Desktop\data integration\metal\QRDATA\QRcode\images\val'
yaml_path = r'C:\Users\Forever\Desktop\data integration\metal\QRDATA\QRcode\data.yaml'
model_cfg = r"ultralytics/cfg/models/11/FRFN.yaml"
pretrained = "yolo11s.pt"
imgsz = 640
epochs = 150
batch = 8

def process_images(folder):
    print(f"正在处理图片文件夹: {folder}")
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            with Image.open(file_path) as img:
                if img.format in ['GIF', 'PNG', 'BMP']:
                    print(f"发现 {img.format} 图像: {file_path}，正在转换...")
                    new_file_path = os.path.splitext(file_path)[0] + '.jpg'
                    img = img.convert('RGB')
                    img.save(new_file_path, 'JPEG')
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
    print("========== YOLO 训练启动 ==========")
    if not os.path.exists(yaml_path):
        print(f"❌ 数据文件 {yaml_path} 不存在，请检查路径和文件是否正确。")
        return
        
    if not os.path.exists(model_cfg):
        print(f"❌ 模型配置文件 {model_cfg} 不存在，请检查路径和文件是否正确。")
        return
    
    print("✅ 配置文件检查通过")
    print(f"📁 数据配置: {yaml_path}")
    print(f"🏗️ 模型配置: {model_cfg}")
    print(f"🎯 预训练权重: {pretrained}")
    try:
        print("🔄 加载模型配置和预训练权重...")
        model = YOLO(model_cfg).load(pretrained)
        print("✅ 模型加载成功")
        print(f"📊 模型参数统计:")
        total_params = sum(p.numel() for p in model.model.parameters())
        trainable_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
        print(f"   总参数数: {total_params:,}")
        print(f"   可训练参数数: {trainable_params:,}")
        
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    grad_checker = GradChecker(model.model, threshold=1e3)
    print("🔍 梯度检查器已启动")
    print("🚀 开始训练...")
    try:
        results = model.train(
            data=yaml_path,
            multi_scale=True,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            lr0=0.005,     # 降低初始学习率，减少震荡
            lrf=0.0001,
            cos_lr=True,
            augment=True,
            degrees=80,      # 减少旋转角度，降低数据增强强度
            mosaic=True,
            close_mosaic=50, # 提前关闭mosaic，稳定后期训练
            verbose=True,
            patience=50,     # 减少早停耐心，避免过拟合
            warmup_epochs=5, # 添加预热，稳定初期训练
            weight_decay=0.0005, # 添加权重衰减，防止过拟合
        )
        print("✅ 训练完成")
        
    except Exception as e:
        print(f"❌ 训练过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        grad_checker.remove_hooks()
        return
    
    grad_checker.remove_hooks()
    print("🧹 梯度检查器已清理")
    print("💾 正在保存模型...")
    try:
        model.save('custom_yolo11_model.pt')
        print("✅ 模型保存成功: custom_yolo11_model.pt")
    except Exception as e:
        print(f"❌ 模型保存失败: {e}")
    print("🔄 正在导出 ONNX 模型...")
    try:
        # 首先导出静态尺寸版本（精度更高）
        print("📦 导出静态尺寸ONNX模型...")
        static_onnx_path = model.export(
            format='onnx',
            dynamic=False,        # 静态尺寸，精度更高
            simplify=False,       # 不简化，保持精度
            opset=17,            # 使用较新算子集
            imgsz=(640, 640),    # 固定尺寸
            half=False,          # 使用FP32精度（不使用FP16）
            int8=False,          # 不使用INT8量化
            verbose=True         # 显示详细信息
        )
        print(f"✅ 静态ONNX模型导出成功: {static_onnx_path}")
        
        # 然后导出动态尺寸版本（灵活性更高）
        print("🔄 导出动态尺寸ONNX模型...")
        onnx_path = model.export(
            format='onnx',
            dynamic=True,         # 动态尺寸，灵活性高
            simplify=False,       # 不简化，保持精度  
            opset=17,            # 使用较新算子集
            imgsz=(640, 640),    # 默认尺寸
            half=False,          # 使用FP32精度
            int8=False,          # 不使用INT8量化
            verbose=True         # 显示详细信息
        )
        print(f"✅ 动态ONNX模型导出成功: {onnx_path}")
        
        # 精度验证建议
        print("\n📊 精度验证建议:")
        print("1. 静态模型 (xxx.onnx) - 推荐用于生产环境，精度最高")
        print("2. 动态模型 (xxx_dynamic.onnx) - 用于多尺寸推理，精度略有下降")
        print("3. 建议使用验证集对比PyTorch和ONNX模型的mAP差异")
        
    except Exception as e:
        print(f"❌ ONNX 导出失败: {e}")
        import traceback
        traceback.print_exc()
    print("🎉 ========== 训练与导出全部完成 ==========")

if __name__ == '__main__':
    print("🖼️ 开始预处理图像...")
    process_images(train_folder)
    process_images(val_folder)
    print("✅ 图像预处理完成")
    main()
