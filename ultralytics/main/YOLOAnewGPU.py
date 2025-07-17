import os
import shutil
import yaml
from datetime import datetime
from ultralytics import YOLO
from PIL import Image
import random

# 定义原始数据集和新数据集路径
original_train_folder = r'E:\data integration\person\dataset\images\train'
original_val_folder = r'E:\data integration\person\dataset\images\val'
new_data_folder = r'E:\new_data'  # 包含新数据的文件夹

# 增量训练配置
merged_dataset_path = r'E:\merged_dataset'  # 合并后的数据集路径
train_ratio = 0.8  # 训练集比例
model_path = 'custom_yolo11_model.pt'  # 已训练模型路径
new_model_name = 'incremental_model'  # 新模型名称

def prepare_dataset():
    """准备合并数据集"""
    print("准备增量训练数据集...")
    
    # 创建合并数据集目录结构
    merged_img_path = os.path.join(merged_dataset_path, 'images')
    merged_labels_path = os.path.join(merged_dataset_path, 'labels')
    os.makedirs(os.path.join(merged_img_path, 'train'), exist_ok=True)
    os.makedirs(os.path.join(merged_img_path, 'val'), exist_ok=True)
    os.makedirs(os.path.join(merged_labels_path, 'train'), exist_ok=True)
    os.makedirs(os.path.join(merged_labels_path, 'val'), exist_ok=True)
    
    # 复制原始数据集
    print("复制原始数据集...")
    def copy_original_data(folder_type):
        # 复制图像
        for file in os.listdir(os.path.join(original_train_folder, folder_type)):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                src = os.path.join(original_train_folder, folder_type, file)
                dst = os.path.join(merged_img_path, 'train', file)
                shutil.copy2(src, dst)
        
        # 复制标签
        for file in os.listdir(os.path.join(original_val_folder, folder_type)):
            if file.lower().endswith('.txt'):
                src = os.path.join(original_val_folder, folder_type, file)
                dst = os.path.join(merged_labels_path, 'train', file)
                shutil.copy2(src, dst)
    
    # 根据您的实际目录结构调整
    copy_original_data('')  # 如果原始目录直接包含图像
    
    # 处理并添加新数据集
    print("添加新数据集...")
    new_images = [f for f in os.listdir(new_data_folder) 
                 if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]
    new_labels = [f for f in os.listdir(new_data_folder) 
                 if f.lower().endswith('.txt')]
    
    # 分割新数据为训练集和验证集
    random.shuffle(new_images)
    split_idx = int(len(new_images) * train_ratio)
    train_images = new_images[:split_idx]
    val_images = new_images[split_idx:]
    
    # 处理图像格式并复制
    for img_file in train_images + val_images:
        img_path = os.path.join(new_data_folder, img_file)
        try:
            with Image.open(img_path) as img:
                # 处理GIF格式
                if img.format == 'GIF':
                    print(f"转换GIF图像: {img_file}")
                    img = img.convert('RGB')
                    img_file = os.path.splitext(img_file)[0] + '.jpg'
                
                # 保存到相应位置
                dest_folder = 'train' if img_file in train_images else 'val'
                dest_path = os.path.join(merged_img_path, dest_folder, img_file)
                img.save(dest_path, 'JPEG')
                
                # 复制对应的标签文件
                label_file = os.path.splitext(img_file)[0] + '.txt'
                if label_file in new_labels:
                    src_label = os.path.join(new_data_folder, label_file)
                    dest_label = os.path.join(merged_labels_path, dest_folder, label_file)
                    shutil.copy2(src_label, dest_label)
        except Exception as e:
            print(f"处理 {img_file} 时出错: {e}")
    
    # 创建data.yaml配置文件
    print("创建数据集配置文件...")
    data_config = {
        'path': merged_dataset_path,
        'train': 'images/train',
        'val': 'images/val',
        'names': ['person']  # 根据您的类别修改
    }
    
    config_path = os.path.join(merged_dataset_path, 'data.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(data_config, f)
    
    print(f"数据集准备完成，保存在: {merged_dataset_path}")
    return config_path

def incremental_training(data_config):
    """执行增量训练"""
    print("开始增量训练...")
    
    # 加载已训练模型
    model = YOLO(model_path)
    
    # 增量训练配置
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    training_name = f"{new_model_name}_{timestamp}"
    
    # 执行增量训练
    results = model.train(
        resume=True,  # 继续训练
        data=data_config,
        epochs=200,  # 增量训练轮次
        imgsz=640,
        batch=16,  # 根据GPU内存调整
        lr0=0.00001,  # 使用较小的学习率
        lrf=0.0000001,
        cos_lr=True,
        augment=True,
        degrees=15,  # 减少增强强度
        mosaic=True,
        close_mosaic=50,
        save_period=10,
        name=training_name,
        patience=30,  # 早停机制
        exist_ok=True
    )
    
    # 保存模型
    new_model_path = f"{new_model_name}.pt"
    model.save(new_model_path)
    
    # 导出ONNX
    onnx_path = model.export(format='onnx')
    print(f"增量训练完成! 模型已保存为: {new_model_path}")
    print(f"ONNX模型路径: {onnx_path}")
    
    return new_model_path

def main():
    # 准备合并数据集
    data_config = prepare_dataset()
    
    # 执行增量训练
    new_model = incremental_training(data_config)
    
    print(f"增量训练完成! 新模型保存在: {new_model}")
    print("您可以在以下位置查看训练结果:")
    print(f"TensorBoard日志: runs/train/{new_model_name}_*")

if __name__ == '__main__':
    main()