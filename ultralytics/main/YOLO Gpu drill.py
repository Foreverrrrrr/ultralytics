import os
from ultralytics import YOLO
from PIL import Image

# 定义训练集和验证集图像文件夹路径
train_folder = r'D:\AI\Ai\Internship 2.v2i.yolov8\train'
val_folder = r'D:\AI\Ai\Internship 2.v2i.yolov8\valid'

# 定义一个函数用于处理指定文件夹中的图像
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

def main():
    model = YOLO(r"ultralytics\cfg\models\11\FRFN.yaml").load("yolo11m.pt")
    data_path = r'D:\AI\Ai\Internship 2.v2i.yolov8\data.yaml'
    if os.path.exists(data_path):
        print(f"文件 {data_path} 存在。")
    else:
        print(f"文件 {data_path} 不存在，请检查路径和文件是否正确。")
        return

    results= model.train(
            data=data_path,
            multi_scale=True,
            epochs=400,
            imgsz=640,
            batch=8,
            lr0=0.0001,  # 初始学习率
            lrf=0.000001,  # 最终学习率
            cos_lr=True,  # 余弦调度器
            augment=True,  # 数据增强
            degrees=30,  # 随机旋转
            mosaic=True,
            close_mosaic=100,)

    model.save('custom_yolo11_model.pt')
    onnx_path = model.export(format='onnx')
    print(f"模型已保存为 ONNX 格式，路径为: {onnx_path}")

if __name__ == '__main__':
    process_images(train_folder)
    process_images(val_folder)
    main()