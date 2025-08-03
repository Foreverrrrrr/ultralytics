import os
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import time

class YOLOFolderInference:
    def __init__(self, model_path, conf_threshold=0.25, iou_threshold=0.45):
        """
        初始化YOLO推理器
        
        Args:
            model_path: YOLO模型文件路径 (.pt文件)
            conf_threshold: 置信度阈值
            iou_threshold: NMS IoU阈值
        """
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        print(f"🔄 正在加载模型: {model_path}")
        try:
            self.model = YOLO(model_path)
            print("✅ 模型加载成功")
            print(f"📊 模型类别: {self.model.names}")
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            raise
    
    def inference_single_image(self, img_path, save_path=None, show_result=True):
        """
        对单张图片进行推理
        
        Args:
            img_path: 图片路径
            save_path: 保存结果图片的路径（可选）
            show_result: 是否显示结果
        
        Returns:
            results: YOLO推理结果
        """
        try:
            print(f"🔍 正在推理: {os.path.basename(img_path)}")
            
            # 记录推理时间
            start_time = time.time()
            
            # 执行推理
            results = self.model(
                img_path,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                verbose=False
            )
            
            inference_time = (time.time() - start_time) * 1000
            print(f"⏱️ 推理耗时: {inference_time:.2f} ms")
            
            # 获取检测结果
            result = results[0]
            boxes = result.boxes
            
            if boxes is not None and len(boxes) > 0:
                print(f"🎯 检测到 {len(boxes)} 个目标:")
                
                # 读取原图用于绘制
                img = cv2.imread(img_path)
                img_height, img_width = img.shape[:2]
                
                for i, box in enumerate(boxes):
                    # 获取边界框坐标
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    cls = int(box.cls[0].cpu().numpy())
                    class_name = self.model.names[cls]
                    
                    print(f"   目标{i+1}: {class_name} 置信度:{conf:.3f} 坐标:({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f})")
                    
                    # 绘制边界框
                    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    
                    # 绘制标签
                    label = f"{class_name}: {conf:.3f}"
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    cv2.rectangle(img, (int(x1), int(y1) - label_size[1] - 10), 
                                (int(x1) + label_size[0], int(y1)), (0, 255, 0), -1)
                    cv2.putText(img, label, (int(x1), int(y1) - 5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                
                # 保存结果图片
                if save_path:
                    cv2.imwrite(save_path, img)
                    print(f"💾 结果已保存到: {save_path}")
                
                # 显示结果
                if show_result:
                    # 调整显示大小
                    max_display_size = 800
                    if max(img_height, img_width) > max_display_size:
                        scale = max_display_size / max(img_height, img_width)
                        new_width = int(img_width * scale)
                        new_height = int(img_height * scale)
                        img = cv2.resize(img, (new_width, new_height))
                    
                    cv2.imshow(f'Detection Result - {os.path.basename(img_path)}', img)
                    print("📺 按任意键继续下一张图片...")
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
            else:
                print("❌ 未检测到任何目标")
            
            return results
            
        except Exception as e:
            print(f"❌ 推理失败: {e}")
            return None
    
    def inference_folder(self, input_folder, output_folder=None, show_results=False):
        """
        对文件夹中的所有图片进行推理
        
        Args:
            input_folder: 输入图片文件夹路径
            output_folder: 输出结果文件夹路径（可选）
            show_results: 是否逐张显示结果
        """
        if not os.path.exists(input_folder):
            print(f"❌ 输入文件夹不存在: {input_folder}")
            return
        
        # 创建输出文件夹
        if output_folder:
            os.makedirs(output_folder, exist_ok=True)
            print(f"📁 结果将保存到: {output_folder}")
        
        # 支持的图片格式
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        
        # 获取所有图片文件
        image_files = []
        for filename in os.listdir(input_folder):
            if any(filename.lower().endswith(ext) for ext in image_extensions):
                image_files.append(filename)
        
        if not image_files:
            print(f"❌ 在 {input_folder} 中未找到图片文件")
            return
        
        print(f"📸 找到 {len(image_files)} 张图片")
        print("🚀 开始批量推理...")
        
        total_time = 0
        successful_inferences = 0
        
        for i, filename in enumerate(image_files, 1):
            print(f"\n--- 处理第 {i}/{len(image_files)} 张图片 ---")
            
            img_path = os.path.join(input_folder, filename)
            
            # 设置保存路径
            save_path = None
            if output_folder:
                name, ext = os.path.splitext(filename)
                save_path = os.path.join(output_folder, f"{name}_result{ext}")
            
            # 执行推理
            start_time = time.time()
            results = self.inference_single_image(img_path, save_path, show_results)
            inference_time = time.time() - start_time
            
            if results is not None:
                total_time += inference_time
                successful_inferences += 1
        
        # 统计信息
        print(f"\n🎉 批量推理完成!")
        print(f"📊 成功处理: {successful_inferences}/{len(image_files)} 张图片")
        if successful_inferences > 0:
            avg_time = (total_time / successful_inferences) * 1000
            print(f"⏱️ 平均推理时间: {avg_time:.2f} ms")

def main():
    # 配置参数
    MODEL_PATH = r"D:\train\weights\best.pt"  # 你的模型路径
    INPUT_FOLDER = r"D:\AI\Ai\demo712\cut1"         # 输入图片文件夹
    OUTPUT_FOLDER = r"D:\AI\Ai\demo712\aaa"       # 输出结果文件夹
    
    CONF_THRESHOLD = 0.55    # 置信度阈值
    IOU_THRESHOLD = 0.55     # NMS IoU阈值
    SHOW_RESULTS = True      # 是否显示推理结果
    
    print("========== YOLO 文件夹推理 ==========")
    print(f"🎯 模型路径: {MODEL_PATH}")
    print(f"📁 输入文件夹: {INPUT_FOLDER}")
    print(f"💾 输出文件夹: {OUTPUT_FOLDER}")
    print(f"🎚️ 置信度阈值: {CONF_THRESHOLD}")
    print(f"🔧 IoU阈值: {IOU_THRESHOLD}")
    
    try:
        # 初始化推理器
        inference = YOLOFolderInference(
            model_path=MODEL_PATH,
            conf_threshold=CONF_THRESHOLD,
            iou_threshold=IOU_THRESHOLD
        )
        
        # 执行文件夹推理
        inference.inference_folder(
            input_folder=INPUT_FOLDER,
            output_folder=OUTPUT_FOLDER,
            show_results=SHOW_RESULTS
        )
        
    except Exception as e:
        print(f"❌ 程序执行失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
