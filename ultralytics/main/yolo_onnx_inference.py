import os
import cv2
import numpy as np
import time
import onnxruntime as ort
from typing import List, Tuple, Optional

class YOLOONNXInference:
    def __init__(self, onnx_path: str, conf_threshold: float = 0.25, iou_threshold: float = 0.45, class_names: Optional[List[str]] = None):
        """
        初始化YOLO ONNX推理器
        
        Args:
            onnx_path: ONNX模型文件路径 (.onnx文件)
            conf_threshold: 置信度阈值
            iou_threshold: NMS IoU阈值
            class_names: 类别名称列表，如果为None则使用数字ID
        """
        self.onnx_path = onnx_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.class_names = class_names
        
        print(f"🔄 正在加载ONNX模型: {onnx_path}")
        try:
            # 设置ONNX Runtime提供者（优先使用GPU）
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            self.session = ort.InferenceSession(onnx_path, providers=providers)
            
            # 获取模型输入输出信息
            self.input_name = self.session.get_inputs()[0].name
            self.output_names = [output.name for output in self.session.get_outputs()]
            
            # 获取输入尺寸
            input_shape = self.session.get_inputs()[0].shape
            self.input_height = input_shape[2] if len(input_shape) > 2 else 640
            self.input_width = input_shape[3] if len(input_shape) > 3 else 640
            
            print("✅ ONNX模型加载成功")
            print(f"📊 输入尺寸: {self.input_width}x{self.input_height}")
            print(f"🔧 执行提供者: {self.session.get_providers()}")
            
            # 如果没有提供类别名称，创建默认的
            if self.class_names is None:
                self.class_names = [f"class_{i}" for i in range(1000)]  # 默认1000个类别
                
        except Exception as e:
            print(f"❌ ONNX模型加载失败: {e}")
            raise
    
    def preprocess_image(self, image: np.ndarray) -> Tuple[np.ndarray, float, float, float, float]:
        """
        预处理图像
        
        Args:
            image: 输入图像(BGR格式)
            
        Returns:
            处理后的图像数据、缩放因子和填充信息
        """
        # 获取原始图像尺寸
        original_height, original_width = image.shape[:2]
        
        # 计算缩放比例
        scale = min(self.input_width / original_width, self.input_height / original_height)
        
        # 计算新尺寸
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)
        
        # 调整图像大小
        resized_image = cv2.resize(image, (new_width, new_height))
        
        # 创建填充后的图像
        padded_image = np.full((self.input_height, self.input_width, 3), 114, dtype=np.uint8)
        
        # 计算填充位置
        pad_x = (self.input_width - new_width) // 2
        pad_y = (self.input_height - new_height) // 2
        
        # 将调整后的图像放入填充图像中
        padded_image[pad_y:pad_y + new_height, pad_x:pad_x + new_width] = resized_image
        
        # 转换为RGB并归一化
        rgb_image = cv2.cvtColor(padded_image, cv2.COLOR_BGR2RGB)
        normalized_image = rgb_image.astype(np.float32) / 255.0
        
        # 转换维度 (H, W, C) -> (1, C, H, W)
        input_tensor = np.transpose(normalized_image, (2, 0, 1))
        input_tensor = np.expand_dims(input_tensor, axis=0)
        
        return input_tensor, scale, pad_x, pad_y, original_width, original_height
    
    def postprocess_outputs(self, outputs: List[np.ndarray], scale: float, pad_x: float, pad_y: float, 
                          original_width: int, original_height: int) -> List[dict]:
        """
        后处理模型输出
        
        Args:
            outputs: 模型输出
            scale: 缩放因子
            pad_x, pad_y: 填充量
            original_width, original_height: 原始图像尺寸
            
        Returns:
            检测结果列表
        """
        # 获取输出数据 (假设输出格式为 [batch, num_detections, 5+num_classes])
        predictions = outputs[0][0]  # 移除batch维度
        
        detections = []
        
        for prediction in predictions:
            # 解析预测结果
            x_center, y_center, width, height = prediction[:4]
            class_scores = prediction[4:]
            
            # 找到最高置信度的类别
            max_score = np.max(class_scores)
            if max_score < self.conf_threshold:
                continue
                
            class_id = np.argmax(class_scores)
            
            # 转换坐标到原始图像尺寸
            x_center = (x_center - pad_x) / scale
            y_center = (y_center - pad_y) / scale
            width = width / scale
            height = height / scale
            
            # 计算边界框坐标
            x1 = x_center - width / 2
            y1 = y_center - height / 2
            x2 = x_center + width / 2
            y2 = y_center + height / 2
            
            # 限制在图像范围内
            x1 = max(0, min(x1, original_width))
            y1 = max(0, min(y1, original_height))
            x2 = max(0, min(x2, original_width))
            y2 = max(0, min(y2, original_height))
            
            detections.append({
                'bbox': [x1, y1, x2, y2],
                'confidence': float(max_score),
                'class_id': int(class_id),
                'class_name': self.class_names[class_id] if class_id < len(self.class_names) else f"class_{class_id}"
            })
        
        # 应用NMS
        if len(detections) > 0:
            detections = self.apply_nms(detections)
        
        return detections
    
    def apply_nms(self, detections: List[dict]) -> List[dict]:
        """
        应用非极大值抑制
        
        Args:
            detections: 检测结果列表
            
        Returns:
            NMS后的检测结果
        """
        if len(detections) == 0:
            return detections
        
        # 提取边界框和置信度
        boxes = np.array([det['bbox'] for det in detections])
        scores = np.array([det['confidence'] for det in detections])
        
        # 使用OpenCV的NMS
        indices = cv2.dnn.NMSBoxes(
            boxes.tolist(), 
            scores.tolist(), 
            self.conf_threshold, 
            self.iou_threshold
        )
        
        if len(indices) > 0:
            indices = indices.flatten()
            return [detections[i] for i in indices]
        else:
            return []
    
    def inference_single_image(self, img_path: str, save_path: Optional[str] = None, show_result: bool = True) -> Optional[List[dict]]:
        """
        对单张图片进行推理
        
        Args:
            img_path: 图片路径
            save_path: 保存结果图片的路径（可选）
            show_result: 是否显示结果
        
        Returns:
            检测结果列表
        """
        try:
            print(f"🔍 正在推理: {os.path.basename(img_path)}")
            
            # 读取图像
            image = cv2.imread(img_path)
            if image is None:
                print(f"❌ 无法读取图像: {img_path}")
                return None
            
            # 预处理
            start_time = time.time()
            input_tensor, scale, pad_x, pad_y, original_width, original_height = self.preprocess_image(image)
            preprocess_time = (time.time() - start_time) * 1000
            
            # 推理
            start_time = time.time()
            outputs = self.session.run(self.output_names, {self.input_name: input_tensor})
            inference_time = (time.time() - start_time) * 1000
            
            # 后处理
            start_time = time.time()
            detections = self.postprocess_outputs(outputs, scale, pad_x, pad_y, original_width, original_height)
            postprocess_time = (time.time() - start_time) * 1000
            
            total_time = preprocess_time + inference_time + postprocess_time
            print(f"⏱️ 总耗时: {total_time:.2f} ms (预处理: {preprocess_time:.2f} ms, 推理: {inference_time:.2f} ms, 后处理: {postprocess_time:.2f} ms)")
            
            if len(detections) > 0:
                print(f"🎯 检测到 {len(detections)} 个目标:")
                
                # 绘制检测结果
                result_image = image.copy()
                for i, detection in enumerate(detections):
                    x1, y1, x2, y2 = [int(coord) for coord in detection['bbox']]
                    confidence = detection['confidence']
                    class_name = detection['class_name']
                    
                    print(f"   目标{i+1}: {class_name} 置信度:{confidence:.3f} 坐标:({x1},{y1},{x2},{y2})")
                    
                    # 绘制边界框
                    cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # 绘制标签
                    label = f"{class_name}: {confidence:.3f}"
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    cv2.rectangle(result_image, (x1, y1 - label_size[1] - 10), 
                                (x1 + label_size[0], y1), (0, 255, 0), -1)
                    cv2.putText(result_image, label, (x1, y1 - 5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                
                # 保存结果图片
                if save_path:
                    cv2.imwrite(save_path, result_image)
                    print(f"💾 结果已保存到: {save_path}")
                
                # 显示结果
                if show_result:
                    # 调整显示大小
                    max_display_size = 800
                    img_height, img_width = result_image.shape[:2]
                    if max(img_height, img_width) > max_display_size:
                        scale_display = max_display_size / max(img_height, img_width)
                        new_width = int(img_width * scale_display)
                        new_height = int(img_height * scale_display)
                        result_image = cv2.resize(result_image, (new_width, new_height))
                    
                    cv2.imshow(f'ONNX Detection Result - {os.path.basename(img_path)}', result_image)
                    print("📺 按任意键继续下一张图片...")
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
            else:
                print("❌ 未检测到任何目标")
            
            return detections
            
        except Exception as e:
            print(f"❌ 推理失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def inference_folder(self, input_folder: str, output_folder: Optional[str] = None, show_results: bool = False):
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
        total_detections = 0
        
        for i, filename in enumerate(image_files, 1):
            print(f"\n--- 处理第 {i}/{len(image_files)} 张图片 ---")
            
            img_path = os.path.join(input_folder, filename)
            
            # 设置保存路径
            save_path = None
            if output_folder:
                name, ext = os.path.splitext(filename)
                save_path = os.path.join(output_folder, f"{name}_onnx_result{ext}")
            
            # 执行推理
            start_time = time.time()
            detections = self.inference_single_image(img_path, save_path, show_results)
            inference_time = time.time() - start_time
            
            if detections is not None:
                total_time += inference_time
                successful_inferences += 1
                total_detections += len(detections)
        
        # 统计信息
        print(f"\n🎉 批量推理完成!")
        print(f"📊 成功处理: {successful_inferences}/{len(image_files)} 张图片")
        print(f"🎯 总检测目标: {total_detections} 个")
        if successful_inferences > 0:
            avg_time = (total_time / successful_inferences) * 1000
            avg_detections = total_detections / successful_inferences
            print(f"⏱️ 平均推理时间: {avg_time:.2f} ms")
            print(f"📈 平均检测目标: {avg_detections:.1f} 个/图")

def main():
    # 配置参数
    ONNX_MODEL_PATH = r"D:\train\weights\best_high_precision.onnx"  # 你的ONNX模型路径
    INPUT_FOLDER = r"D:\AI\Ai\demo712\cut1"         # 输入图片文件夹
    OUTPUT_FOLDER = r"D:\AI\Ai\demo712\aaa"  # 输出结果文件夹
    
    CONF_THRESHOLD = 0.55    # 置信度阈值
    IOU_THRESHOLD = 0.55     # NMS IoU阈值
    SHOW_RESULTS = True      # 是否显示推理结果
    
    # 类别名称（根据你的模型修改）
    CLASS_NAMES = ['QR_code']  # 根据你的模型类别修改
    
    print("========== YOLO ONNX 文件夹推理 ==========")
    print(f"🎯 ONNX模型路径: {ONNX_MODEL_PATH}")
    print(f"📁 输入文件夹: {INPUT_FOLDER}")
    print(f"💾 输出文件夹: {OUTPUT_FOLDER}")
    print(f"🎚️ 置信度阈值: {CONF_THRESHOLD}")
    print(f"🔧 IoU阈值: {IOU_THRESHOLD}")
    print(f"📋 类别名称: {CLASS_NAMES}")
    
    try:
        # 初始化ONNX推理器
        inference = YOLOONNXInference(
            onnx_path=ONNX_MODEL_PATH,
            conf_threshold=CONF_THRESHOLD,
            iou_threshold=IOU_THRESHOLD,
            class_names=CLASS_NAMES
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
