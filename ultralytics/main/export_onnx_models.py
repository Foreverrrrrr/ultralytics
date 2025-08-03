import os
from ultralytics import YOLO
import onnx
from onnxsim import simplify

class YOLOONNXExporter:
    def __init__(self, pt_path):
        """
        初始化YOLO ONNX导出器
        
        Args:
            pt_path: YOLO .pt模型文件路径
        """
        self.pt_path = pt_path
        
        if not os.path.exists(pt_path):
            raise FileNotFoundError(f"模型文件不存在: {pt_path}")
            
        print(f"🔄 正在加载模型: {pt_path}")
        try:
            self.model = YOLO(pt_path)
            print("✅ 模型加载成功")
            print(f"📊 模型类别: {self.model.names}")
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            raise
    
    def export_dynamic_onnx(self, output_path=None, imgsz=640, opset=17, simplify_model=False):
        """
        导出动态ONNX模型（支持动态batch size）
        
        Args:
            output_path: 输出ONNX文件路径（可选）
            imgsz: 输入图像尺寸
            opset: ONNX opset版本
            simplify_model: 是否简化模型
        
        Returns:
            导出的ONNX文件路径
        """
        print("\n========== 导出动态ONNX模型 ==========")
        
        if output_path is None:
            base_name = os.path.splitext(os.path.basename(self.pt_path))[0]
            output_dir = os.path.dirname(self.pt_path)
            output_path = os.path.join(output_dir, f"{base_name}_dynamic_batch_{imgsz}x{imgsz}_opset{opset}.onnx")
        
        try:
            print(f"🚀 开始导出动态ONNX模型...")
            print(f"📝 参数配置:")
            print(f"   - 输入尺寸: {imgsz}")
            print(f"   - OPSET版本: {opset}")
            print(f"   - 动态batch: True")
            print(f"   - 模型简化: {simplify_model}")
            
            # 导出动态ONNX
            exported_path = self.model.export(
                format='onnx',
                dynamic=True,           # 启用动态batch
                simplify=simplify_model,
                opset=opset,
                imgsz=imgsz,
                verbose=True
            )
            
            # 重命名到自定义路径
            if exported_path != output_path:
                import shutil
                shutil.move(exported_path, output_path)
                exported_path = output_path
            
            print(f"✅ 动态ONNX模型导出成功!")
            print(f"📁 保存路径: {exported_path}")
            
            # 验证导出的模型
            self._verify_onnx_model(exported_path, is_dynamic=True)
            
            return exported_path
            
        except Exception as e:
            print(f"❌ 动态ONNX导出失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def export_static_onnx(self, output_path=None, imgsz=640, opset=17, batch_size=1, simplify_model=True):
        """
        导出静态ONNX模型（固定batch size）
        
        Args:
            output_path: 输出ONNX文件路径（可选）
            imgsz: 输入图像尺寸
            opset: ONNX opset版本
            batch_size: 固定的batch大小
            simplify_model: 是否简化模型
        
        Returns:
            导出的ONNX文件路径
        """
        print("\n========== 导出静态ONNX模型 ==========")
        
        if output_path is None:
            base_name = os.path.splitext(os.path.basename(self.pt_path))[0]
            output_dir = os.path.dirname(self.pt_path)
            output_path = os.path.join(output_dir, f"{base_name}_static_batch{batch_size}_{imgsz}x{imgsz}_opset{opset}.onnx")
        
        try:
            print(f"🚀 开始导出静态ONNX模型...")
            print(f"📝 参数配置:")
            print(f"   - 输入尺寸: {imgsz}")
            print(f"   - OPSET版本: {opset}")
            print(f"   - 固定batch: {batch_size}")
            print(f"   - 模型简化: {simplify_model}")
            
            # 导出静态ONNX
            exported_path = self.model.export(
                format='onnx',
                dynamic=False,          # 禁用动态batch
                simplify=simplify_model,
                opset=opset,
                imgsz=imgsz,
                batch=batch_size,       # 设置固定batch size
                verbose=True
            )
            
            # 重命名到自定义路径
            if exported_path != output_path:
                import shutil
                shutil.move(exported_path, output_path)
                exported_path = output_path
            
            print(f"✅ 静态ONNX模型导出成功!")
            print(f"📁 保存路径: {exported_path}")
            
            # 验证导出的模型
            self._verify_onnx_model(exported_path, is_dynamic=False)
            
            return exported_path
            
        except Exception as e:
            print(f"❌ 静态ONNX导出失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _verify_onnx_model(self, onnx_path, is_dynamic=True):
        """
        验证导出的ONNX模型
        
        Args:
            onnx_path: ONNX文件路径
            is_dynamic: 是否为动态模型
        """
        try:
            print(f"🔍 正在验证ONNX模型: {os.path.basename(onnx_path)}")
            
            # 加载ONNX模型
            onnx_model = onnx.load(onnx_path)
            
            # 检查模型
            onnx.checker.check_model(onnx_model)
            print("✅ ONNX模型结构验证通过")
            
            # 输出模型信息
            print(f"📊 模型信息:")
            print(f"   - 文件大小: {os.path.getsize(onnx_path) / 1024 / 1024:.2f} MB")
            print(f"   - OPSET版本: {onnx_model.opset_import[0].version}")
            
            # 输出输入张量信息
            print(f"📥 输入张量信息:")
            for input_tensor in onnx_model.graph.input:
                shape = input_tensor.type.tensor_type.shape
                print(f"   - 名称: {input_tensor.name}")
                shape_str = []
                for i, dim in enumerate(shape.dim):
                    if dim.HasField('dim_value'):
                        shape_str.append(str(dim.dim_value))
                    elif dim.HasField('dim_param'):
                        shape_str.append(dim.dim_param)
                    else:
                        shape_str.append('?')
                print(f"     形状: [{', '.join(shape_str)}]")
            
            # 输出输出张量信息
            print(f"📤 输出张量信息:")
            for output_tensor in onnx_model.graph.output:
                shape = output_tensor.type.tensor_type.shape
                print(f"   - 名称: {output_tensor.name}")
                shape_str = []
                for i, dim in enumerate(shape.dim):
                    if dim.HasField('dim_value'):
                        shape_str.append(str(dim.dim_value))
                    elif dim.HasField('dim_param'):
                        shape_str.append(dim.dim_param)
                    else:
                        shape_str.append('?')
                print(f"     形状: [{', '.join(shape_str)}]")
            
            # 检查是否真的是动态/静态
            has_dynamic_shape = False
            for input_tensor in onnx_model.graph.input:
                shape = input_tensor.type.tensor_type.shape
                for dim in shape.dim:
                    if dim.HasField('dim_param') or not dim.HasField('dim_value'):
                        has_dynamic_shape = True
                        break
            
            if is_dynamic and has_dynamic_shape:
                print("✅ 确认为动态batch模型")
            elif not is_dynamic and not has_dynamic_shape:
                print("✅ 确认为静态batch模型")
            else:
                print("⚠️ 模型类型与预期不符")
                
        except Exception as e:
            print(f"❌ ONNX模型验证失败: {e}")
    
    def export_both(self, output_dir=None, imgsz=640, opset=17, batch_size=1, simplify_model=True):
        """
        同时导出动态和静态ONNX模型
        
        Args:
            output_dir: 输出目录
            imgsz: 输入图像尺寸
            opset: ONNX opset版本
            batch_size: 静态模型的batch大小
            simplify_model: 是否简化模型
        
        Returns:
            tuple: (动态模型路径, 静态模型路径)
        """
        print("========== 同时导出动态和静态ONNX模型 ==========")
        
        if output_dir is None:
            output_dir = os.path.dirname(self.pt_path)
        
        base_name = os.path.splitext(os.path.basename(self.pt_path))[0]
        
        # 设置输出路径
        dynamic_path = os.path.join(output_dir, f"{base_name}_dynamic_batch_{imgsz}x{imgsz}_opset{opset}.onnx")
        static_path = os.path.join(output_dir, f"{base_name}_static_batch{batch_size}_{imgsz}x{imgsz}_opset{opset}.onnx")
        
        # 导出动态模型
        dynamic_result = self.export_dynamic_onnx(
            output_path=dynamic_path,
            imgsz=imgsz,
            opset=opset,
            simplify_model=simplify_model
        )
        
        # 导出静态模型
        static_result = self.export_static_onnx(
            output_path=static_path,
            imgsz=imgsz,
            opset=opset,
            batch_size=batch_size,
            simplify_model=simplify_model
        )
        
        print("\n🎉 ========== 导出完成 ==========")
        if dynamic_result:
            print(f"✅ 动态模型: {dynamic_result}")
        if static_result:
            print(f"✅ 静态模型: {static_result}")
        
        return dynamic_result, static_result

def main():
    # 配置参数
    PT_MODEL_PATH = r"D:\train18\weights\best.pt"  # 你的.pt模型路径
    OUTPUT_DIR = None        # 输出目录（None表示与源文件同目录）
    IMG_SIZE = 640          # 输入图像尺寸
    OPSET_VERSION = 17      # ONNX opset版本
    BATCH_SIZE = 1          # 静态模型的batch大小
    SIMPLIFY_MODEL = False   # 是否简化模型
    
    print("========== YOLO ONNX 模型导出工具 ==========")
    print(f"🎯 源模型: {PT_MODEL_PATH}")
    print(f"📁 输出目录: {OUTPUT_DIR or '与源文件同目录'}")
    print(f"🖼️ 图像尺寸: {IMG_SIZE}")
    print(f"🔧 OPSET版本: {OPSET_VERSION}")
    print(f"📦 静态batch: {BATCH_SIZE}")
    print(f"⚡ 模型简化: {SIMPLIFY_MODEL}")
    
    try:
        # 初始化导出器
        exporter = YOLOONNXExporter(PT_MODEL_PATH)
        
        # 同时导出动态和静态模型
        dynamic_path, static_path = exporter.export_both(
            output_dir=OUTPUT_DIR,
            imgsz=IMG_SIZE,
            opset=OPSET_VERSION,
            batch_size=BATCH_SIZE,
            simplify_model=SIMPLIFY_MODEL
        )
        
        print("\n📝 使用建议:")
        print("- 动态模型适用于推理时batch size变化的场景")
        print("- 静态模型通常有更好的推理性能")
        print("- 可以使用 onnxruntime 加载模型进行推理测试")
        
    except Exception as e:
        print(f"❌ 程序执行失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
