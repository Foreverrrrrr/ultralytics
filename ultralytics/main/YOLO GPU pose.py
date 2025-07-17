from ultralytics import YOLO
def main():
# Load a model
 model = YOLO("yolo11n-pose.pt")  # load a pretrained model (recommended for training)
 results = model.train(data=r"C:\Users\Forever\Desktop\data integration\hand gesture recognition\data.yaml", epochs=50, imgsz=640,batch=24)
 model.save('pose_yolo11_model.pt')
 onnx_path = model.export(format='onnx')
 print(f"模型已保存为 ONNX 格式，路径为: {onnx_path}")

if __name__ == '__main__':
    main()