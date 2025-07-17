import onnxruntime
print(onnxruntime.__version__)
print(onnxruntime.get_device())

ort_session = onnxruntime.InferenceSession(r"C:\Users\Forever\Desktop\data integration\train8\weights\best.onnx", providers=['CUDAExecutionProvider'])
print(ort_session.get_providers())
