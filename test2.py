import onnxruntime as ort
import cv2
import numpy as np

# Carregar a sessão
session = ort.InferenceSession("trained_by_yolo11n_quant.onnx", providers=["CPUExecutionProvider"])

# Preparar imagem
img = cv2.imread("datasets/images/test/image1.jpg")
img_resized = cv2.resize(img, (640, 640))
img_input = img_resized.transpose(2, 0, 1).astype(np.float32)  # CHW
img_input = np.expand_dims(img_input, axis=0)  # NCHW

# Executar inferência
input_name = session.get_inputs()[0].name
outputs = session.run(None, {input_name: img_input})

print("Saídas:", outputs)
