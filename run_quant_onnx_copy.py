import onnxruntime as ort
import numpy as np
import cv2
import os
import time

# Caminho do modelo quantizado ONNX
model_path = "trained_by_yolo11n_quant.onnx"

# Pasta com imagens de teste
images_folder = r"C:\workspace\oxford_tower\datasets\images\test"

# Cria sessão ONNXRuntime para CPU
session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape  # ex: [1, 3, 640, 640]

def preprocess_image(img_path, input_shape):
    # Carrega imagem com OpenCV
    img = cv2.imread(img_path)
    # Redimensiona para o tamanho esperado pelo modelo
    img = cv2.resize(img, (input_shape[3], input_shape[2]))
    # Converte BGR para RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Normaliza (exemplo comum: 0-1)
    img = img.astype(np.float32) / 255.0
    # Transpõe HWC para CHW
    img = np.transpose(img, (2, 0, 1))
    # Adiciona batch dimension
    img = np.expand_dims(img, axis=0)
    return img

# Lista as imagens
image_files = [f for f in os.listdir(images_folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

for img_file in image_files:
    img_path = os.path.join(images_folder, img_file)
    input_tensor = preprocess_image(img_path, input_shape)

    start = time.time()
    outputs = session.run(None, {input_name: input_tensor})
    end = time.time()

    print(f"Imagem: {img_file} | Tempo inferência: {end - start:.4f}s")
    # Aqui você pode tratar outputs para extrair as métricas que quiser


