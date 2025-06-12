import torch
from torch.quantization import quantize_dynamic

if torch.cuda.is_available():
    device = torch.device('cuda')
    print("GPU disponível. Carregando modelo na GPU.")
else:
    device = torch.device('cpu')
    print("GPU não disponível. Carregando modelo na CPU.")

# Load your 
model_data = torch.load('trained_by_yolo11n.pt', map_location=device, weights_only=False)
model = model_data['model'].float()

# Apply dynamic quantization
quantized_model = quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)

# Save the quantized model
torch.save(quantized_model, 'quantized_oxford_n.pt')

print("Modelo processado e salvo com sucesso!")