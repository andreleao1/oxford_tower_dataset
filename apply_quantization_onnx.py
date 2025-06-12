from onnxruntime.quantization import quantize_dynamic, QuantType

quantize_dynamic(
    model_input="trained_by_yolo11n.onnx",
    model_output="trained_by_yolo11n_quant.onnx",
    weight_type=QuantType.QInt8
)