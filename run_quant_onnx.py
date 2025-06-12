import onnxruntime as ort
import numpy as np
import yaml
from PIL import Image
import glob
import os
from mean_average_precision import MetricBuilder

# --- Funções auxiliares ---

def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)

def preprocess_image(img_path, img_size=640):
    img = Image.open(img_path).convert("RGB").resize((img_size, img_size))
    img = np.array(img).transpose(2, 0, 1).astype("float32") / 255.0
    img = img[np.newaxis, :]
    return img

def xywhn_to_xyxy(xywhn, img_w, img_h):
    # Converte [x_center, y_center, w, h] (normalizado) para [x1, y1, x2, y2] em pixels
    x_c, y_c, w, h = xywhn
    x1 = (x_c - w / 2) * img_w
    y1 = (y_c - h / 2) * img_h
    x2 = (x_c + w / 2) * img_w
    y2 = (y_c + h / 2) * img_h
    return [x1, y1, x2, y2]

def load_labels(label_path, img_w, img_h):
    boxes = []
    classes = []
    with open(label_path, "r") as f:
        for line in f.readlines():
            c, x_c, y_c, w, h = map(float, line.strip().split())
            boxes.append(xywhn_to_xyxy([x_c, y_c, w, h], img_w, img_h))
            classes.append(int(c))
    return np.array(boxes), np.array(classes)

def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45):
    # prediction: [num_preds, 6] -> [x1,y1,x2,y2,score,class]
    # Implementar NMS usando torchvision ou numpy
    # Aqui uma versão simples com PyTorch (recomendo instalar pytorch)
    import torch
    boxes = torch.tensor(prediction[:, :4])
    scores = torch.tensor(prediction[:, 4])
    classes = torch.tensor(prediction[:, 5])
    keep = []
    output_boxes = []
    for cls in classes.unique():
        mask = classes == cls
        b = boxes[mask]
        s = scores[mask]
        idxs = torch.ops.torchvision.nms(b, s, iou_thres)
        output_boxes.append(torch.cat((b[idxs], s[idxs, None], cls.repeat(len(idxs),1).float()), 1))
    if len(output_boxes):
        return torch.cat(output_boxes).numpy()
    else:
        return np.zeros((0,6))

# --- Main ---

data_yaml = "data.yaml"
data = load_yaml(data_yaml)

test_images = glob.glob(os.path.join(data['test'], "*.jpg"))
num_classes = data['nc']

sess = ort.InferenceSession("trained_by_yolo11n_quant.onnx")
input_name = sess.get_inputs()[0].name

metric_fn = MetricBuilder.build_evaluation_metric("map_2d", async_mode=False, num_classes=num_classes)

for img_path in test_images:
    img_orig = Image.open(img_path)
    img_w, img_h = img_orig.size

    img = preprocess_image(img_path)
    outputs = sess.run(None, {input_name: img})

    preds = outputs[0][0]  # shape (num_preds, 6) exemplo: [x1,y1,x2,y2,score,class]

    # Filtrar por confiança e NMS
    preds = preds[preds[:,4] > 0.25]
    preds = non_max_suppression(preds, conf_thres=0.25, iou_thres=0.45)

    # Preparar para métrica
    # preds no formato [image_idx, class, x1, y1, x2, y2, score]
    preds_formatted = []
    for p in preds:
        x1, y1, x2, y2, conf, cls = p
        preds_formatted.append([0, int(cls), x1, y1, x2, y2, conf])
    preds_formatted = np.array(preds_formatted)

    # Carregar ground truth
    label_path = os.path.splitext(img_path)[0] + ".txt"
    gt_boxes, gt_classes = load_labels(label_path, img_w, img_h)
    # GT no formato [image_idx, class, x1, y1, x2, y2]
    gt_formatted = []
    for i in range(len(gt_boxes)):
        x1,y1,x2,y2 = gt_boxes[i]
        cls = gt_classes[i]
        gt_formatted.append([0, cls, x1, y1, x2, y2])
    gt_formatted = np.array(gt_formatted)

    # Atualizar métrica
    if preds_formatted.shape[0] > 0:
        metric_fn.add(preds_formatted, gt_formatted)
    else:
        # Se sem predições, adicionar GT vazio
        metric_fn.add(np.zeros((0,7)), gt_formatted)

# Resultado
metrics = metric_fn.value(iou_thresholds=np.arange(0.5, 0.96, 0.05))
print("mAP@0.5:0.95 =", metrics["mAP"])
