from ultralytics import YOLO

if __name__ == '__main__':
    #custom_trained_yolom = "../trained_models/yolo11m_oxford_tower_custom.pt"
    custom_trained_yolon = "quantized_oxford_n.pt"

    model = YOLO(custom_trained_yolon)

    metrics = model.val(verbose=True)

    accuracy = metrics.box.map * 100

    print("\nMetricas finais:")
    print("mAP50-95:", metrics.box.map)
    print("mAP50:", metrics.box.map50)
    print("mAP75:", metrics.box.map75)
    print("mAP per category:", metrics.box.maps)
    print(f"Percentual (mAP50-95): {accuracy:.2f}%")