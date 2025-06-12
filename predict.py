from ultralytics import YOLO

if __name__ == '__main__':
    custom_trained_yolon = "./pruned/3_percent/pruned_yolo11n.pt"

    model = YOLO(custom_trained_yolon)

    oxford_test_dataset = "./datasets/images/test"

    for i in range(5):
        print(f"\nExecutando predicao {i+1} de 5...")
        results = model.predict(
            source=oxford_test_dataset,
            save=False,
            project="runs",
            name=f"results_{i+1}",
            save_txt=False,
            show=False
        )