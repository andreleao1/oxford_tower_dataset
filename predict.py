from ultralytics import YOLO

if __name__ == '__main__':
    #custom_trained_yolom = "../trained_models/yolo11m_oxford_tower_custom.pt"
    custom_trained_yolon = "test_pruned.pt"

    model = YOLO(custom_trained_yolon)

    metrics = model.val(verbose=True)

    accuracy = metrics.box.map * 100

    oxford_test_dataset = "./datasets/images/test"
    oxford_tower_video = "./datasets/oxfordtower.mp4"
    random_person_video = "./datasets/persons.mp4"
    output_video = "runs/results/prediction_result.mp4"

    # for i in range(5):
    #     print(f"\nExecutando predicao {i+1} de 5...")
    #     results = model.predict(
    #         source=oxford_test_dataset,
    #         save=True,
    #         project="runs",
    #         name=f"results_{i+1}",
    #         save_txt=False,
    #         show=True
    #     )

    print("\nMetricas finais:")
    print("mAP50-95:", metrics.box.map)
    print("mAP50:", metrics.box.map50)
    print("mAP75:", metrics.box.map75)
    print("mAP per category:", metrics.box.maps)
    print(f"Percentual (mAP50-95): {accuracy:.2f}%")