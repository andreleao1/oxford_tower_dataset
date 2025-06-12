import os
import time
from datetime import datetime
import pandas as pd
from ultralytics import YOLO

if __name__ == '__main__':
    percentages = ['3', '10', '15', '20', '25', '30', '35', '40']
    yolo_versions = ['n', 's', 'm', 'l', 'x']

    results = []

    for pruning_percent in percentages:
        for version in yolo_versions:
            model_name = f"YOLO11{version}"
            model_file = f"./pruned_models/{pruning_percent}_percent/pruned_yolo11{version}.pt"

            print(f"\n🔍 Avaliando modelo: {model_file}")

            start_time = datetime.now()

            model = YOLO(model_file)
            metrics = model.val(verbose=False)

            end_time = datetime.now()
            duration = (end_time - start_time)

            results.append({
                "Model": model_name,
                "Pruning (%)": pruning_percent,
                "Dataset": "oxford tower",
                "Execution Date": start_time.strftime('%d/%m/%Y'),
                "Start Time": start_time.strftime('%H:%M:%S'),
                "End Time": end_time.strftime('%H:%M:%S'),
                "Execution Time": f"{duration.seconds//60}:{duration.seconds%60:02d}",
                "mAP50-95": round(metrics.box.map * 100, 3),
                "mAP50": round(metrics.box.map50 * 100, 3),
                "mAP75": round(metrics.box.map75 * 100, 3)
            })

    # Salvar em planilha
    os.makedirs("results", exist_ok=True)
    df = pd.DataFrame(results)
    df.to_excel("results/yolo_pruning_metrics.xlsx", index=False)

    print("\n✅ Planilha salva em: results/yolo_pruning_metrics.xlsx")
