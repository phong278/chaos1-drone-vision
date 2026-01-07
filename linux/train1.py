from ultralytics import YOLO
import torch
import time
from multiprocessing import freeze_support

def main():
    # -------------------------------
    # CONFIG
    # -------------------------------
    DATA_YAML = "dataset/data.yml"
    MODEL_NAME = "yolov8n.pt"
    IMG_SIZE = 640
    EPOCHS = 100
    BATCH = 16
    DEVICE = 0

    # -------------------------------
    # GPU CHECK
    # -------------------------------
    assert torch.cuda.is_available(), "CUDA not available!"
    print("Using GPU:", torch.cuda.get_device_name(0))

    # -------------------------------
    # TRAIN
    # -------------------------------
    start = time.time()

    model = YOLO(MODEL_NAME)

    model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH,
        device=DEVICE,
        workers=8,          # OK now
        cache="disk",       # deterministic + avoids RAM issues
        project="runs_yolo",
        name="yolov8n_kiit_mita",
        patience=20,
    )

    print(f"\nTraining completed in {(time.time() - start)/60:.1f} minutes")

    # -------------------------------
    # VALIDATION
    # -------------------------------
    val_results = model.val(
        data=DATA_YAML,
        imgsz=IMG_SIZE,
        device=DEVICE
    )

    print("\nValidation metrics:")
    print(f"mAP50-95: {val_results.box.map:.4f}")
    print(f"mAP50:    {val_results.box.map50:.4f}")
    print(f"Precision:{val_results.box.mp:.4f}")
    print(f"Recall:   {val_results.box.mr:.4f}")

    # -------------------------------
    # TEST SET
    # -------------------------------
    test_results = model.val(
        data=DATA_YAML,
        imgsz=IMG_SIZE,
        device=DEVICE,
        split="test"
    )

    print("\nTest metrics:")
    print(f"mAP50-95: {test_results.box.map:.4f}")
    print(f"mAP50:    {test_results.box.map50:.4f}")
    print(f"Precision:{test_results.box.mp:.4f}")
    print(f"Recall:   {test_results.box.mr:.4f}")

    # -------------------------------
    # EXPORT
    # -------------------------------
    model.export(format="onnx", imgsz=IMG_SIZE, simplify=True)
    model.export(format="openvino", imgsz=IMG_SIZE)

    print("\nAll done!")

if __name__ == "__main__":
    freeze_support()
    main()
