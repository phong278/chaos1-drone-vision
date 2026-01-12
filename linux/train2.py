from ultralytics import YOLO
import torch
import time
from multiprocessing import freeze_support

def main():
    # -------------------------------
    # CONFIG
    # -------------------------------
    DATA_YAML = "../dataset/data.yml"
    MODEL_NAME = "yolov8n.pt"  # start from pretrained YOLOv8n
    IMG_SIZE = 640
    BATCH = 16
    DEVICE = 0
    MAX_EPOCHS = 200
    PATIENCE = 30  # Early stopping patience

    # -------------------------------
    # GPU CHECK
    # -------------------------------
    assert torch.cuda.is_available(), "CUDA not available!"
    print("Using GPU:", torch.cuda.get_device_name(DEVICE))

    # -------------------------------
    # INIT MODEL
    # -------------------------------
    model = YOLO(MODEL_NAME)

    # -------------------------------
    # TRAINING
    # -------------------------------
    start = time.time()

    # Use advanced features for better generalization:
    # - auto_augment for data augmentation
    # - mixup/cutmix for robust training
    # - early stopping based on validation loss
    # - save the best model only
    model.train(
        data=DATA_YAML,
        epochs=200,
        imgsz=640,
        batch=16,
        device=DEVICE,
        workers=8,
        cache="disk",
        project="runs_yolo",
        name="yolov8n_mili_v1",
        patience=30,

        optimizer="AdamW",
        lr0=0.01,
        lrf=0.01,
        cos_lr=True,

        augment=True,
        mixup=0.05,
        close_mosaic=20,
        cls=0.6,
        freeze=10,

        val=True,
        plots=True,
        exist_ok=True
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
