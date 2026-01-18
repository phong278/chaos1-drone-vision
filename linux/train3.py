from ultralytics import YOLO
import torch
import time
from multiprocessing import freeze_support
from pathlib import Path

def main():
    # ------------------------------- 
    # CONFIG
    # ------------------------------- 
    DATA_YAML = "../dataset/data.yml"         # Updated dataset including Ships
    PRETRAINED_CHECKPOINT = "runs_yolo/yolov8n_kiit_mita_v2/weights/best.pt"  
    IMG_SIZE = 640
    BATCH = 16
    DEVICE = 0
    MAX_EPOCHS = 150
    PATIENCE = 50   # Early stopping patience

    # ------------------------------- 
    # GPU CHECK
    # ------------------------------- 
    assert torch.cuda.is_available(), "CUDA not available!"
    print("Using GPU:", torch.cuda.get_device_name(DEVICE))

    # ------------------------------- 
    # INIT MODEL
    # ------------------------------- 
    model = YOLO(PRETRAINED_CHECKPOINT)

    # ------------------------------- 
    # TRAINING
    # ------------------------------- 
    start = time.time()
    print("\nStarting V4 training...\n")

    model.train(
        data=DATA_YAML,
        epochs=MAX_EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH,
        device=DEVICE,
        workers=8,
        cache="disk",
        project="runs_yolo",
        name="yolov8n_mili_v5",
        patience=PATIENCE,

        optimizer="AdamW",
        lr0=0.003,         # moderate LR
        lrf=0.05,          # final LR factor
        cos_lr=True,       # cosine decay

        augment=True,
        mosaic=True,
        mixup=0.05,
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        flipud=0.5, fliplr=0.5,
        degrees=10,
        translate=0.1,
        scale=0.1,
        shear=2,
        perspective=0.0,

        cls=0.6,
        freeze=10,          # freeze backbone for first few epochs

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

    print("\nAll done! Model exported to ONNX and OpenVINO.")

if __name__ == "__main__":
    freeze_support()
    main()
