from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO(r"C:\Users\USER\OneDrive\เดสก์ท็อป\baby\YOLO11s_TrainByDefault\yolo11s.pt")
    results = model.train(
        data=r"C:\Users\USER\OneDrive\เดสก์ท็อป\baby\Augment_method1\data_original_plus_aug.yaml",
        cfg=r"C:\Users\USER\OneDrive\เดสก์ท็อป\baby\Augment_method1\train_params.yaml",
    )
