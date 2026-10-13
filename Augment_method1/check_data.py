from ultralytics.data.utils import check_det_dataset
data = check_det_dataset(r"C:\Users\USER\OneDrive\เดสก์ท็อป\baby\Augment_method1\data_original_plus_aug.yaml")
print("nc:", data["nc"])
print("names:", data["names"])
print("train:", data["train"])
print("val:", data["val"])
