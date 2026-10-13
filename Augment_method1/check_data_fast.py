import yaml
with open(r"C:\Users\USER\OneDrive\เดสก์ท็อป\baby\Augment_method1\data_original_plus_aug.yaml", "r", encoding="utf-8") as f:
    data = yaml.safe_load(f)
print("nc:", data.get("nc"))
print("names:", data.get("names"))
print("train:", data.get("train"))
print("val:", data.get("val"))
