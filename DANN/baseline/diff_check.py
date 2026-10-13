import csv, sys
a = list(csv.DictReader(open(sys.argv[1])))
b = list(csv.DictReader(open(sys.argv[2])))
same, diff = 0, 0
for ra, rb in zip(a, b):
    if ra["split"] != rb["split"] or ra["class"] != rb["class"]:
        print(f"ROW MISMATCH: {ra['split']}/{ra['class']} vs {rb['split']}/{rb['class']}")
        continue
    if ra["mAP50"] == rb["mAP50"]:
        same += 1
    else:
        diff += 1
        print(f"DIFFERENT: {ra['split']:15s} {ra['class']:6s} A={ra['mAP50']}  B={rb['mAP50']}")
print(f"\nidentical rows: {same}")
print(f"different rows: {diff}")
