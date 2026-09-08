#!/usr/bin/env python3
"""One-shot Phase-0 patcher: adapts the archived YOLOv5-era JEPA code to this
repo (YOLOv11 + torch 2.6 + widened adverse-environment scope).

Idempotent: re-running it is a no-op. Every replacement is asserted, so a
silent partial patch is impossible.
"""
import sys
from pathlib import Path

REPO = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]

NEW_WEIGHTS_FN = '''# attribute value -> config key (yaml keys can't contain '/' or ' ')
_TOD_KEY = {"daytime": "daytime", "night": "night", "dawn/dusk": "dawn_dusk"}
_WEA_KEY = {"clear": "clear", "overcast": "overcast", "partly cloudy": "partly_cloudy",
            "snowy": "snowy", "rainy": "rainy", "foggy": "foggy"}
ADVERSE_WEATHER = {"rainy", "snowy", "foggy"}


def build_sample_weights(names, attr_index, over):
    """Multiplicative oversampling weights: one factor from timeofday, one from
    weather. Scope widened 2026-09 from night/dawn-dusk/rainy to the full
    adverse-environment set, so snowy and foggy are oversamplable too. Any key
    absent from the config defaults to 1.0 (no change).

    Logs the expected epoch composition next to the raw pool composition, so an
    oversampling config that does nothing is visible in the log rather than
    discovered 15 hours later.
    """
    attrs = json.load(open(attr_index))
    unknown, ws = set(), []
    for n in names:
        a = attrs.get(n, {})
        tod, wea = a.get("timeofday"), a.get("weather")
        if tod is not None and tod not in _TOD_KEY:
            unknown.add(f"timeofday={tod}")
        if wea is not None and wea not in _WEA_KEY:
            unknown.add(f"weather={wea}")
        ws.append(over.get(_TOD_KEY.get(tod, "\\0"), 1.0) * over.get(_WEA_KEY.get(wea, "\\0"), 1.0))
    if unknown:
        raise SystemExit(f"FLAG: unrecognized attribute values {sorted(unknown)} in "
                         f"{attr_index} - the split was built with a different vocabulary. "
                         f"Fix before pretraining.")
    unused = set(over) - set(_TOD_KEY.values()) - set(_WEA_KEY.values())
    if unused:
        raise SystemExit(f"FLAG: oversample keys {sorted(unused)} match no attribute value "
                         f"and would silently do nothing. Valid keys: "
                         f"{sorted(set(_TOD_KEY.values()) | set(_WEA_KEY.values()))}")

    tot, n_img = sum(ws), len(names)
    sels = {
        "night": lambda a: a.get("timeofday") == "night",
        "dawn/dusk": lambda a: a.get("timeofday") == "dawn/dusk",
        "rainy": lambda a: a.get("weather") == "rainy",
        "snowy": lambda a: a.get("weather") == "snowy",
        "foggy": lambda a: a.get("weather") == "foggy",
        "adverse_weather": lambda a: a.get("weather") in ADVERSE_WEATHER,
        "night_or_adverse": lambda a: (a.get("timeofday") == "night"
                                       or a.get("weather") in ADVERSE_WEATHER),
    }
    comp = {}
    for key, sel in sels.items():
        comp[key] = {
            "raw": round(sum(1 for n in names if sel(attrs.get(n, {}))) / n_img, 4),
            "sampled": round(sum(w for n, w in zip(names, ws) if sel(attrs.get(n, {}))) / tot, 4),
        }
    return ws, comp
'''

OLD_FN_START = "def build_sample_weights(names, attr_index, over):"
OLD_FN_END = "\n\ndef param_groups("

EDITS = [
    # (file, old, new, required_count)
    ("jepa_distill/teacher.py",
     'ckpt = torch.load(checkpoint, map_location="cpu")',
     'ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)', 1),
    ("jepa_distill/train_distill.py",
     'ck = torch.load(resume_from, map_location="cpu")',
     'ck = torch.load(resume_from, map_location="cpu", weights_only=False)', 1),
    ("jepa_pretrain/train_ijepa.py",
     'ck = torch.load(latest, map_location="cpu")',
     'ck = torch.load(latest, map_location="cpu", weights_only=False)', 1),
    # student construction: YOLOv5 yaml + nc=10 -> YOLOv11 yaml + nc=5, and pass student_init
    ("jepa_distill/train_distill.py",
     '''    student = BackboneStudent(cfg.get("model_yaml", "yolov5/models/yolov5s.yaml"),
                              nc=cfg.get("nc", 10), teacher_dim=teacher.embed_dim).to(device)''',
     '''    student = BackboneStudent(cfg.get("model_yaml", "yolo11s.yaml"),
                              nc=cfg.get("nc", 5), teacher_dim=teacher.embed_dim,
                              student_init=cfg.get("student_init", "random")).to(device)''', 1),
]


def main():
    changed, skipped = [], []
    for rel, old, new, count in EDITS:
        p = REPO / rel
        src = p.read_text()
        if new in src and old not in src:
            skipped.append(f"{rel}: already patched")
            continue
        n = src.count(old)
        assert n == count, f"{rel}: expected {count} occurrence(s) of\n{old!r}\nfound {n}"
        p.write_text(src.replace(old, new))
        changed.append(f"{rel}: {old.splitlines()[0][:60]}...")

    # build_sample_weights: whole-function replacement
    p = REPO / "jepa_pretrain/train_ijepa.py"
    src = p.read_text()
    if "ADVERSE_WEATHER" in src:
        skipped.append("jepa_pretrain/train_ijepa.py: build_sample_weights already widened")
    else:
        i = src.index(OLD_FN_START)
        j = src.index(OLD_FN_END, i)
        p.write_text(src[:i] + NEW_WEIGHTS_FN.rstrip("\n") + src[j:])
        changed.append("jepa_pretrain/train_ijepa.py: build_sample_weights widened to "
                       "snowy/foggy + raw-vs-sampled logging")

    for c in changed:
        print("PATCHED  " + c)
    for s in skipped:
        print("skip     " + s)
    print(f"\n{len(changed)} edit(s) applied, {len(skipped)} already in place.")


if __name__ == "__main__":
    main()
