#!/usr/bin/env python3
"""Turn slice_scores_noisefloor.csv into a verdict: is the T1-vs-BASE gap real?

THE PROBLEM THIS SOLVES
-----------------------
Every arm in this project is n=1. A single number like "T1 is 0.66 points below
BASE" is meaningless without knowing how much a number like that moves for
reasons that have nothing to do with the arm. Retraining BASE at a second seed
would answer that properly, but it costs ~2 nights of shared A40.

The cheap substitute: `save_period: 10` left four checkpoints of the SAME
t1_jepa_100 run - epoch 80, epoch 90, best, last. Same weights lineage, same
data, same seed, same recipe. They differ only in which epoch they came from,
and all four sit in the mosaic-free tail (close_mosaic: 20 -> mosaic off from
epoch 80). Their spread is therefore an estimate of run-internal wobble.

    if |T1_best - BASE|  <=  spread(T1 checkpoints):   the gap is NOT a finding
    if |T1_best - BASE|  >   spread(T1 checkpoints):   the gap survives, tentatively

HONEST CAVEAT, state it in the write-up
---------------------------------------
This is a WITHIN-run spread, not a BETWEEN-run (seed) spread. Between-seed
variance is normally LARGER, because it includes different initialisation of
the head, different augmentation draws and different data order. So this is a
*lower bound* on the true noise floor: a gap that fails this test is definitely
not a finding, but a gap that passes it is not thereby proven. Do not describe
this as a seed study. The only thing that settles it is a repeated BASE run
(MASTER-RECORD.md section 4.2).

Also: epoch 80 sits exactly at the mosaic-off transition and the model is still
genuinely improving after it, so spread computed including e80 mixes real
training progress with noise. The script reports both the 4-checkpoint spread
and the tighter e90/best/last spread; prefer the tighter one as the noise floor
and treat the wider one as context.

Usage:
  python evaluation/noise_floor_report.py                       # defaults below
  python evaluation/noise_floor_report.py --csv results/slice_scores_noisefloor.csv
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

HEADLINE = ["overall", "day", "night", "dawndusk", "clear", "rainy", "snowy",
            "adverse", "night_adverse", "day_adverse", "night_clear"]

BASE_ARM = "base_100"
T1_MAIN = "t1_best"
T1_ALL = ["t1_best", "t1_last", "t1_e90", "t1_e80"]     # widest spread
T1_TIGHT = ["t1_best", "t1_last", "t1_e90"]             # converged-tail spread
EXTRA = ["set2_100", "m3_lowlight_irfs"]                # scored for other reasons


def load(csv_path):
    """-> d[arm][slice][class][metric] = value (float or None)"""
    d = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    meta = {}
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            v = r["value"]
            d[r["arm"]][r["slice"]][r["class"]][r["metric"]] = (
                None if v in ("", "None") else float(v))
            # Only the class=='all' rows carry the SLICE-level reportable flag.
            # Per-class rows AND it with the >=800-instance rule, so reading them
            # here made every slice inherit the last class's flag (bike, almost
            # always under 800) and flagged even `overall` as too few images.
            if r["class"] == "all":
                meta[r["slice"]] = (int(r["n_images"]), r["reportable"] == "True")
    return d, meta


def spread(d, arms, sl, cls, metric):
    vals = [d[a][sl][cls].get(metric) for a in arms if a in d]
    vals = [v for v in vals if v is not None]
    return (max(vals) - min(vals)) if len(vals) > 1 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/slice_scores_noisefloor.csv")
    ap.add_argument("--class-name", default="all")
    args = ap.parse_args()

    path = Path(args.csv)
    if not path.is_absolute():
        path = ROOT / path
    if not path.exists():
        raise SystemExit(f"FLAG: no csv at {path} - has score_slices.py finished?")

    d, meta = load(path)
    present = [a for a in T1_ALL if a in d]
    if len(present) < 2:
        raise SystemExit(f"FLAG: need >=2 T1 checkpoints to measure spread, found {present}")
    cls = args.class_name

    print(f"csv: {path}")
    print(f"arms: {', '.join(d)}")
    print(f"T1 checkpoints used for the noise floor: {', '.join(present)}")
    print(f"class: {cls}\n")

    for metric in ("mAP50", "mAP75"):
        print("=" * 96)
        print(f"{metric}  -  is the T1 vs BASE gap bigger than the run's own wobble?")
        print("=" * 96)
        hdr = (f"{'slice':<15}{'BASE':>9}{T1_MAIN:>10}{'delta':>9}"
               f"{'floor(e90+)':>13}{'floor(all4)':>13}   verdict")
        print(hdr)
        print("-" * len(hdr))

        for sl in HEADLINE:
            if sl not in meta:
                continue
            b = d[BASE_ARM][sl][cls].get(metric)
            t = d[T1_MAIN][sl][cls].get(metric)
            if b is None or t is None:
                continue
            delta = (t - b) * 100
            tight = spread(d, T1_TIGHT, sl, cls, metric)
            wide = spread(d, T1_ALL, sl, cls, metric)
            tight_pts = tight * 100 if tight is not None else None
            wide_pts = wide * 100 if wide is not None else None

            # The floor is a LOWER bound on true noise, so clearing it by a hair
            # is not enough. Ratio bands, not a threshold:
            #   <=1x   inside noise, not a finding
            #   1-2x   weak - would not survive a real seed study
            #   2-4x   holds
            #   >4x    holds clearly
            ratio = (abs(delta) / tight_pts) if tight_pts else None
            if ratio is None:
                verdict = "n/a"
            elif ratio <= 1.0:
                verdict = "INSIDE noise - not a finding"
            elif ratio <= 2.0:
                verdict = f"WEAK ({ratio:.1f}x floor) - do not headline"
            elif ratio <= 4.0:
                verdict = f"holds ({ratio:.1f}x floor)"
            else:
                verdict = f"HOLDS CLEARLY ({ratio:.1f}x floor)"

            n_img, reportable = meta[sl]
            flag = "" if reportable else "  [too few images]"
            print(f"{sl:<15}{b:>9.4f}{t:>10.4f}{delta:>+9.2f}"
                  f"{(tight_pts if tight_pts is not None else float('nan')):>13.2f}"
                  f"{(wide_pts if wide_pts is not None else float('nan')):>13.2f}   {verdict}{flag}")
        print()

    # ---- the question four training recipes could not move ----
    print("=" * 96)
    print("car day->night mAP75   (BASE/M1/M2/M3 all sat at -9.6 to -9.8)")
    print("=" * 96)
    for a in d:
        day = d[a]["day"]["car"].get("mAP75")
        night = d[a]["night"]["car"].get("mAP75")
        if day is None or night is None:
            continue
        print(f"  {a:<20} day {day:.4f}  night {night:.4f}   delta {(night-day)*100:+.2f} pts")

    # ---- the other arms, scored for their own reasons ----
    for extra in EXTRA:
        if extra not in d:
            continue
        print("\n" + "=" * 96)
        print(f"{extra} vs {BASE_ARM}  -  mAP50, class '{cls}'")
        print("=" * 96)
        for sl in HEADLINE:
            b = d[BASE_ARM][sl][cls].get("mAP50")
            e = d[extra][sl][cls].get("mAP50")
            if b is None or e is None:
                continue
            print(f"  {sl:<15}{b:>9.4f}{e:>10.4f}{(e-b)*100:>+9.2f}")

    print("\nREMINDER: this is a within-run spread, a LOWER BOUND on the true noise")
    print("floor. Between-seed variance is larger. A gap that fails this test is not")
    print("a finding; a gap that passes it is not proven. See MASTER-RECORD.md 4.2.")


if __name__ == "__main__":
    main()
