# Project Plan & Live Status

*The map of the whole experiment. Update the status column as runs finish.*
*(Roles of the other docs: RUNBOOK.md = how to run each stage; REPO_SURVEY.md =
flags/history; STAGE2_DISTILL_DESIGN.md = signed-off stage-2 spec;
PROJECT_EXPLAINED.md = plain-words narrative.)*

## The question we are answering

> Does an I-JEPA-derived backbone initialization beat standard COCO
> pretraining for night / adverse-weather vehicle detection on BDD100K —
> especially when labels are scarce?

## The deliverable: one table

| arm (backbone init) | 100% labels | 10% labels |
|---|---|---|
| COCO (baseline)     | ✅ done     | ⬜ |
| T1 (generic I-JEPA ViT-H, distilled) | ⬜ | ⬜ |
| T2 (BDD-night-pretrained I-JEPA ViT-B, distilled) | ⬜ | ⬜ |

Each cell = mAP50 / mAP75 / mAP50-95 on BDD-val, sliced overall / night /
dawn-dusk / rainy, all scored by `evaluation/eval_detections.py` (one referee).
Reference row: friend's best.pt re-scored on our protocol (⬜ pending).

## Dependency graph (what unlocks what)

```
                       ┌──────────────────────────────────────────┐
 data plumbing ✅ ──►  │ baseline fine-tune ✅ ──► baseline@10% ⬜ │
       │               └──────────────────────────────────────────┘
       │               ┌──────────────────────────────────────────┐
       ├─────────────► │ T1: teacher download ⬜ ► distill ⬜      │
       │               │     ► assemble ⬜ ► fine-tune ⬜ ► @10% ⬜ │
       │               └──────────────────────────────────────────┘
       │               ┌──────────────────────────────────────────┐
       └─────────────► │ T2: I-JEPA pretrain ⬜ ► distill ⬜       │
                       │     ► assemble ⬜ ► fine-tune ⬜ ► @10% ⬜ │
                       └──────────────────────────────────────────┘
 T1 and T2 are independent of the baseline and of each other EXCEPT:
 run T1's distill before T2's (T1 validates the distillation machinery
 with a known-good teacher, so bugs surface cheaply).
 Final: score every arm on val once ► make_results_table ► write-up.
```

## Milestones (brief §10) ↔ RUNBOOK sections ↔ status (2026-07-19)

| # | Milestone | RUNBOOK | Status |
|---|---|---|---|
| 1 | Repo survey, requirements, runbook skeleton | — | ✅ 07-12 |
| 2 | Data plumbing: converter, attr index, committed splits | §3 | ✅ 07-12 (code) / ✅ server run |
| 3 | Baseline arm (COCO-init, friend's recipe) | §4 | ✅ **07-19: val overall 0.623 / night 0.593 / dd 0.660 / rainy 0.647 mAP50** (27.6 h) |
| 3b | Re-score friend's best.pt with our evaluator | §4.4 | ⬜ next, ~10 min |
| 4 | T1: distill ViT-H → fine-tune → row 2 | §5 | ⬜ code ready; smoke test then ~2-3 h + ~28 h |
| 5 | T2: I-JEPA pretrain → distill → fine-tune → row 3 | §6 | ⬜ code ready; ~10-15 h + ~1 h + ~28 h |
| 6 | Label-fraction grid: the three @10% arms | §7.1 | ⬜ ~3-4 h each (Kaggle-suitable) |
| 7 | Final val scoring of remaining arms + table + write-up support | §7.2-7.3 | ⬜ baseline's val row already done |

## Standing rules (the ones that protect the result)

1. Arms differ ONLY in backbone init; everything else byte-identical.
2. One evaluator for all arms. YOLOv5's built-in val printout ≠ table numbers.
3. Val is read-only: no decision may be based on a val number until the table
   is final. Development decisions happen on modelsel.
4. Never delete run outputs; rename aborted runs.
5. FLAG, don't assume. Open: FLAG 2 (kaggle CLI - moot if dataset already up),
   FLAG 4 (64,520 vs 69,863 train labels - CHECK ON SERVER + friend's copy).

## Night-shift accounting (professor's sharing policy)

Long jobs run overnight in resumable chunks (RUNBOOK top section). At the
baseline's measured pace (~11 h/night, ~17 min/epoch):
T1 fine-tune ≈ 3 nights · T2 pretrain ≈ 1–2 nights · T2 fine-tune ≈ 3 nights ·
distills fit inside a night alongside nothing else · 10% arms → Kaggle days.
Everything remaining ≈ 8–10 calendar nights of GPU, announced in the group line.

## Note on execution order

RUNBOOK sections are grouped by STAGE, not strict chronology. Interleaving is
expected and fine (e.g. baseline was val-scored early; the 10% arms may run on
Kaggle while a 100% arm runs on the A40). The dependency graph above is the
true ordering constraint; the standing rules are the guardrails.
