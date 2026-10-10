# Phenikaa Campus Courier — pipeline v6 (val end-to-end 64.80%)

Dự đoán bước đi đầu tiên (UP/DOWN/LEFT/RIGHT) cho 10 robot từ ảnh bản đồ + mission tiếng Việt.
Điểm = macro accuracy 10 robot. Baseline majority 26.6%.

## Cấu trúc
- `oracle/policy_v5.py` — simulator Dijkstra + turn-cost + tie-break theo robot + greedy R9. Per-robot costs fit trên train-oracle, có conditional splits (R1-style, R3-goal, R5-style). Xem REVIEW chi tiết.
- `nlp/` — `train_phobert_v3.py` (PhoBERT-base-v2 multitask) + `nlp_rules.py` (rule gref_kind + anchor từ alias mined-train, single-instance, negation). Missions: `v3_validation/test_missions.json` (PhoBERT) + rules áp lúc infer.
- `cv/` — FRCNN nodes/legend, ResNet18 landmark/edge, MobileNet robot/weather/stairs/oneway, Siamese legend-swap. Models `.pt` (git-ignored, train local).
- `pipeline/infer_val_v5.py` — chấm val end-to-end (CV + NLP hybrid + v5 + night-detect). `infer_test_v5.py` — sinh test.
- `predictions.json` — bản nộp test 12000 (v6). `predictions_v5_val.json` — val 3000 (64.80%).

## Số đo thực (không ước lượng)
- Oracle (true graph + true mission): **train 77.82% / val 72.93%**.
- True-graph + improved NLP: **66.97%**. Pred-graph + true mission: 66.33%. → NLP cost +6.0, CV cost +2.2 (ablation A/B/C/D).
- End-to-end val (CV+NLP+v6): **64.80%** (v3 cũ 58.07%). Per-robot: R0 .777, R1 .630, R2 .620, R3 .640, R4 .587, R5 .707, R6 .660, R7 .607, R8 .587, R9 .667.

## Chạy
```bash
# val end-to-end (cần models .pt + v3 missions)
CUDA_VISIBLE_DEVICES=1 python3 solution/pipeline/infer_val_v5.py
# test -> predictions.json
CUDA_VISIBLE_DEVICES=1 python3 solution/pipeline/infer_test_v5.py && cp solution/predictions_v5_test.json solution/predictions.json
```

## Ghi chú trung thực
- Policy chưa crack triệt để (R0 failures 100% là ties; R1-detours không khớp feature nào đã thử; nhiều family/ML-rankers đã thử và thua hand-Dijkstra trên val). Ba conditional splits (R1-night→base, R3-goal-GLL→turns, R5-night→base) là gains lớn cuối cùng tìm được (+2.5% oracle).
- NLP rules (kind 87%, anchor 71%) thắng PhoBERT ở các trường đó; goal/via giữ PhoBERT. via_ref chưa xử lý (rule thử chỉ 40% val).
- Muốn 70% cần biến cost mới cho ties/detours hoặc retrain CV lớn — ngoài phạm vi bản này.
