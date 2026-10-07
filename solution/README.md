# Full dự án — chạy từ đầu tới predictions.json

## Cấu trúc
- `oracle/`: simulator Dijkstra + tie-break F>R>L>B, weights R0-R9, `val_oracle_true_nlp.py` (val 85.4%), `sim_all_oracle.py` (train 88.7%).
- `nlp/`: `nlp_tfidf_baseline.py` (goal 64%, urgent/fragile 84-86%), `train_phobert.py` (PhoBERT-base-v2 135M multitask, chạy GPU2), `review_val_tfidf_oracle.py` (val end-to-end 67.8%).
- `cv/`: `gen_crops_stats.py` (121k train edges), `train_edge.py` (EfficientNet-B0 + Siamese legend).
- `pipeline/infer_test.py`: sinh `predictions.json` 12k đúng format BTC.
- `predictions.json`: bản nộp v1 (majority, val 26.6%, an toàn). `predictions_v1.json`: thử nghiệm stat (val 25.6%, không dùng).
- `REVIEW.md`: số liệu 3 vòng review.

## Chạy
```bash
python solution/nlp/nlp_tfidf_baseline.py
CUDA_VISIBLE_DEVICES=2 python solution/nlp/train_phobert.py --epochs 5
python solution/cv/gen_crops_stats.py
CUDA_VISIBLE_DEVICES=2 python solution/cv/train_edge.py --epochs 8
python solution/pipeline/infer_test.py --data Phenikaa_Campus_Courier_2026/Phenikaa_Campus_Courier_2026/delivery_public --out solution/predictions.json
python Phenikaa_Campus_Courier_2026/Phenikaa_Campus_Courier_2026/starter/starter.py --data ... --out /tmp/pred.json  # check val
```

## Điểm hiện tại (val)
- Majority: 26.6%. Oracle TRUE NLP: 85.4%. Oracle + TF-IDF: 67.8%. Không CV: 25.6%.
- Tiếp: PhoBERT (goal 64%→85%+), crack R4/R5/R8, CV SiameseDialog để lên 75%+ end-to-end.
