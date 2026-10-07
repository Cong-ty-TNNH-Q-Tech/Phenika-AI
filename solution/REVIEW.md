# Review nhiều vòng — kết quả một phát ngon trong ngày

## Vòng 1 — Simulator oracle + R0
- R0 = BFS ít hops + phá hòa chung `F>R>L>B` theo heading.
- Clean no-via single-goal: **train 1021/1021 = 100%, val 79/79 = 100%**.
- Tie 30% số cảnh, label luôn trong tập shortest, chưa từng chọn Back khi có lựa chọn khác.
- File: `oracle/sim_r0_bfs.py`, `oracle/sim_tiebreak.py`.

## Vòng 2 — Weights R1-R9 (train oracle, graph chuẩn + mission chuẩn)
- R0 base=1: **100%** (kể cả goal 2 bản → chọn bản min-cost).
- R1 avoid crowded `1+5*crowded`: **100%**. R2 thích covered `covered=0.5`: **100%**.
- R3: mưa `covered=0.1` → 99.7%, khô `1+3*crowded` → 93.6%, overall R3 96.0%.
- R6: urgent=True base → 100%, urgent=False `1+5*crowded` → 92.8%, overall 95.6%.
- R7: non-fragile base → 100%, fragile=True hiện tốt nhất `avoid crowded` ~67% (agree R5 77% → nghi fragile=ít rẽ, chưa crack).
- R4 legged=True base → 78.2% (chưa crack), R5 turn-uniform → 69.5%, R8 turn-uniform → 74.4%.
- R9 greedy Manhattan tới via/goal + phá hòa F>R>L>B → **86.9%** (Euclid chỉ 72.3%).
- Tổng train oracle: **macro 88.7%**. Val oracle TRUE NLP: **macro 85.4%** (R0/R1/R2 100%, R3 95%, R6 96%, R7 86%, R9 78%, R8 71%, R5 65%, R4 61%).
- File: `oracle/sim_r1r2.py`, `oracle/sim_r9_greedy.py`, `oracle/sim_all_oracle.py`, `oracle/val_oracle_true_nlp.py`.

## Vòng 3 — NLP TF-IDF baseline (train 2000, val 300, review overfit)
- TF-IDF 8k (1,2)-gram + Logistic C=4:
- goal 63.7% (train 99.8%), via 68.7% (100%), urgent 84.0% (100%), fragile 86.0% (100%), goal_ref 63.7%, via_ref 81.7%.
- Val end-to-end (oracle graph + TF-IDF NLP): **macro 67.8%** vs baseline majority **26.6%** (x2.5).
- R0 79%, R1 82%, R2 78%, R3 76%, R6 72%, R7 64%, R9 63%, R8 60%, R5 54%, R4 49%.
- Kết luận: urgent/fragile ổn, goal/ref yếu do diễn đạt mới → cần PhoBERT + augment bỏ dấu/typo.
- File: `nlp/nlp_tfidf_baseline.py`, `nlp/review_val_tfidf_oracle.py`.

## Chốt hôm nay
- Đã có pipeline oracle chạy được + số val reviewable. CV full (YOLO/EfficientNet + Siamese legend) sang ngày mai.
- Tiếp theo: PhoBERT-base-v2 multitask, crack R4/R5/R8/R7-fragile, gen CV crops ~100k edge.
