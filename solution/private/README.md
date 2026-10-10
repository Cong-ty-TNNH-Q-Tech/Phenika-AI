# Private v3 optimization artifacts

Mọi lệnh dưới đây chạy từ root `Phenika-AI`. Data local nằm ở
`data/Phenikaa_Campus_Courier_2026_v3/delivery_public`.

## File dùng ngay trong lượt nộp thử

[results/predictions_refined_test.json](results/predictions_refined_test.json)
là bản được lưu trên GitHub; bản sinh local nằm ở
`artifacts/private/predictions_refined_test.json`: 12000 integer, chỉ đổi R1,
R5 và R8 so với `solution/predictions.json`. Validation 64.6667% -> 65.3667%.
Đây là ứng viên tăng điểm nhỏ, chưa đủ mức cạnh tranh top 3. Refiner dùng
prediction của các robot khác làm candidate, vì vậy chưa bảo đảm candidate
được chọn là cạnh hợp lệ nếu không có graph dự đoán. Không thay file gốc.
Manifest SHA256 và metrics nằm trong [results/](results/) và thư mục artifact.
Model đã train nằm ở `artifacts/` trên máy local, không được commit; dùng các
lệnh tái lập bên dưới để tạo lại. Data cuộc thi và các checkpoint CV/PhoBERT
cần được cung cấp riêng, không nằm trong bản clone GitHub.

## Chạy CV/policy mới sau khi có checkpoint

Copy 10 file `.pt` hiện tại từ server vào `solution/cv/` hoặc thư mục khác:
`nodes_frcnn`, `legend_frcnn`, `landmark_resnet18`, `robot_mobilenet`,
`weather_img`, `stairs_mobilenet`, `oneway_dir`, `edge_resnet18`,
`edge_presence`, `siamese_mlp`.

```powershell
$courierData = 'data/Phenikaa_Campus_Courier_2026_v3/delivery_public'
.venv/Scripts/python.exe solution/private/server_pipeline.py --data $courierData --models solution/cv --out artifacts/live --policy artifacts/enriched/policy.joblib --style artifacts/private/style.joblib --anchor-aliases artifacts/private/anchor_aliases.json
.venv/Scripts/python.exe solution/private/select_policy.py --graphs artifacts/live/graphs_validation.json --labels "$courierData/validation/labels.json" --baseline artifacts/live/predictions_optimized_validation.json --out artifacts/live/selection.json
.venv/Scripts/python.exe solution/private/server_pipeline.py --data $courierData --models solution/cv --out artifacts/live --policy artifacts/enriched/policy.joblib --style artifacts/private/style.joblib --anchor-aliases artifacts/private/anchor_aliases.json --selection artifacts/live/selection.json --split test
```

Mặc định lượt validation dùng policy v5 đã sửa legal fallback và lưu đồng
thời predictions của ranker để chọn từng robot. Không cần chạy detector hai
lần để chọn. Test đọc selection đã cố định; tuyệt đối không fit vào test.

## Budget 200M

CV gốc có 123.28M tham số. Cộng PhoBERT ~135M có thể vượt 200M. Không coi
việc lưu NLP predictions ra JSON là cách giảm tham số của cả pipeline.
`joint_detector.py` thay hai detector ResNet bằng một MobileNet detector
chung, đưa tổng CV + PhoBERT ước tính về 194.53M.

```powershell
.venv/Scripts/python.exe solution/private/joint_detector.py --data $courierData --out artifacts/compact_cv --epochs 5 --pretrained --freeze-backbone
```

Thêm `--joint-detector artifacts/compact_cv/joint_detector_epoch5.pt` vào
lệnh validation/infer để thử checkpoint mới. Phải chọn epoch theo E2E
validation. Chưa có checkpoint joint được huấn luyện trong phiên này.
Local hiện chạy CPU; huấn luyện CV/PhoBERT đầy đủ sẽ không xong trong 30 phút.
GTX 1650 có mặt nhưng cần PyTorch CUDA và đủ dung lượng ổ để dùng nó.

## Huấn luyện từng module bằng đường dẫn local

`train_local.py` thực thi bản source trong bộ nhớ; không sửa đường dẫn Linux
trong file gốc, không ghi đè checkpoint có sẵn. Output mặc định:
`artifacts/local_checkpoints/`. Windows dùng workers=0.

```powershell
.venv/Scripts/python.exe solution/private/train_local.py landmark --data $courierData --epochs 4 --batch 16
.venv/Scripts/python.exe solution/private/train_local.py nlp --data $courierData --epochs 4 --batch 1
```

NLP cần thêm `transformers` và `sentencepiece`. Phải cài phiên bản tương thích
trong environment riêng. PhoBERT inference gốc đã sửa để xuất cả
`via_ref_kind`/`via_ref_anchor`; cached JSON hiện tại chưa có các trường đó.

## Tái lập các phép đo đã chạy

```powershell
python solution/private/test_policy.py
python solution/private/policy_ranker.py --data $courierData --out artifacts/enriched --rounds 400
python solution/private/style_fast.py --data $courierData
python solution/private/anchor_miner.py --data $courierData
python solution/private/evaluate_nlp.py --data $courierData
python solution/private/submission_refiner.py train --data $courierData
python solution/private/submission_refiner.py infer --data $courierData
```

Policy oracle, NLP true-graph và E2E refiner là ba phép đo khác nhau. Không
diễn giải 78.10% oracle như điểm end-to-end hoặc điểm leaderboard.
