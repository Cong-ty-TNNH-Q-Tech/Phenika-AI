# Kế hoạch Courier v3: hướng tới nhóm đầu

## Ràng buộc không được phá vỡ

- Chỉ dùng `train` để fit luật, alias, mô hình và hyperparameter; dùng
  `validation` để chọn phiên bản cuối.
- Không đọc `test/observations.json` hoặc ảnh test khi phát triển luật/từ điển.
  Test chỉ được mở bởi lệnh inference cuối để viết `predictions.json`.
- Chấm theo macro accuracy 10 robot.  Mọi split nội bộ phải group theo `scene_id`
  (10 dòng của cùng cảnh luôn ở cùng một fold).
- Tổng tham số lúc infer không quá 200M.

## Sự thật của bộ private v3

- Data private khác vòng công khai. Tuy nhiên đã xác minh trực tiếp rằng
  `policy_v5.py` trong commit `7bdf90e` đã được fit cho chính bộ v3 này:
  train 77.815%, validation 72.933%. Nhận định trước rằng toàn bộ repo chỉ
  chứa policy v2 là sai; dùng kết quả đo này làm baseline.
- Train có 2,000 scenes; validation hard có 300; test có 1,200. Validation có
  map lớn/dày hơn, nhiều via, reference và ảnh nhiễu hơn.
- `goal_ref` còn có `north_most`, `south_most`, `west_most`, `east_most`,
  `anchor_near`; trước khi chạy policy phải resolve chúng trên graph đã đọc.

## Cổng đo bắt buộc

Đánh giá theo cùng 4 setting. Mỗi thay đổi chỉ được nhận nếu tốt hơn trên
validation và không làm một robot tụt mạnh.

| Setting | Input graph | Input mission | Mục đích |
|---|---|---|---|
| P | thật | thật | trần policy; reverse-engineer robot |
| N | thật | dự đoán | đo phần NLP |
| V | dự đoán | thật | đo phần CV |
| E2E | dự đoán | dự đoán | điểm để quyết định submission |

Mốc ship nên là E2E validation xấp xỉ 0.88--0.90, không chỉ tăng public
leaderboard. Top 3 hiện quanh 0.904 public nên không có cơ sở trung thực để
hứa thứ hạng khi chưa qua cổng này.

## Thứ tự triển khai

1. **Oracle policy trước.** Hoàn thiện `solution/private/oracle_xgb.py` thành
   benchmark cho graph/mission thật. Với từng robot, grid-search cost của
   crowded/covered, stairs, weather, urgent/fragile, turn penalty và toàn bộ
   tie-break tương đối/tuyệt đối. Fit điều kiện chỉ bằng train folds; chọn qua
   validation. R9 phải có family greedy riêng, không ép vào Dijkstra.
2. **NLP.** Train PhoBERT-base đa head trên data v3: `goal`, `via`,
   `urgent`, `fragile`, `goal_ref.kind/anchor`, `via_ref.kind/anchor`.
   Bắt buộc xuất *cả* via-ref (pipeline cũ làm mất nó). Augment không dấu,
   typo, correction “không phải X mà là Y”, negation và thứ tự via/goal.
3. **CV graph.** Giữ checkpoint v3 hiện có để kiểm tra thay đổi trong thời gian
   ngắn; huấn luyện detector chung cho budget tham số. Node keypoint detector
   -> affine/RANSAC grid; detector legend; Siamese edge-vs-swatch để chống
   `road_look` swap; crop classifier landmark/robot-heading/weather/oneway.
   Augment rotate, blur, JPEG theo `degradation` của train. Không tái sử dụng
   checkpoint mới phải thắng E2E validation trước khi thay thế.
4. **Ensemble có kiểm chứng.** Chỉ ensemble những model có lỗi khác nhau, ví
   dụ graph-CV model A/B + policy candidates. Lưu OOF scene-level predictions
   để lựa chọn theo confidence, không vote mù.
5. **Nộp bài.** Giữ submission baseline đã chấm và chỉ thay bằng phiên bản
   thắng E2E validation. Tối đa 5 lượt/ngày: không dùng public score để viết
   luật theo test; dùng các lượt cho 1--2 ứng viên đã được chọn trước.

## Tình trạng hiện tại

- Data v3 đã giải nén ở `data/Phenikaa_Campus_Courier_2026_v3/` và bị git-ignore.
- Starter baseline validation: **0.2490** macro.
- Policy v5 (graph/mission thật): **0.72933** validation.
- Ranker hướng/điểm ghé + prior + greedy features mới: **0.78100** validation
  trên graph/mission thật. Đây là oracle, không phải E2E.
- NLP trên graph thật: **0.66300 -> 0.67133** với flags/anchor đã sửa.
- Style classifier train-only: **1.000** validation (300 ảnh).
- Refiner trên prediction CV đã lưu: **0.64667 -> 0.65367** validation.
  Artifact test mới: `artifacts/private/predictions_refined_test.json`, đổi
  340/12000 nhãn ở R1/R5/R8, giữ nguyên file submission gốc.
- CI bootstrap của delta refiner: `[0.00100, 0.01333]`, group theo scene.
  Khoảng này có điều kiện trên lựa chọn model bằng chính validation, không
  phải một kiểm định độc lập hay bảo đảm điểm test tăng.
- CV hiện có: **123,279,178** tham số; cộng PhoBERT ~135M là ~258.28M,
  vượt budget nếu cả hai được tính vào pipeline tạo predictions.
- Detector MobileNet chung thay hai ResNet detector: **18,985,379** tham số;
  tổng pipeline ước tính còn **194,534,370**. Mã train/infer đã có; checkpoint
  mới chưa được huấn luyện, không dùng nó để tuyên bố kết quả hiện tại.
- E2E ranker/CV mới chưa chạy: local thiếu 10 checkpoint CV; SSH server chưa
  có xác thực. Môi trường local đã có TorchVision 0.25, PyTorch 2.10 CPU.
