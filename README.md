# Phenika-AI — Phenikaa Campus Courier v2

> Robot sẽ đi hướng nào? Bài thi AI Hackathon kết hợp CV + NLP + học chiến thuật từ dữ liệu.

## 1. Yêu cầu đề bài

### 1.1. Input / Output

Mỗi dòng dữ liệu:

| Trường | Ý nghĩa |
|---|---|
| `id` | mã dòng, vd `test-00012-R7` |
| `robot_id` | 0–9 |
| `image` | đường dẫn ảnh PNG/JPEG, tính từ thư mục split |
| `mission` | yêu cầu tiếng Việt |

Output mỗi dòng: **1 số nguyên** `0=UP, 1=DOWN, 2=LEFT, 3=RIGHT` — hướng tuyệt đối trên ảnh (theo hàng/cột lưới đường, UP = sang giao lộ hàng phía trên). Không phụ thuộc hướng mặt robot.

Mỗi **cảnh (scene)** = 1 ảnh + 1 mission chung cho đúng 10 dòng liên tiếp R0→R9. Các cảnh độc lập: không lịch sử, không phần thưởng, không tương tác.

### 1.2. Trong ảnh có gì?

- **Lưới giao lộ:** 5–9 hàng/cột (train chủ yếu 5–8, val/test lớn/dày hơn). Có ô khuyết (hồ nước), có cặp kề không nối đường, tâm lệch pixel, xoay nhẹ/mờ/nén JPEG.
- **Trạng thái đoạn đường:** `normal / crowded (đông) / covered (mái che) / closed (đóng)`.
- **Ký hiệu thêm:** `stairs` (bậc thang, chỉ R4 đi được), `oneway` (1 chiều theo mũi tên), robot `R` + mũi hướng mặt, icon thời tiết `rain/dry` (nắng/mây/mưa).
- **Địa điểm:** 5–9/cảnh, thuộc 10 loại. Một loại có thể xuất hiện 2 lần:

| key `scenes.json` | ký hiệu | tên |
|---|---|---|
| `library` | TV | Thư viện |
| `dorm` | KTX | Ký túc xá |
| `sports` | TT | Nhà thể thao |
| `clinic` | YT | Trạm y tế |
| `canteen` | CA | Căn tin |
| `parking` | XE | Bãi xe |
| `lecture` | GĐ | Giảng đường |
| `lab` | TN | Phòng thí nghiệm |
| `office` | HC | Phòng hành chính |
| `gate` | CT | Cổng trường |

- **4 kiểu vẽ:** `classic / night / print / sketch` — khác bảng màu, cách vẽ trạng thái đường, cách vẽ địa điểm, font/vị trí chú giải, kích thước ảnh.
- **Chú giải (legend) của chính ảnh mới đúng.** Màu/nét `normal/crowded/covered` có thể bị hoán đổi (`road_look`, vd `{"crowded":"covered"}`). PNG palette → đọc bằng `Image.open(p).convert("RGB")`.

### 1.3. Trong yêu cầu có gì?

- **goal (nơi giao):** tên thẳng / tên khác (“nơi mượn giáo trình”, “khu nội trú”) / chỉ người nhận (“thủ thư đang chờ”).
- **via (điểm ghé, optional):** phải tới via trước rồi mới tới goal. Thứ tự nhắc trong câu có thể đảo (“Trước khi mang tới A, nhớ ghé B”).
- **Tham chiếu không gian** (khi 1 loại có 2 bản): `north/south/west/east` (Bắc = trên ảnh theo mũi tên B) hoặc `near/far X` (chim bay tới landmark đơn X). Chỉ dùng ref rõ ràng (cách ≥2 hàng/cột hoặc chênh ≥1 ô tới X).
- **`urgent`, `fragile` có phủ định** (“không gấp”, “hàng chắc chắn, không dễ vỡ”).
- **Địa điểm gây nhiễu** (“không cần ghé…”, “đừng nhầm với…”).
- Câu **không dấu / sai chính tả**. Val/test có **diễn đạt mới chưa từng thấy ở train**.

### 1.4. 10 robot (`robots.json`)

| id | tên | gợi ý |
|---|---|---|
| 0 | Tia Chớp | ít đoạn đường nhất |
| 1 | Yên Tĩnh | ngại crowded |
| 2 | Mái Hiên | thích covered, chịu vòng |
| 3 | Mây Mưa | theo thời tiết |
| 4 | Sơn Dương | có chân, duy nhất qua stairs |
| 5 | Thẳng Tắp | ít đổi hướng, heading quan trọng |
| 6 | Hỏa Tốc | theo urgent |
| 7 | Nâng Niu | theo fragile |
| 8 | Lề Phải | thói quen riêng khi đổi hướng |
| 9 | Tham Lam | chỉ nhìn 1 bước |

Quy tắc chung (công khai):
1. Cấm vào `closed`, cấm ngược `oneway`, cấm `stairs` trừ R4.
2. Có via → via trước, goal sau.
3. Có ref không gian rõ → mọi robot tới đúng bản đó. Không rõ + 2 bản → theo chiến thuật riêng.
4. Chiến thuật cố định + tất định + **chung 1 luật phá hòa theo hướng mũi robot** (không công bố, phải tự suy từ label).
5. Trọng số/điều kiện/luật phá hòa phải học từ dữ liệu.

### 1.5. Dữ liệu, nộp bài, luật

```
delivery_public/
  robots.json
  train/       observations.json labels.json scenes.json dataset_card.json images/ # 2000 cảnh / 20000 dòng
  validation/  observations.json labels.json scenes.json dataset_card.json images/ # 300 cảnh / 3000 dòng, profile hard
  test/        observations.json sample_submission.json dataset_card.json images/  # 1200 cảnh / 12000 dòng
starter/
  starter.py   # đọc data, chấm val, baseline majority, vẽ scenes (--show 0)
```

- `labels.json`: list int, cùng thứ tự `observations.json`.
- `scenes.json` (chỉ train/val): chú thích phụ để train riêng CV/NLP: `scene_id,image,width,height,grid,nodes[{rc,xy}],edges[{a,b,status,stairs,oneway_to}],landmarks[{type,rc}],robot{rc,heading},weather,mission{text,goal,goal_ref{kind,anchor,rc},via,via_ref,urgent,fragile},legend[{kind,text,swatch,label}],weather_box,road_look,degradation,style`.
- Test không có `scenes.json`, map mới hoàn toàn, chỉ dùng ảnh + mission + `robot_id`. Cấm gán nhãn tay / đọc test để thêm từ điển-luật / train (kể cả self-supervised) trên test.
- Nộp `predictions.json`: list int 0–3 cùng độ dài/thứ tự test. Sai format bị loại.
- Điểm = **macro accuracy trung bình 10 robot**. Test trộn cảnh chấm + không chấm; public ~30% cảnh, final là phần còn lại, lấy bài public cao nhất.
- Baseline majority theo robot: **val macro ~0.266** (đã chạy `starter.py`).
- Bất đồng robot: train 89.1% cảnh, val 97.0% cảnh → bắt buộc phân biệt robot.
- Giới hạn: tổng params model lúc infer **≤200M**, được dùng pretrained công khai (PhoBERT, ResNet, MobileNet), **cấm gọi API AI ngoài**.

Chi tiết gốc: `Phenikaa_Campus_Courier_2026/Phenikaa_Campus_Courier_2026/DE_BAI.md`.

---

## 2. Hướng giải quyết

Tách 3 khâu như đề gợi ý, đo từng khâu trên val bằng `scenes.json`:

```
ảnh ──CV──► graph (nodes, edges, landmarks, robot, weather + đọc legend) ─┐
                                                                          ├─► simulator chiến thuật robot_id ─► hướng đi
yêu cầu ──NLP──► (goal, via, ref, urgent, fragile) ────────────────────────┘
```

### Bước 1 — Reverse-engineer chiến thuật (làm trước, oracle upper-bound)
Dùng `scenes.json` train như graph chuẩn, viết simulator Dijkstra/A*:

- R0: `cost=1`, BFS.
- R1: `1 + w_crowded`, R2: `1 - w_covered`, R3: 2 bộ trọng số theo `dry/rain`.
- R4: mở `stairs`, còn lại `stairs=closed`.
- R5: `1 + w_turn*đổi_hướng`, R6: 2 chế độ theo `urgent`, R7: 2 chế độ theo `fragile`.
- R8: ưu tiên rẽ phải (học thứ tự thẳng/phải/trái/quay đầu).
- R9: greedy 1 bước (kề gần goal/via nhất theo chim bay).
- Học grid-search trọng số + luật phá hòa chung theo heading (vd thẳng > phải > trái > sau) bằng cách khớp label train. So 10 robot cùng cảnh là tín hiệu mạnh nhất.

### Bước 2 — NLP parser
Multi-task trên `mission.text`: `goal(10) / via(11 gồm null) / urgent / fragile / goal_ref+via_ref (null/north/south/east/west/near/far + anchor)`, lọc nhiễu. Augment bỏ dấu + typo + paraphrase để chịu diễn đạt mới val/test.

### Bước 3 — CV parser (khó nhất, quyết định test)
- Nodes: heatmap tâm `xy` → gán hàng/cột.
- Edges: crop dọc cặp kề → classify `status(4) + stairs + oneway(3)`. So khớp với `swatch` legend của chính ảnh (Siamese/metric) để xử `road_look` swap.
- Landmarks/robot-heading/weather: crop tại node / `weather_box` → classify.
- Chịu 4 style + xoay/mờ/JPEG bằng augment + backbone nhẹ.

### Bước 4 — Tích hợp
`graph_CV + mission_NLP → simulator robot → bước đầu tiên → map chênh `rc` thành UP/DOWN/LEFT/RIGHT`. Đánh giá end-to-end macro trên val, rồi infer test → `predictions.json`.

---

## 3. Model sẽ dùng (tổng ≤200M)

| Khâu | Model | Params ước tính | Vai trò |
|---|---|---|---|
| NLP | **PhoBERT-base-v2** + 6 head phân loại (goal, via, urgent, fragile, ref-kind, anchor) | ~135M | Hiểu tiếng Việt có/không dấu, phủ định, ref, nhiễu. Fine-tune trên `scenes[].mission`. Fallback nhẹ: TF-IDF+LogReg/BiLSTM nếu thiếu GPU |
| CV-nodes | **UNet + MobileNetV3-Small encoder** (heatmap 1 kênh) | ~6M | Tìm tâm giao lộ `xy` |
| CV-edges/status | **MobileNetV3-Large** chia sẻ backbone + 3 head (status4, stairs1, oneway3) + nhánh **Siamese** so edge-crop vs legend-swatch | ~8M | Đọc trạng thái đường, chịu swap chú giải |
| CV-landmark/robot/weather | **MobileNetV3-Large** (hoặc ResNet18) multi-head: landmark11 (10+none), robot-heading4, weather2 | ~8M | Nhận địa điểm, vị trí+hướng robot, thời tiết |
| Policy | **Dijkstra/A* + trọng số học được** (không phải neural, 0 params) hoặc **GNN 2-layer** tiny (<1M) nếu muốn học end-to-end | ~0–1M | Mô phỏng 10 chiến thuật, phá hòa theo heading |

Tổng infer: ~135 + 6 + 8 + 8 ≈ **157M < 200M** ✅. Không gọi API ngoài. Toàn bộ train trên train (+val để early-stop), cấm dùng test.

**Lộ trình:** simulator oracle → NLP → CV → tích hợp → pseudo-check val macro → sinh test `predictions.json`.
