# Stage 2 — Train-val window sweep runbook

> **Mục tiêu:** Chạy Stage 2 theo từng `window_size` trên các category thuộc
> split `train_val`, có thể chạy độc lập nhiều đợt rồi aggregate cộng dồn giống
> Stage 1. Stage 2 đo trade-off giữa window size và quality/FPS/RAM, sau đó chọn
> `N*`.

Spec: `docs/superpowers/specs/2026-05-08-stage2-window-sweep-design.md`.
Plan: `docs/superpowers/plans/2026-05-08-stage2-window-sweep.md`.

---

## 1. Stage 2 window semantics

Trong Stage 2, `window_size` được truyền vào inference dưới dạng
`--keep_window_maskmem={window_size}`. Tham số này là **phạm vi scan/retention
maskmem**, không phải kích thước memory bank của SAM 2/SAMURAI.

Các giá trị cố định trong Stage 2 hiện tại:

```text
num_maskmem = 7
enable_auto_promote = false (--no_auto_promote)
release_interval = 10
keep_window_pred_masks = 60
max_cache_frames = 60
```

Ý nghĩa:

- `num_maskmem=7` là kích thước mask memory bank của model, gồm frame 0
  conditioning frame và tối đa 6 non-conditioning maskmem frames.
- `window_size=N` chỉ giới hạn phạm vi tìm non-cond maskmem candidates. Khi xử
  lý frame `F`, SAMURAI chỉ scan các frame `F-1 ... F-N`.
- Nếu trong `N` frame gần nhất không đủ 6 non-cond maskmem hợp lệ, memory bank
  dùng ít hơn 6 non-cond frames; code không scan ngoài `N` để lấp đầy slot.
- Vì Stage 2 dùng `--no_auto_promote`, conditioning frame được giữ là frame 0.
  Các propagated frames còn lại nằm trong `non_cond_frame_outputs`.
- `release_old_frames()` giải phóng `maskmem_features` và `maskmem_pos_enc` của
  các non-cond frames có `frame_idx < current_frame_idx - keep_window_maskmem`.
  Frame 0 không bị xóa vì nằm trong `cond_frame_outputs`.
- Release chạy theo `release_interval=10` để giảm overhead, nên frame ngoài
  window có thể còn tensor tạm thời giữa hai mốc release. Tuy nhiên memory
  selection vẫn không scan tới các frame đó ở từng frame inference.
- `max_obj_ptrs_in_encoder=16` là giới hạn object-pointer tokens trong encoder,
  không phải kích thước maskmem bank. Stage 2 sweep chỉ thay đổi maskmem
  scan/retention window qua `keep_window_maskmem`.

Ví dụ với `window_size=75`, khi xử lý frame `F=100`:

```text
maskmem scan range: 99, 98, ..., 25
maskmem release boundary tại mốc release frame 100: xóa non-cond frame < 25
mask memory bank thực tế: frame 0 + tối đa 6 non-cond frames được chọn trong 99..25
```

Ví dụ với `window_size=6`, khi xử lý frame `F=20`:

```text
maskmem scan range: 19, 18, 17, 16, 15, 14
maskmem release boundary tại mốc release frame 20: xóa non-cond frame < 14
mask memory bank thực tế: frame 0 + tối đa 6 non-cond frames được chọn trong 19..14
```

---

## 2. Output và resume model

Stage 2 có 3 bước chính:

```text
scripts/stage2_run_batch.py
  -> metrics/stage2_lasot/{window_size}/stage2/{video}.csv
  -> results/stage2/{window_size}/{video}.txt

scripts/stage2_aggregate.py
  -> analysis/stage2/stage2_results.csv
  -> analysis/stage2/stage2_attribute_results.csv
  -> analysis/stage2/stage2_summary.json

scripts/stage2_select_n_star.py
  -> analysis/stage2/n_star_selection.json
```

`stage2_run_batch.py` luôn đọc video từ `splits/splits_v1.json` và mặc định
chỉ lấy split `train_val`. Với config hiện tại, mỗi category có 2 video
`train_val`, tổng cộng 140 video trên 70 categories.

Một cặp `(window_size, video)` được xem là hoàn thành khi cả 2 file sau tồn tại,
có số frame khớp, và CSV có cột `maskmem_bytes` hợp lệ:

```text
metrics/stage2_lasot/{window_size}/stage2/{video}.csv
results/stage2/{window_size}/{video}.txt
```

`maskmem_bytes` được ghi khi batch runner gọi `main_inference.py` với
`--log_metrics --log_state_size`. Đây là nguồn đo memory-bank RAM cho Stage 2;
`ram_mb` chỉ là process RSS và không được dùng cho `membank_ram_*`.
Các CSV Stage 2 cũ thiếu/empty/non-numeric `maskmem_bytes` được xem là legacy
incomplete và phải rerun trước khi kết luận về RAM.

Vì vậy có thể chạy nhiều lần theo từng category/window. Lần sau script tự skip
cặp đã hoàn thành và chỉ chạy phần còn thiếu.

---

## 3. Kiểm tra trước khi chạy

Tất cả lệnh chạy từ repo root:

```bash
cd /home/phuocbui/Khoa_luan_tot_nghiep_sam2/samurai_optimized
```

Kiểm tra split đã lock:

```bash
python tests/test_splits_disjoint.py
```

Dry-run để xem category nào đã có data trên disk và pending jobs:

```bash
python scripts/stage2_run_batch.py \
    --data_root data/LaSOT \
    --splits splits/splits_v1.json \
    --metrics_dir metrics/stage2_lasot \
    --window_sizes 6,7,8,75,150 \
    --dry_run
```

Nếu `On disk: 0`, cần tải LaSOT category vào đúng layout:

```text
data/LaSOT/{category}/{video}/
  groundtruth.txt
  full_occlusion.txt
  out_of_view.txt
  img/*.jpg
```

---

## 4. Category lifecycle mode

Đây là mode nên dùng khi chỉ giữ một category trên disk tại một thời điểm để
tiết kiệm dung lượng. Wrapper bên ngoài chịu trách nhiệm tải category, gọi
Stage 2, rồi xóa category sau khi lệnh Stage 2 kết thúc.

Mỗi lần gọi chỉ truyền một category và toàn bộ window sizes cần thu thập dữ
liệu:

```bash
python scripts/stage2_run_batch.py \
    --data_root data/LaSOT \
    --splits splits/splits_v1.json \
    --metrics_dir metrics/stage2_lasot \
    --window_sizes 6,7,8,75,150 \
    --categories airplane
```

`stage2_run_batch.py` tự lấy các video `train_val` của `airplane` từ
`splits/splits_v1.json`. Không truyền danh sách video cụ thể. Với split hiện
tại, một category có 2 video `train_val`, nên lệnh trên tạo tối đa 10 cặp chạy:
`2 videos × 5 window sizes`.

Output vẫn được ghi theo `(window_size, video)`:

```text
metrics/stage2_lasot/6/stage2/airplane-*.csv
metrics/stage2_lasot/7/stage2/airplane-*.csv
...
results/stage2/6/airplane-*.txt
results/stage2/7/airplane-*.txt
...
```

Nếu lệnh bị dừng giữa chừng, tải lại category đó rồi chạy lại cùng lệnh. Script
sẽ skip các cặp `(window_size, video)` đã hoàn thành và chỉ chạy phần còn thiếu.

Wrapper ngoài có thể có dạng:

```bash
for CAT in airplane bear bicycle; do
    python scripts/download_lasot_category.py "$CAT"

    python scripts/stage2_run_batch.py \
        --data_root data/LaSOT \
        --splits splits/splits_v1.json \
        --metrics_dir metrics/stage2_lasot \
        --window_sizes 6,7,8,75,150 \
        --categories "$CAT"

    rm -rf "data/LaSOT/$CAT"
done
```

Không đặt thao tác xóa vào `stage2_run_batch.py`; xóa dữ liệu là trách nhiệm
của wrapper vận hành.

---

## 5. Chạy category tiếp theo

Khi tải thêm category mới, chạy tiếp bằng cùng `metrics_dir` và `pred_root`
mặc định:

```bash
python scripts/stage2_run_batch.py \
    --data_root data/LaSOT \
    --splits splits/splits_v1.json \
    --metrics_dir metrics/stage2_lasot \
    --window_sizes 6,7,8,75,150 \
    --categories bear
```

Nếu một category chưa có đủ frame/GT trên disk, script sẽ báo video đó trong
nhóm `missing` và không chạy video đó.

Trước khi chạy aggregate cuối cùng, các category đã xóa cần có lại trên disk vì
`stage2_aggregate.py` cần đọc `groundtruth.txt`, `full_occlusion.txt`, và
`out_of_view.txt`. Các CSV và prediction đã sinh trước đó vẫn được giữ trong
`metrics/` và `results/`.

---

## 6. Chạy theo từng window size cho nhiều GPU

Chỉ dùng mode này khi category data được giữ trên disk cho tới khi mọi window
size chạy xong, hoặc khi bạn cố ý chia việc theo GPU. Ví dụ chạy cùng category
nhưng mỗi GPU một window:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/stage2_run_batch.py \
    --data_root data/LaSOT \
    --splits splits/splits_v1.json \
    --metrics_dir metrics/stage2_lasot \
    --window_sizes 6 \
    --categories airplane

CUDA_VISIBLE_DEVICES=1 python scripts/stage2_run_batch.py \
    --data_root data/LaSOT \
    --splits splits/splits_v1.json \
    --metrics_dir metrics/stage2_lasot \
    --window_sizes 7 \
    --categories airplane
```

Không chạy song song 2 process cùng `--window_sizes 6 --categories airplane`
vì cả hai sẽ ghi vào cùng `results/stage2/6/{video}.txt`.

---

## 7. Aggregate cộng dồn sau nhiều lần chạy

Sau bất kỳ số lần batch nào, aggregate lại toàn bộ kết quả đã có:

```bash
python scripts/stage2_aggregate.py \
    --metrics_dir metrics/stage2_lasot \
    --data_root data/LaSOT \
    --pred_root results/stage2 \
    --splits splits/splits_v1.json \
    --out_dir analysis/stage2
```

Aggregator scan tất cả:

```text
metrics/stage2_lasot/*/stage2/*.csv
results/stage2/{window_size}/{video}.txt
```

Sau đó map `video_id -> category, split` từ `splits/splits_v1.json`. Vì vậy
không cần aggregate riêng từng category. Cứ chạy thêm category/window mới, rồi
aggregate lại cùng một lệnh.

Kiểm tra nhanh số dòng:

```bash
python - <<'PY'
import pandas as pd
df = pd.read_csv("analysis/stage2/stage2_results.csv")
print("rows:", len(df))
print(df.groupby("window_size")["video_id"].nunique())
print(df.groupby("category")["video_id"].nunique().sort_index())
PY
```

Khi chạy full 70 categories × 2 train_val videos × 5 window sizes, kỳ vọng:

```text
stage2_results.csv: 700 rows
stage2_attribute_results.csv: 1400 rows = 700 video/window rows × 2 attributes
per window_size: 140 videos
per category: 10 rows = 2 videos × 5 window sizes
```

---

## 8. Chọn N*

Sau khi aggregate đủ dữ liệu cần phân tích:

```bash
python scripts/stage2_select_n_star.py \
    --results_csv analysis/stage2/stage2_results.csv \
    --out_dir analysis/stage2 \
    --epsilon 0.005
```

Output:

```text
analysis/stage2/n_star_selection.json
```

Selector so sánh AUC theo từng video giữa candidate window và reference window
`150`, dùng Wilcoxon signed-rank test và ngưỡng mean AUC drop `epsilon`.

---

## 9. Smoke test với small_LaSOT nếu có data

Nếu có `data/small_LaSOT`, chạy 2 window trước:

```bash
python scripts/stage2_run_batch.py \
    --data_root data/small_LaSOT \
    --splits splits/splits_small_v1.json \
    --metrics_dir metrics/stage2_small \
    --window_sizes 6,75

python scripts/stage2_aggregate.py \
    --metrics_dir metrics/stage2_small \
    --data_root data/small_LaSOT \
    --pred_root results/stage2 \
    --splits splits/splits_small_v1.json \
    --out_dir analysis/stage2_small

python scripts/stage2_select_n_star.py \
    --results_csv analysis/stage2_small/stage2_results.csv \
    --out_dir analysis/stage2_small
```

Với `splits_small_v1.json`, kỳ vọng 12 train_val videos × 2 window sizes = 24
rows trong `analysis/stage2_small/stage2_results.csv`.

---

## 10. Verification commands

Tests nhanh cho Stage 2:

```bash
python3 tests/test_samurai_memory_selection_released_maskmem.py
python3 tests/test_release_old_frames.py
python3 tests/test_stage2_run_batch.py
python3 tests/test_stage2_aggregate.py
python3 tests/test_stage2_select_n_star.py
```

Toàn bộ smoke suite:

```bash
bash tests/run_all_tests.sh
```
