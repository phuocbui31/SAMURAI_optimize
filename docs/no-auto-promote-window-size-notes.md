# Ngữ cảnh: Cửa sổ bộ nhớ cố định khi không thêm conditioning frame mới

Tài liệu này dùng để cung cấp ngữ cảnh cho phần giải thích thí nghiệm
memory-window. Đây không phải hướng dẫn triển khai hay viết script.

Thiết lập cần mô tả là: dùng bản optimized với cơ chế dọn bộ nhớ được bật, nhưng
không thêm conditioning frame mới trong quá trình propagate. Ở chế độ này,
tracker giữ conditioning frame ban đầu và giới hạn các non-conditioning frame
gần đây còn tensor memory-bank hợp lệ.

## Ý tưởng chính

SAMURAI/SAM 2 lưu thông tin memory theo từng frame trong
`non_cond_frame_outputs`. Các trường tốn bộ nhớ nhất là:

- `maskmem_features`
- `maskmem_pos_enc`
- `pred_masks`

Hai trường đầu là tensor memory-bank quan trọng được dùng trong memory attention
của mô hình. Nếu giữ chúng cho mọi frame, bộ nhớ tăng theo độ dài video. Thiết
lập fixed-window sẽ giải phóng các tensor này ở những non-conditioning frame đã
quá cũ, nhờ đó mô hình chỉ có thể chọn memory từ một đoạn lịch sử gần và bị
chặn trên.

## `keep_window_maskmem`

`keep_window_maskmem` là tham số chính quyết định kích thước cửa sổ memory bank.

Nó điều khiển số non-conditioning frame gần nhất còn giữ:

- `maskmem_features`
- `maskmem_pos_enc`

Các non-conditioning frame nằm ngoài cửa sổ vẫn có thể giữ metadata nhẹ và một
số score, nhưng các tensor memory-bank nặng sẽ được đặt thành `None`.

Ví dụ về mặt khái niệm:

```text
current frame = 120
keep_window_maskmem = 75
oldest retained memory frame = 120 - 75 = 45
```

Tại thời điểm cleanup đó, các non-conditioning frame cũ hơn frame 45 không còn
đóng góp tensor memory-bank. Cơ chế chọn memory của mô hình vẫn có thể chọn từ
các frame nằm trong cửa sổ giữ lại, cộng với conditioning frame ban đầu.

Các giá trị candidate lấy từ notebook Stage 1
`analysis/stage1_thesis_analysis.ipynb`:

```text
6, 7, 8, 75, 150
```

Diễn giải:

- `6`: giá trị ablation tại median/P90 quan sát được.
- `7` hoặc `8`: cửa sổ rất nhỏ, dùng để kiểm tra vùng gần lower bound.
- `75`: điểm bắt đầu thực dụng, có frame coverage cao.
- `150`: biên an toàn lớn hơn cho các video khó.

Default cũ `1000` lớn hơn nhiều so với các candidate được gợi ý từ dữ liệu, nên
giữ nhiều memory hơn mức cần thiết cho phần lớn frame trong Stage 1.

## `keep_window_pred_masks`

`keep_window_pred_masks` điều khiển số non-conditioning frame gần nhất còn giữ
`pred_masks`.

Tham số này chủ yếu ảnh hưởng CPU RAM và sự tiện lợi khi visualize/evaluate. Nó
không phải tham số chính quyết định memory-bank selection của mô hình. Trong
nghiên cứu fixed-window, biến chính là `keep_window_maskmem`.

## `release_interval`

`release_interval` điều khiển tần suất cleanup.

Nó không định nghĩa kích thước cửa sổ bộ nhớ. Nó chỉ định nghĩa bao lâu thì áp
dụng luật cleanup một lần. Cửa sổ giữ lại vẫn được quyết định bởi
`keep_window_maskmem` và `keep_window_pred_masks`.

Ví dụ:

```text
release_interval = 60
cleanup frames = 60, 120, 180, ...
```

Tại mỗi cleanup frame, tracker tính frame cũ nhất được phép giữ dựa trên
current frame và window size, sau đó giải phóng tensor cũ nằm ngoài cửa sổ.

Trade-off:

- `release_interval` nhỏ: memory bám sát target window hơn, nhưng overhead
  cleanup cao hơn.
- `release_interval` lớn: overhead thấp hơn, nhưng memory có thể tạm thời vượt
  target window cho đến lần cleanup tiếp theo.
- `release_interval = 0`: tắt cleanup, memory sẽ tăng theo độ dài video.

Với thí nghiệm window-size nghiêm ngặt, interval nhỏ như `1` hoặc `10` cho kiểm
soát memory sạch hơn. Với inference thực dụng, interval lớn hơn như `60` giúp
giảm overhead cleanup.

## Quan hệ giữa window size và memory selection

Cơ chế fixed-window không trực tiếp viết lại thuật toán memory selection. Thay
vào đó, nó kiểm soát frame nào còn chứa tensor memory-bank hợp lệ.

Khi `maskmem_features` và `maskmem_pos_enc` của một frame cũ đã bị giải phóng,
frame đó về thực tế không còn dùng được như một selected memory frame. Nhờ vậy,
memory selection từ cơ chế full-history không bị chặn trở thành memory selection
trên một đoạn lịch sử gần có giới hạn.

Vì vậy, `keep_window_maskmem` là biến thí nghiệm chính để chọn kích thước
memory-bank window.

## Ngữ cảnh Stage 1

Stage 1 đo việc memory selection gốc của SAMURAI tự nhiên nhìn lùi bao xa khi
chưa áp đặt fixed window. Thống kê quan trọng là Distribution B: khoảng cách xa
nhất của selected memory trên mỗi frame.

Nguồn ưu tiên cho phần này là `analysis/stage1_thesis_analysis.ipynb`. Các file
trong `figures/stage1/`, `tables/stage1/`, `analysis/stage1_findings.md` và
`analysis/stage1/candidate_window_sizes.json` là output được notebook sinh ra để
phục vụ viết báo cáo.

Lưu ý: một vài bảng markdown/csv trong `tables/stage1/` có thể không đồng bộ nhẹ
với notebook sau cùng. Khi có khác biệt nhỏ, ưu tiên số liệu trong
`analysis/stage1_thesis_analysis.ipynb` và
`analysis/stage1/candidate_window_sizes.json`.

### Quy mô dữ liệu

```text
videos analyzed = 418 / 420
categories covered = 70 / 70
valid frames = 1,078,477
total selections = 6,462,389
```

Ý nghĩa:

- Độ phủ video gần đầy đủ: 418/420 video.
- Đủ 70/70 category, nên có thể dùng để phân tích theo category.
- Hơn 1 triệu valid frames và hơn 6.4 triệu memory selections, đủ lớn để chọn
  candidate window size theo percentile thay vì chọn thủ công.

### Distribution A và Distribution B

Notebook định nghĩa hai phân phối:

- Distribution A: khoảng cách của từng selected memory frame tới current frame
  trên toàn bộ selection.
- Distribution B: với mỗi current frame, lấy khoảng cách xa nhất trong các
  selected memory frames. Đây là metric chính để chọn window size vì một frame
  chỉ được cover nếu toàn bộ selected memory cần thiết của nó nằm trong cửa sổ.

Distribution A cho biết đa số selected memory rất gần current frame:

```text
P50 = 4 frames
P90 = 6 frames
P95 = 6 frames
P99 ≈ 54 frames
max = 913 frames
```

Distribution B trong notebook là cơ sở chính cho window-size selection:

```text
P50 = 6 frames
P90 = 6 frames
P95 = 8 frames
P99 = 74 frames
max = 913 frames
```

Phân tích chính:

- 90% frame chỉ cần nhìn lại tối đa khoảng 6 frame.
- 95% frame được cover với window khoảng 8 frame.
- 99% frame cần window khoảng 74 frame.
- `max = 913` cho thấy vẫn tồn tại các outlier dài, nhưng chúng hiếm.

### Candidate window sizes

Notebook xuất candidate vào `analysis/stage1/candidate_window_sizes.json`:

```text
6, 7, 8, 75, 150
```

Coverage tương ứng trong JSON:

```text
N = 6:   frame coverage = 94.01%, selection coverage = 96.39%
N = 7:   frame coverage = 94.72%, selection coverage = 96.72%
N = 8:   frame coverage = 95.23%, selection coverage = 96.98%
N = 75:  frame coverage = 99.01%, selection coverage = 99.22%
N = 150: frame coverage = 99.47%, selection coverage = 99.58%
```

Diễn giải:

- `6` là ablation sát median/P90.
- `7` là lower-bound candidate.
- `8` đại diện cho mức P95.
- `75` xấp xỉ mức P99 và là candidate thực dụng mạnh.
- `150` là safety margin lớn hơn cho các video/category khó.

Mục tiêu là tìm window nhỏ nhất vẫn giữ accuracy gần với hành vi full-history
gốc, đồng thời giảm đáng kể memory growth.

### Biểu đồ đã có

Notebook đã sinh 10 nhóm biểu đồ trong `figures/stage1/`, mỗi biểu đồ có cả
PNG và PDF:

```text
01_dist_A_histogram
02_dist_B_histogram
03_dist_B_cdf
04_coverage_curves
05_per_category_boxplot
06_outlier_categories
07_attribute_stratified
08_attribute_effect_size
09_membank_ram_growth
10_candidate_overlay_cdf
```

Vai trò từng nhóm:

- `01_dist_A_histogram`: cho thấy phân phối khoảng cách theo từng selected
  memory; phần lớn selection nằm rất gần current frame.
- `02_dist_B_histogram`: cho thấy phân phối max distance theo frame; đây là
  phân phối quyết định window size.
- `03_dist_B_cdf`: đọc trực tiếp coverage theo window size.
- `04_coverage_curves`: so sánh frame coverage và selection coverage theo các
  candidate window.
- `05_per_category_boxplot`: so sánh độ khó giữa category qua max distance.
- `06_outlier_categories`: chỉ ra category/video có khoảng cách nhìn lùi dài.
- `07_attribute_stratified`: tách phân phối theo attribute như occlusion hoặc
  out-of-view.
- `08_attribute_effect_size`: định lượng effect size của attribute lên max
  distance.
- `09_membank_ram_growth`: minh họa RAM memory bank tăng tuyến tính theo frame
  ở SAMURAI gốc.
- `10_candidate_overlay_cdf`: overlay các candidate `6, 7, 8, 75, 150` lên CDF
  của Distribution B để giải thích lựa chọn window.

### Bảng và file thống kê đã có

Các output bảng nằm trong `tables/stage1/` với định dạng `.csv`, `.md`, `.tex`.
Nhóm bảng chính:

```text
01_stage1_overview
02_distribution_A_stats
03_distribution_B_stats
04_per_category_summary
05_per_attribute_effect
06_ram_growth_rates
08_01_stage1_overview_final
08_02_distribution_B_key_stats
08_03_top5_hardest_categories
08_04_bottom5_easiest_categories
08_05_per_attribute_effect_final
08_06_candidate_window_sizes_final
```

Nội dung nên dùng:

- `01_stage1_overview` / `08_01_stage1_overview_final`: tổng quan số video,
  category, frame, selection.
- `02_distribution_A_stats`: thống kê Distribution A per-selection.
- `03_distribution_B_stats` / `08_02_distribution_B_key_stats`: thống kê
  Distribution B, là bảng quan trọng nhất cho window-size selection.
- `04_per_category_summary`: breakdown theo category.
- `05_per_attribute_effect` / `08_05_per_attribute_effect_final`: effect của
  attribute lên max distance.
- `06_ram_growth_rates`: tốc độ tăng RAM memory bank theo frame.
- `08_06_candidate_window_sizes_final`: candidate window size và coverage.

### Phân tích theo category

Notebook/findings nêu nhóm category khó nhất:

```text
yoyo, airplane, bear, bicycle, bird
```

Diễn giải: các category này thường có chuyển động mạnh, occlusion hoặc biến đổi
ngoại hình, nên có xu hướng cần memory nhìn xa hơn.

Nhóm category dễ hơn trong findings:

```text
train, turtle, umbrella, volleyball, zebra
```

Diễn giải: các category ổn định hơn thường saturate với window ngắn hơn. Khi
viết báo cáo, nên trình bày đây là phân tích định tính hỗ trợ, còn quyết định
window chính vẫn dựa vào Distribution B toàn tập.

### Phân tích theo attribute

Notebook/findings xác định hai attribute có ảnh hưởng lớn:

```text
full_occlusion: Cohen's d ≈ 0.98, p ≈ 0
out_of_view:    Cohen's d ≈ 1.39, p ≈ 0
```

Diễn giải:

- Khi có `full_occlusion`, frame có xu hướng cần nhìn xa hơn khoảng vài frame so
  với trạng thái không occlusion.
- Khi có `out_of_view`, max distance tăng mạnh hơn, vì tracker có thể cần ký ức
  xa hơn để khôi phục đối tượng sau khi xuất hiện lại.

Điểm này quan trọng cho báo cáo vì nó giải thích vì sao P99 và max distance lớn
hơn nhiều so với P50/P90.

### Phân tích RAM memory bank

Notebook sinh biểu đồ `09_membank_ram_growth` và bảng `06_ram_growth_rates`.
Findings ghi:

```text
linear growth confirmed: R² > 0.95 on sample videos
average slope ≈ 0.524 MB/frame
```

Ý nghĩa:

- SAMURAI gốc có xu hướng tích lũy memory bank tuyến tính theo độ dài video.
- Fixed-window kỳ vọng chặn RAM/VRAM theo `N * slope` thay vì tăng theo toàn bộ
  video length.
- Đây là luận cứ chính cho việc biến `keep_window_maskmem` thành tham số thực
  nghiệm.

### Kết luận Stage 1 cho Stage 2

Stage 1 chưa chọn window cuối cùng. Nó chỉ đề xuất candidate cho Stage 2.

Stage 2 cần sweep các candidate:

```text
6, 7, 8, 75, 150
```

Sau đó chọn `N*` theo trade-off giữa accuracy và memory. Bộ metric chính đã
được chuẩn hóa theo cách report của SAMURAI paper:

```text
AUC, P, Pnorm
```

Trong đó `P` là AUC của precision curve trên ngưỡng center error `[0, 50]`
pixel, còn `Pnorm` là AUC của normalized precision curve trên `[0, 0.5]`.
`OP50` và `OP75` vẫn có thể giữ làm metric phụ vì chúng lấy trực tiếp từ
success curve. `Prec@20`, `NormPrec@0.20` và `mIoU` không còn là output chính
của `--evaluate`.

Metric inline của `--evaluate` đã đổi trong `scripts/eval_utils.py`, nơi bảng
hiện in:

```text
AUC  OP50  OP75  P  Pnorm
```

Để aggregate Stage 2 theo nhiều window size mà không bị overwrite prediction,
spec/plan mới yêu cầu thêm `--pred_dir` cho `scripts/main_inference.py` và để
`scripts/stage2_run_batch.py` truyền `--pred_dir results/stage2/{window_size}`.
Phần này là bước triển khai tiếp theo; nếu chưa có `--pred_dir`, chỉ nên dùng
`--evaluate` để xem metric trên stdout, chưa đủ an toàn cho aggregator cộng dồn.

## Cách diễn đạt trong báo cáo

Có thể mô tả thiết lập này như sau:

> Một biến thể fixed-window memory bank, trong đó chỉ các non-conditioning frame
> gần đây còn giữ tensor memory-bank, trong khi conditioning frame ban đầu vẫn
> được bảo toàn. Kích thước cửa sổ được điều khiển bởi `keep_window_maskmem`,
> còn tần suất áp dụng cleanup được điều khiển riêng bởi `release_interval`.

Điểm cần phân biệt:

- `keep_window_maskmem` trả lời: giữ bao nhiêu lịch sử memory?
- `release_interval` trả lời: bao lâu thì áp dụng luật giữ/xóa đó một lần?
