# Chạy SAMURAI Gốc vs Optimized — Vấn đề & Cách khắc phục

**Ngày:** 2026-04-21
**Bối cảnh:** Khi so sánh metrics giữa hai cây code (`samurai/` — SAMURAI gốc, `sam2/` + `scripts/` — fork optimized), hai run cho ra kết quả **trùng khít từng con số thập phân** trên LaSOT-small (electricfan-1, gecko-1, mouse-1). Điều này khiến nghi ngờ "fork optimized không thực sự khác baseline" hoặc "scripts bị sai".

## TL;DR

Metrics trùng khít vì **cả hai lệnh đang load cùng một bản `sam2` Python package** (bản được `pip install -e` sau cùng), và **default code path tối ưu không được kích hoạt** khi không truyền `--optimized`. File scripts không sai; vấn đề là setup environment + default flags.

Để chạy đúng:

- **SAMURAI gốc bit-exact:** `pip install -e samurai/sam2/` → restart kernel → chạy `samurai/scripts/main_inference.py`.
- **Optimized thực sự:** `pip install -e sam2/` → restart kernel → chạy `scripts/main_inference.py` **kèm flag `--optimized`**.

---

## Vấn đề chi tiết

### 1. Hai cây `sam2` cùng tên distribution `SAM-2`

```
samurai_optimized/
├── sam2/              ← fork optimized
│   ├── setup.py       NAME = "SAM-2"
│   └── sam2/          ← module name = sam2
├── samurai/
│   └── sam2/          ← SAMURAI gốc
│       ├── setup.py   NAME = "SAM-2"
│       └── sam2/      ← module name = sam2
└── scripts/main_inference.py   ← fork
└── samurai/scripts/main_inference.py   ← gốc
```

Cả hai `setup.py` đều khai `NAME = "SAM-2"` và export top-level module `sam2`. Pip coi đây là **cùng một distribution** — chỉ giữ được **một** egg-link tại một thời điểm. Lệnh `pip install -e` sau ghi đè lệnh trước.

### 2. `python scripts/main_inference.py` không thêm CWD vào `sys.path`

Khi gọi script bằng đường dẫn, Python chỉ thêm **thư mục chứa script** (`scripts/`) vào `sys.path[0]`, **không phải CWD**. Cả hai repo đều có `sam2/` ở **cấp repo root** — không cùng cấp với `scripts/` — nên local `./sam2/` **không shadow** site-packages.

→ Mọi `import sam2` đều resolve qua egg-link duy nhất ở site-packages.

### 3. Code path tối ưu mặc định bị vô hiệu hoá

Trong `scripts/main_inference.py` (fork), block tối ưu chỉ chạy khi `args.optimized=True`:

```python
propagate_kwargs = {}
if args.optimized:
    propagate_kwargs["release_interval"] = args.release_interval
    propagate_kwargs["keep_window_maskmem"] = ...
    # ... các tối ưu khác

predictor.propagate_in_video(state, **propagate_kwargs)
```

Trong `sam2_video_predictor.py` (fork), `release_old_frames()` và `_maybe_promote_cond_frame()` chỉ chạy trong block `if release_interval > 0`. Default `release_interval=0` → **toàn bộ logic tối ưu bị bypass**.

→ Không truyền `--optimized` thì fork về numerical = baseline (chỉ khác async loader streaming, không đổi tensor values).

### 4. Hệ quả

User chạy hai lệnh "khác nhau" (script fork không-optimized vs script gốc), nhưng:

- Cùng `sam2` package (bản được install -e gần nhất).
- Cùng code path numerical (`release_interval=0`).
- ⇒ Metrics trùng khít từng số.

---

## Quy trình chuẩn

### Setup chung (1 lần/Kaggle session)

```bash
!pip install matplotlib==3.7 tikzplotlib jpeg4py opencv-python lmdb pandas scipy loguru psutil
```

Dependencies này độc lập với việc chọn bản `sam2`, không cần cài lại khi chuyển bản.

### A. Chạy SAMURAI GỐC

```bash
# Cell 1: cài bản gốc + restart kernel ngay trong cell
!pip uninstall SAM-2 -y
!pip install -e /kaggle/working/SAMURAI_optimize/samurai/sam2
!pip show SAM-2 | grep "Editable project location"
# Phải thấy: .../samurai/sam2

import os
os._exit(0)   # restart kernel (kernel sẽ tự reconnect)
```

```python
# Cell 2 (sau khi kernel restart): verify
import sam2
print(sam2.__file__)
print(sam2.__path__)
# Phải ra: .../samurai/sam2/sam2/__init__.py
# KHÔNG được là None hoặc path .../SAMURAI_optimize/sam2/...
```

```bash
# Cell 3: chạy script gốc
!python /kaggle/working/SAMURAI_optimize/samurai/scripts/main_inference.py \
    --model_name base_plus \
    --evaluate \
    --data_root /kaggle/input/datasets/sc0v1n0/lasot-small/small_LaSOT \
    --testing_set /kaggle/working/testing_set.txt
```

Lưu ý: nếu script gốc báo `unrecognized arguments: --log_metrics --run_tag` → bỏ 2 flag đó (script gốc chưa support).

### B. Chạy bản OPTIMIZED

```bash
# Cell 1: cài bản optimized + restart
!pip uninstall SAM-2 -y
!pip install -e /kaggle/working/SAMURAI_optimize/sam2
!pip show SAM-2 | grep "Editable project location"
# Phải thấy: .../SAMURAI_optimize/sam2

import os
os._exit(0)
```

```python
# Cell 2 (sau restart): verify
import sam2
print(sam2.__file__)
# Phải ra: .../SAMURAI_optimize/sam2/sam2/__init__.py
```

```bash
# Cell 3: chạy script optimized — BẮT BUỘC có --optimized
!python /kaggle/working/SAMURAI_optimize/scripts/main_inference.py \
    --optimized \
    --model_name base_plus \
    --log_metrics \
    --run_tag optimized \
    --evaluate \
    --data_root /kaggle/input/datasets/sc0v1n0/lasot-small/small_LaSOT \
    --testing_set /kaggle/working/testing_set.txt
```

Flag tối ưu có thể tinh chỉnh (xem `scripts/main_inference.py --help`):

| Flag | Default | Ý nghĩa |
|---|---:|---|
| `--release_interval` | 60 | Mỗi N frame giải phóng frame cũ |
| `--keep_window_maskmem` | 1000 | Số frame giữ maskmem_features |
| `--keep_window_pred_masks` | 60 | Số frame giữ pred_masks |
| `--max_cache_frames` | 60 | LRU cache images trong RAM |
| `--no_auto_promote` | off | Tắt auto-promote cond frames (reproduce SAMURAI 1 cond frame) |

---

## Checklist mỗi lần chuyển bản

1. `!pip uninstall SAM-2 -y`
2. `!pip install -e <path_đúng>`
3. `!pip show SAM-2 | grep Editable` — kiểm tra `Editable project location` đúng cây.
4. **Restart kernel** (`import os; os._exit(0)` hoặc menu Run → Restart).
5. `import sam2; print(sam2.__file__)` — verify path đúng và **không phải `None`**.
6. Chạy script tương ứng.

Bỏ qua bước 4-5 là nguyên nhân phổ biến nhất gây nhầm lẫn.

---

## Triệu chứng cần cảnh giác

| Triệu chứng | Nguyên nhân | Khắc phục |
|---|---|---|
| `sam2.__file__` = `None`, `__path__` = `_NamespacePath([...])` | Pip uninstall đã chạy, install lại fail hoặc chưa restart kernel; Python fallback sang namespace package từ thư mục `sam2/` cùng tên trên `sys.path` | `pip install -e <path>` lại + restart kernel |
| `TypeError: unexpected keyword argument 'max_cache_frames'` khi chạy script fork | Đang load `sam2` gốc (gốc không nhận kwarg này); script fork luôn truyền nó | `pip install -e samurai_optimized/sam2` (fork) + restart |
| Metrics 2 run trùng khít từng số thập phân | (a) Cùng `sam2` package, hoặc (b) `--optimized` không bật → `release_interval=0` → bypass tối ưu | Verify `sam2.__file__` + chắc chắn truyền `--optimized` |
| Kernel báo "died" sau `os._exit(0)` | Bình thường — Kaggle tự reconnect | Đợi vài giây hoặc click Reconnect |

---

## Tại sao không thể cài song song 2 bản

Pip dùng `NAME` trong `setup.py` làm khoá distribution. Cùng `NAME = "SAM-2"` → coi như cùng package → ghi đè. Cùng module name `sam2` → kể cả force install cũng conflict ở `site-packages/sam2/`.

**Cách work-around** (không khuyến nghị trong workflow này):

- Đổi `NAME = "SAM-2-baseline"` trong `samurai/sam2/setup.py` + đổi tên thư mục module `sam2` → `sam2_baseline` + sửa toàn bộ import → cài song song được. Tốn công, dễ vỡ upstream.
- Dùng 2 virtualenv riêng. Sạch hơn nhưng trên Kaggle bất tiện vì phải cài lại torch/CUDA.

→ Workflow `pip install -e + restart kernel` mỗi lần chuyển bản là **đơn giản nhất** cho mục đích so sánh baseline vs optimized trong luận văn.

---

## So sánh kết quả công bằng

Sau khi chạy cả hai bản với cùng `--testing_set`, dùng `scripts/plot_metrics.py`:

```bash
!python /kaggle/working/SAMURAI_optimize/scripts/plot_metrics.py \
    --run metrics/samurai_base_plus/baseline \
    --run metrics/samurai_base_plus/optimized \
    --label Baseline --label Optimized \
    --mode per_video
```

- **Accuracy (AUC/OP50/...)**: trùng → tối ưu numerically safe; khác → auto-promote thay đổi thuật toán, cần ghi chú.
- **Thời gian (`dt_ms`) + VRAM peak (`vram_peak_mb`)**: đọc từ CSV trong `metrics/<exp>/<run_tag>/<video>.csv` để chứng minh hiệu quả tối ưu.
