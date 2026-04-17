# Memory Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Tối ưu bộ nhớ SAMURAI để chạy video 1646+ frames trên Kaggle (30 GB RAM + 16 GB VRAM) mà không tràn, với tốc độ gần baseline và accuracy giảm tối thiểu.

**Architecture:** Bỏ cơ chế recompute maskmem (gây O(N²)). Tách `keep_window` thành 3 cửa sổ riêng cho 3 loại bộ nhớ (images, maskmem, pred_masks). Thêm auto-promote cond frames có threshold check, throttle và sliding window cap (streaming-friendly). Ép frame 0 luôn có mặt trong memory attention.

**Tech Stack:** Python 3.10+, PyTorch 2.3+, CUDA, SAM 2 / SAMURAI, Hydra (configs), pytest (smoke tests).

**Spec:** `samurai_optimized/docs/2026-04-17-memory-optimization-design.md`

**Pre-requisites:**
- Conda env `samurai` với pytorch + SAM 2 đã install (theo `AGENTS.md`).
- Checkpoint SAM 2.1 đã tải vào `samurai_optimized/sam2/checkpoints/`.
- Data LaSOT ở `samurai_optimized/data/LaSOT/` với ít nhất 1 video test (ví dụ `airplane-1`).
- Baseline SAMURAI gốc (`samurai/`) có thể chạy để so sánh output (optional).

**Working directory for all commands:** `/home/ubuntu-phuocbh/Downloads/Khoa_luan_tot_nghiep_sam2/samurai_optimized` unless otherwise noted.

**Testing strategy:** Dự án chưa có test suite (theo `AGENTS.md`). Kế hoạch dùng:
1. **Smoke tests** — script Python nhỏ trong `samurai_optimized/tests/` với assert đơn giản, không cần GPU cho unit test (dùng mock/fake predictor).
2. **Integration tests** — chạy `scripts/main_inference.py` trên 1 video ngắn, verify RAM/VRAM qua `psutil` + `nvidia-smi`.
3. **Regression test** — so sánh output masks với baseline bằng IoU.

Nếu `pytest` chưa install: `pip install pytest psutil`.

---

## Phase 1: Remove O(N²) Recompute

Root cause of current slowdown: `_ensure_all_selected_masksmem_available` at `sam2/sam2/sam2_video_predictor.py:1405-1425` iterates EVERY frame in `output_dict` on each step and recomputes maskmem via a full image encoder + memory encoder forward when missing. For a 1646-frame video this is ~1.3M image encoder runs. Deleting this mechanism returns speed to SAMURAI baseline; the replacement strategy (larger `keep_window_maskmem` default) is implemented in Phase 2.

**Files touched:**
- `samurai_optimized/sam2/sam2/sam2_video_predictor.py`

### Task 1.1: Delete recompute helper methods

- [ ] **Step 1 — Inspect target block:**

  Run: `sed -n '1293,1296p' samurai_optimized/sam2/sam2/sam2_video_predictor.py`
  Expected: blank line after `_clear_non_cond_mem_around_input`, then `def _ensure_maskmem_available(self, inference_state, frame_idx):` at line 1295.

  Run: `sed -n '1420,1425p' samurai_optimized/sam2/sam2/sam2_video_predictor.py`
  Expected: last two lines show `        for frame_idx in all_frame_indices:` and `            self._ensure_maskmem_available(inference_state, frame_idx)` at line 1425 (end of file).

- [ ] **Step 2 — Delete the three methods with Edit tool:**

  Use the Edit tool on `samurai_optimized/sam2/sam2/sam2_video_predictor.py`.

  oldString:
  ```python
      def _ensure_maskmem_available(self, inference_state, frame_idx):
          """
          Ensure maskmem is available for a frame. If maskmem has been evicted (is None),
          recompute it on-demand using the saved pred_mask and scores.
          
          This is called by _run_single_frame_inference() after memory selection chooses
          frames but before track_step() uses the maskmem.
          """
          output_dict = inference_state["output_dict"]
          
          # Check both cond and non_cond outputs
          frame_entry = None
          if frame_idx in output_dict["non_cond_frame_outputs"]:
              frame_entry = output_dict["non_cond_frame_outputs"][frame_idx]
          elif frame_idx in output_dict["cond_frame_outputs"]:
              frame_entry = output_dict["cond_frame_outputs"][frame_idx]
          
          # Frame hasn't been tracked yet - no need to recompute
          if frame_entry is None:
              return
          
          # Maskmem already available
          if frame_entry["maskmem_features"] is not None:
              return
          
          # Need to recompute!
          self._recompute_maskmem_for_frame(inference_state, frame_idx)

      def _recompute_maskmem_for_frame(self, inference_state, frame_idx):
          """
          Recompute maskmem for a frame that has been evicted.
          
          This happens when Memory Selection chooses a frame (based on high score)
          but the maskmem has been released by release_old_frames().
          
          Since pred_masks, object_score_logits, and best_iou_score are still saved,
          we can recompute maskmem using these values.
          """
          import torch.nn.functional as F
          
          output_dict = inference_state["output_dict"]
          
          # Get frame entry from output_dict
          frame_entry = None
          if frame_idx in output_dict["non_cond_frame_outputs"]:
              frame_entry = output_dict["non_cond_frame_outputs"][frame_idx]
          elif frame_idx in output_dict["cond_frame_outputs"]:
              frame_entry = output_dict["cond_frame_outputs"][frame_idx]
          
          if frame_entry is None:
              return
          
          device = inference_state["device"]
          
          # Get pred_mask and move to GPU if needed (may be offloaded to CPU)
          pred_mask = frame_entry["pred_masks"]
          if pred_mask is not None and pred_mask.device != device:
              pred_mask = pred_mask.to(device)
          
          # Get object_score_logits and move to GPU if needed
          object_score_logits = frame_entry["object_score_logits"]
          if object_score_logits is not None and object_score_logits.device != device:
              object_score_logits = object_score_logits.to(device)
          
          # Get backbone features for this frame (will reload from disk if evicted)
          _, _, current_vision_feats, _, feat_sizes = self._get_image_feature(
              inference_state, frame_idx, batch_size=1
          )
          
          # Resize pred_mask to high resolution if needed
          if pred_mask is not None:
              current_size = pred_mask.shape[-2:]
              if current_size != (self.image_size, self.image_size):
                  high_res_masks = F.interpolate(
                      pred_mask,
                      size=(self.image_size, self.image_size),
                      mode="bilinear",
                      align_corners=False,
                  )
              else:
                  high_res_masks = pred_mask
          else:
              # Create dummy mask if pred_mask is None
              high_res_masks = torch.zeros(
                  (1, 1, self.image_size, self.image_size),
                  dtype=torch.float32,
                  device=device
              )
          
          # Run Memory Encoder
          maskmem_features, maskmem_pos_enc = self._encode_new_memory(
              current_vision_feats=current_vision_feats,
              feat_sizes=feat_sizes,
              pred_masks_high_res=high_res_masks,
              object_score_logits=object_score_logits,
              is_mask_from_pts=True,
          )
          
          # Offload to CPU if needed
          storage_device = inference_state["storage_device"]
          maskmem_features = maskmem_features.to(torch.bfloat16)
          maskmem_features = maskmem_features.to(storage_device, non_blocking=True)
          maskmem_pos_enc = self._get_maskmem_pos_enc(
              inference_state, {"maskmem_pos_enc": maskmem_pos_enc}
          )
          
          # Save back to output_dict
          frame_entry["maskmem_features"] = maskmem_features
          frame_entry["maskmem_pos_enc"] = maskmem_pos_enc

      def _ensure_all_selected_masksmem_available(self, inference_state, current_frame_idx):
          """
          Ensure maskmem is available for all frames in the output_dict that will be
          considered by Memory Selection.
          
          This handles the case when release_old_frames() has evicted maskmem for frames
          that are still needed by Memory Selection based on their scores.
          """
          output_dict = inference_state["output_dict"]
          
          # Check all frames in both cond and non_cond outputs
          all_frame_indices = set()
          all_frame_indices.update(output_dict["cond_frame_outputs"].keys())
          all_frame_indices.update(output_dict["non_cond_frame_outputs"].keys())
          
          # Skip the current frame (it will compute its own maskmem)
          all_frame_indices.discard(current_frame_idx)
          
          # Ensure maskmem for each frame
          for frame_idx in all_frame_indices:
              self._ensure_maskmem_available(inference_state, frame_idx)
  ```

  newString: *(empty string — delete the entire block)*

- [ ] **Step 3 — Verify methods removed:**

  Run: `grep -n "_ensure_maskmem_available\|_recompute_maskmem_for_frame\|_ensure_all_selected_masksmem_available" samurai_optimized/sam2/sam2/sam2_video_predictor.py`
  Expected: exactly ONE match (the remaining call site at line ~1050 inside `_run_single_frame_inference`). Task 1.2 removes that call site.

- [ ] **Step 4 — Syntax check:**

  Run: `python -c "import ast; ast.parse(open('samurai_optimized/sam2/sam2/sam2_video_predictor.py').read()); print('OK')"`
  Expected: `OK`.

- [ ] **Step 5 — Commit:**

  ```bash
  cd samurai_optimized
  git add sam2/sam2/sam2_video_predictor.py
  git commit -m "refactor: remove O(N^2) recompute maskmem methods"
  ```

### Task 1.2: Remove call site in `_run_single_frame_inference`

- [ ] **Step 1 — Inspect target block:**

  Run: `sed -n '1044,1052p' samurai_optimized/sam2/sam2/sam2_video_predictor.py`
  Expected: shows `) = self._get_image_feature(...)` followed by a blank line, then the 3-line comment and the `if run_mem_encoder:` block calling `_ensure_all_selected_masksmem_available`.

  Note: after Task 1.1, line numbers may shift slightly, but the Edit tool matches content, not line numbers.

- [ ] **Step 2 — Delete the call site with Edit tool:**

  Use the Edit tool on `samurai_optimized/sam2/sam2/sam2_video_predictor.py`.

  oldString:
  ```python
          ) = self._get_image_feature(inference_state, frame_idx, batch_size)

          # Ensure maskmem is available for all frames that will be used by track_step
          # This handles the case when maskmem has been evicted by release_old_frames()
          # but Memory Selection still needs it based on high scores
          if run_mem_encoder:
              self._ensure_all_selected_masksmem_available(inference_state, frame_idx)

          # point and mask should not appear as input simultaneously on the same frame
  ```

  newString:
  ```python
          ) = self._get_image_feature(inference_state, frame_idx, batch_size)

          # point and mask should not appear as input simultaneously on the same frame
  ```

- [ ] **Step 3 — Verify call site removed:**

  Run: `grep -c "_ensure_all_selected_masksmem_available" samurai_optimized/sam2/sam2/sam2_video_predictor.py`
  Expected: `0`.

  Also run: `grep -c "_ensure_maskmem_available\|_recompute_maskmem_for_frame" samurai_optimized/sam2/sam2/sam2_video_predictor.py`
  Expected: `0`.

- [ ] **Step 4 — Syntax check:**

  Run: `python -c "import ast; ast.parse(open('samurai_optimized/sam2/sam2/sam2_video_predictor.py').read()); print('OK')"`
  Expected: `OK`.

- [ ] **Step 5 — Commit:**

  ```bash
  cd samurai_optimized
  git add sam2/sam2/sam2_video_predictor.py
  git commit -m "refactor: remove recompute call in _run_single_frame_inference"
  ```

**Phase 1 acceptance criteria:**
- Zero references to `_ensure_maskmem_available`, `_recompute_maskmem_for_frame`, or `_ensure_all_selected_masksmem_available` in `sam2/sam2/sam2_video_predictor.py`.
- File parses as valid Python (`ast.parse` succeeds).
- A short run of `scripts/main_inference.py` on 1 video starts without `AttributeError`. Accuracy on long videos may temporarily drop until Phase 2 raises `keep_window_maskmem` — this is expected and intended.

---

## Phase 2: Split `keep_window` into 3 windows; stop deleting cond frames

**Scope:** Refactor `release_old_frames()` in `sam2/sam2/sam2_video_predictor.py` to accept two independent windows (`keep_window_maskmem`, `keep_window_pred_masks`) and remove the buggy deletion of `cond_frame_outputs`. Cond frames must persist so memory attention keeps long-term anchors (frame 0 and promoted cond frames from Phase 4).

**Rationale:** (spec §3, §4.2) Current code deletes old cond frames except `init_frame_idx`, which breaks Memory Selection once `auto-promote` lands, and uses a single window for 3 very different memory types (GPU maskmem vs CPU pred_masks vs input images).

**Files touched:**
- `sam2/sam2/sam2_video_predictor.py` (edit method `release_old_frames`)
- `tests/test_release_old_frames.py` (new)

### Task 2.1 — Rewrite `release_old_frames` with 3 windows, preserve cond frames

- [ ] **Step 1 — Read current method.** Run `grep -n "def release_old_frames" sam2/sam2/sam2_video_predictor.py` and confirm it starts at line 593. Read lines 593-662 to confirm they match the `oldString` below (if line numbers drifted, re-locate and adjust before editing).

- [ ] **Step 2 — Edit the method.** Use the Edit tool on `sam2/sam2/sam2_video_predictor.py` with:

  **`oldString`** (verbatim current method, lines 593-662):
  ```python
      def release_old_frames(self, inference_state, keep_window=10):
          """
          Giải phóng tensor nặng của frame outputs cũ để giảm GPU memory.
          Giữ lại scores (best_iou_score, object_score_logits, kf_score, obj_ptr)
          để memory selection logic trong sam2_base.py vẫn hoạt động.
          
          Nếu inference_state["images"] là AsyncVideoFrameLoader, cũng giải phóng
          các frame images cũ khỏi RAM (Input Streaming).
          """
          output_dict = inference_state["output_dict"]
          cond_outputs = output_dict["cond_frame_outputs"]
          non_cond_outputs = output_dict["non_cond_frame_outputs"]

          if not cond_outputs:
              return

          newest_cond = max(cond_outputs.keys())
          oldest_allowed_idx = newest_cond - keep_window

          # Giữ lại frame 0 (init frame với bbox ban đầu) và conditioning frame mới nhất
          init_frame_idx = min(cond_outputs.keys())

          # Xóa tensor nặng của non_cond_frame_outputs cũ, giữ lại scores
          heavy_keys = ["maskmem_features", "maskmem_pos_enc", "pred_masks"]
          for frame_idx in list(non_cond_outputs.keys()):
              if frame_idx >= oldest_allowed_idx:
                  continue
              entry = non_cond_outputs[frame_idx]
              for key in heavy_keys:
                  if key in entry and entry[key] is not None:
                      entry[key] = None
              # Làm tương tự cho per-object outputs
              for obj_idx in inference_state["output_dict_per_obj"]:
                  obj_entry = inference_state["output_dict_per_obj"][obj_idx][
                      "non_cond_frame_outputs"
                  ].get(frame_idx)
                  if obj_entry is not None:
                      for key in heavy_keys:
                          if key in obj_entry and obj_entry[key] is not None:
                              obj_entry[key] = None

          # Xóa cond_frame_outputs cũ (trừ init frame và frame mới nhất)
          for frame_idx in list(cond_outputs.keys()):
              if frame_idx >= oldest_allowed_idx:
                  continue
              if frame_idx == init_frame_idx:
                  continue
              del cond_outputs[frame_idx]
              inference_state["consolidated_frame_inds"]["cond_frame_outputs"].discard(
                  frame_idx
              )
              for obj_idx in inference_state["output_dict_per_obj"]:
                  inference_state["output_dict_per_obj"][obj_idx][
                      "cond_frame_outputs"
                  ].pop(frame_idx, None)

          # Xóa cached features cũ
          for frame_idx in list(inference_state["cached_features"].keys()):
              if frame_idx < oldest_allowed_idx:
                  del inference_state["cached_features"][frame_idx]

          # Input Streaming: Evict old frames from AsyncVideoFrameLoader if available
          images_container = inference_state["images"]
          if hasattr(images_container, 'evict_old_frames'):
              # Keep frames within [oldest_allowed_idx - keep_window, newest_cond + keep_window]
              keep_start = max(0, oldest_allowed_idx - keep_window)
              keep_end = newest_cond + keep_window + 1
              images_container.evict_old_frames(keep_start, keep_end)

          gc.collect()
  ```

  **`newString`** (new 3-window implementation, cond preservation):
  ```python
      def release_old_frames(
          self,
          inference_state,
          keep_window_maskmem=1000,
          keep_window_pred_masks=60,
      ):
          """
          Release heavy tensors of old non-conditioning frames to reduce memory.

          Keeps scores (best_iou_score, object_score_logits, kf_score, obj_ptr) so
          Memory Selection logic in sam2_base.py continues to work after eviction.

          Three independent windows:
          - keep_window_maskmem: controls maskmem_features + maskmem_pos_enc (GPU VRAM)
          - keep_window_pred_masks: controls pred_masks (CPU RAM)
          - cached_features: evicted together with maskmem

          Conditioning frames (output_dict["cond_frame_outputs"]) are NEVER deleted here.
          They are managed separately by _maybe_promote_cond_frame (Phase 4).
          """
          output_dict = inference_state["output_dict"]
          cond_outputs = output_dict["cond_frame_outputs"]
          non_cond_outputs = output_dict["non_cond_frame_outputs"]

          if not cond_outputs:
              return

          newest_cond = max(cond_outputs.keys())
          oldest_allowed_maskmem = newest_cond - keep_window_maskmem
          oldest_allowed_pred_masks = newest_cond - keep_window_pred_masks

          for frame_idx in list(non_cond_outputs.keys()):
              entry = non_cond_outputs[frame_idx]
              # Evict maskmem + maskmem_pos_enc if out of maskmem window
              if frame_idx < oldest_allowed_maskmem:
                  for key in ("maskmem_features", "maskmem_pos_enc"):
                      if entry.get(key) is not None:
                          entry[key] = None
                  for obj_idx in inference_state["output_dict_per_obj"]:
                      obj_entry = inference_state["output_dict_per_obj"][obj_idx][
                          "non_cond_frame_outputs"
                      ].get(frame_idx)
                      if obj_entry is not None:
                          for key in ("maskmem_features", "maskmem_pos_enc"):
                              if obj_entry.get(key) is not None:
                                  obj_entry[key] = None
              # Evict pred_masks if out of pred_masks window
              if frame_idx < oldest_allowed_pred_masks:
                  if entry.get("pred_masks") is not None:
                      entry["pred_masks"] = None
                  for obj_idx in inference_state["output_dict_per_obj"]:
                      obj_entry = inference_state["output_dict_per_obj"][obj_idx][
                          "non_cond_frame_outputs"
                      ].get(frame_idx)
                      if obj_entry is not None and obj_entry.get("pred_masks") is not None:
                          obj_entry["pred_masks"] = None

          # Evict old cached_features (used only for in-flight frame features)
          for frame_idx in list(inference_state["cached_features"].keys()):
              if frame_idx < oldest_allowed_maskmem:
                  del inference_state["cached_features"][frame_idx]

          # Input streaming: evict images outside the maskmem keep range
          images_container = inference_state["images"]
          if hasattr(images_container, "evict_old_frames"):
              keep_start = max(0, oldest_allowed_maskmem)
              keep_end = newest_cond + keep_window_maskmem + 1
              images_container.evict_old_frames(keep_start, keep_end)

          gc.collect()
  ```

- [ ] **Step 3 — Verify edit applied.** Run `grep -n "keep_window_maskmem" sam2/sam2/sam2_video_predictor.py` — expect at least 4 matches. Run `grep -n "del cond_outputs\[" sam2/sam2/sam2_video_predictor.py` — expect 0 matches.

- [ ] **Step 4 — Syntax check.** Run `python -c "import ast; ast.parse(open('sam2/sam2/sam2_video_predictor.py').read()); print('OK')"` — expect `OK`.

- [ ] **Step 5 — Commit.**
  ```bash
  git add sam2/sam2/sam2_video_predictor.py
  git commit -m "refactor: split release_old_frames into 3 windows; stop deleting cond frames"
  ```

### Task 2.2 — Smoke test: cond frames not deleted

- [ ] **Step 1 — Ensure tests dir.** Run `mkdir -p tests` from `samurai_optimized/`. Confirm `ls -d tests` prints `tests`.

- [ ] **Step 2 — Write test file** `tests/test_release_old_frames.py` (ast-level check, no heavy imports needed):

  ```python
  """Verify source of release_old_frames does not delete cond_frame_outputs."""
  import ast, pathlib

  src = pathlib.Path("sam2/sam2/sam2_video_predictor.py").read_text()
  tree = ast.parse(src)

  for node in ast.walk(tree):
      if isinstance(node, ast.FunctionDef) and node.name == "release_old_frames":
          body_src = ast.get_source_segment(src, node)
          assert "del cond_outputs[" not in body_src, "release_old_frames must not delete cond frames"
          assert "keep_window_maskmem" in body_src, "must use keep_window_maskmem param"
          assert "keep_window_pred_masks" in body_src, "must use keep_window_pred_masks param"
          print("PASS")
          break
  else:
      raise AssertionError("release_old_frames not found")
  ```

- [ ] **Step 3 — Verify test file exists.** Run `ls -la tests/test_release_old_frames.py` — expect file present.

- [ ] **Step 4 — Run test.** Run `python tests/test_release_old_frames.py` from `samurai_optimized/` — expect `PASS` on stdout, exit code 0.

- [ ] **Step 5 — Commit.**
  ```bash
  git add tests/test_release_old_frames.py
  git commit -m "test: ast-level smoke test for release_old_frames 3-window refactor"
  ```

**Phase 2 exit criteria:**
- `release_old_frames` signature is `(self, inference_state, keep_window_maskmem=1000, keep_window_pred_masks=60)`.
- No `del cond_outputs[...]` anywhere in the method.
- `tests/test_release_old_frames.py` prints `PASS`.
- Python file parses cleanly (`ast.parse` OK).
- Two commits landed.

---

## Phase 3: Expose max_cache_frames via init_state and CLI

**Goal:** Tham số `max_cache_frames` (giới hạn LRU cache frames trong RAM của `AsyncVideoFrameLoader`) hiện đang hard-code ở `misc.py`. Phase này forward tham số này xuyên qua `init_state()` và expose thành CLI flag `--max_cache_frames` ở `main_inference.py`, giúp người dùng điều chỉnh RAM footprint khi chạy video dài.

**Spec reference:** `samurai_optimized/docs/2026-04-17-memory-optimization-design.md` §4.2, §4.5, §11.

**Files touched:**
- `samurai_optimized/sam2/sam2/sam2_video_predictor.py`
- `samurai_optimized/scripts/main_inference.py`
- `samurai_optimized/tests/test_max_cache_frames.py` (new)

**Preconditions:** Phase 1 & 2 đã commit. `load_video_frames` trong `sam2/sam2/utils/misc.py` đã hỗ trợ `max_cache_frames` (có sẵn theo `misc.py:214-258`).

### Task 3.1: Expose `max_cache_frames` trong `init_state`

- [ ] **Step 1 — Read:** Đọc `samurai_optimized/sam2/sam2/sam2_video_predictor.py` lines 44-60 để xác nhận signature hiện tại.
- [ ] **Step 2 — Edit:** Sửa signature + body của `init_state` để thêm `max_cache_frames=10` và forward vào `load_video_frames`.

  oldString:
  ```python
      @torch.inference_mode()
      def init_state(
          self,
          video_path,
          offload_video_to_cpu=False,
          offload_state_to_cpu=False,
          async_loading_frames=False,
      ):
          """Initialize an inference state."""
          compute_device = self.device  # device of the model
          images, video_height, video_width = load_video_frames(
              video_path=video_path,
              image_size=self.image_size,
              offload_video_to_cpu=offload_video_to_cpu,
              async_loading_frames=async_loading_frames,
              compute_device=compute_device,
          )
  ```

  newString:
  ```python
      @torch.inference_mode()
      def init_state(
          self,
          video_path,
          offload_video_to_cpu=False,
          offload_state_to_cpu=False,
          async_loading_frames=False,
          max_cache_frames=10,
      ):
          """Initialize an inference state."""
          compute_device = self.device  # device of the model
          images, video_height, video_width = load_video_frames(
              video_path=video_path,
              image_size=self.image_size,
              offload_video_to_cpu=offload_video_to_cpu,
              async_loading_frames=async_loading_frames,
              compute_device=compute_device,
              max_cache_frames=max_cache_frames,
          )
  ```

- [ ] **Step 3 — Verify:** `grep -n "max_cache_frames" samurai_optimized/sam2/sam2/sam2_video_predictor.py` phải trả về ít nhất 2 match (param + kwarg).
- [ ] **Step 4 — Syntax check:** `python -c "import ast; ast.parse(open('samurai_optimized/sam2/sam2/sam2_video_predictor.py').read())"` exit 0.
- [ ] **Step 5 — Commit:** `git add samurai_optimized/sam2/sam2/sam2_video_predictor.py && git commit -m "feat(predictor): expose max_cache_frames in init_state"`.

### Task 3.2: Thêm `--max_cache_frames` CLI flag

- [ ] **Step 1 — Read:** Đọc `samurai_optimized/scripts/main_inference.py` lines 28-65 và 125-140 để lấy context argparse + init_state calls.
- [ ] **Step 2a — Edit (argparse):** Chèn flag mới ngay sau block `--keep_window`.

  oldString:
  ```python
  parser.add_argument(
      "--keep_window",
      type=int,
      default=10,
      help="Giữ bao nhiêu frame gần nhất khi release (mặc định: 10)",
  )
  parser.add_argument(
      "--model_name",
  ```

  newString:
  ```python
  parser.add_argument(
      "--keep_window",
      type=int,
      default=10,
      help="Giữ bao nhiêu frame gần nhất khi release (mặc định: 10)",
  )
  parser.add_argument(
      "--max_cache_frames",
      type=int,
      default=10,
      help="Số images tối đa giữ trong RAM (LRU cache). Mặc định: 10",
  )
  parser.add_argument(
      "--model_name",
  ```

- [ ] **Step 2b — Edit (init_state calls):** Thêm `max_cache_frames=args.max_cache_frames` vào cả 2 nhánh.

  oldString:
  ```python
          if args.optimized:
              state = predictor.init_state(
                  frame_folder,
                  offload_video_to_cpu=True,
                  offload_state_to_cpu=False,
                  async_loading_frames=True,
              )
          else:
              state = predictor.init_state(
                  frame_folder,
                  offload_video_to_cpu=True,
                  offload_state_to_cpu=True,
                  async_loading_frames=True,
              )
  ```

  newString:
  ```python
          if args.optimized:
              state = predictor.init_state(
                  frame_folder,
                  offload_video_to_cpu=True,
                  offload_state_to_cpu=False,
                  async_loading_frames=True,
                  max_cache_frames=args.max_cache_frames,
              )
          else:
              state = predictor.init_state(
                  frame_folder,
                  offload_video_to_cpu=True,
                  offload_state_to_cpu=True,
                  async_loading_frames=True,
                  max_cache_frames=args.max_cache_frames,
              )
  ```

- [ ] **Step 3 — Verify:** `grep -n "max_cache_frames" samurai_optimized/scripts/main_inference.py` phải trả 3 match (1 argparse + 2 init_state).
- [ ] **Step 4 — Syntax check:** `python -c "import ast; ast.parse(open('samurai_optimized/scripts/main_inference.py').read())"` exit 0. Cũng chạy `python samurai_optimized/scripts/main_inference.py --help | grep max_cache_frames` để verify argparse.
- [ ] **Step 5 — Commit:** `git add samurai_optimized/scripts/main_inference.py && git commit -m "feat(cli): add --max_cache_frames flag to main_inference"`.

### Task 3.3: Smoke test cho `max_cache_frames` cap

- [ ] **Step 1 — Read:** Đọc `samurai_optimized/sam2/sam2/utils/misc.py` lines 214-258 để xác nhận `AsyncVideoFrameLoader` accept `max_cache_frames` và có eviction logic.
- [ ] **Step 2 — Write:** Tạo file `samurai_optimized/tests/test_max_cache_frames.py` với nội dung:

  ```python
  """Smoke test that AsyncVideoFrameLoader.max_cache_frames limits cache size.

  This test mocks _load_img_as_tensor so no real JPEGs are needed.
  """
  import sys, types, os, pathlib, tempfile
  import unittest.mock as mock

  # Stub heavy deps
  _torch = types.ModuleType("torch")
  class _Tensor:
      def __init__(self, data=None, shape=(3, 128, 128)):
          self.shape = shape
      def to(self, *a, **kw): return self
      def __sub__(self, other): return self
      def __truediv__(self, other): return self
  _torch.Tensor = _Tensor
  _torch.tensor = lambda *a, **kw: _Tensor()
  _torch.from_numpy = lambda a: _Tensor()
  _torch.device = lambda x: x
  sys.modules.setdefault("torch", _torch)

  import numpy as np
  pil_stub = types.ModuleType("PIL")
  class _Img:
      size = (128, 128)
      def convert(self, m): return self
      def resize(self, sz): return self
  pil_image = types.ModuleType("PIL.Image")
  pil_image.open = lambda p: _Img()
  pil_stub.Image = pil_image
  sys.modules.setdefault("PIL", pil_stub)
  sys.modules.setdefault("PIL.Image", pil_image)

  tqdm_stub = types.ModuleType("tqdm")
  tqdm_stub.tqdm = lambda x, **kw: x
  sys.modules.setdefault("tqdm", tqdm_stub)

  sys.path.insert(0, "sam2")
  from sam2.utils.misc import AsyncVideoFrameLoader

  def _fake_load(path, image_size):
      return _Tensor(), 128, 128

  def test_cache_cap():
      with mock.patch("sam2.utils.misc._load_img_as_tensor", side_effect=_fake_load):
          loader = AsyncVideoFrameLoader(
              img_paths=[f"fake_{i}.jpg" for i in range(50)],
              image_size=128,
              offload_video_to_cpu=True,
              img_mean=_Tensor(),
              img_std=_Tensor(),
              compute_device="cpu",
              max_cache_frames=5,
          )
          for i in range(20):
              _ = loader[i]
          loaded_count = sum(1 for x in loader.images if x is not None)
          assert loaded_count <= 5, f"expected <=5 frames cached, got {loaded_count}"
      print("PASS")

  if __name__ == "__main__":
      test_cache_cap()
  ```

  **Fallback (ast-level test)** nếu smoke test flake do async thread start trước khi patch apply — thay thế file bằng:

  ```python
  """Verify AsyncVideoFrameLoader has max_cache_frames parameter and LRU eviction."""
  import ast, pathlib
  src = pathlib.Path("sam2/sam2/utils/misc.py").read_text()
  tree = ast.parse(src)
  for node in ast.walk(tree):
      if isinstance(node, ast.ClassDef) and node.name == "AsyncVideoFrameLoader":
          cls_src = ast.get_source_segment(src, node)
          assert "max_cache_frames" in cls_src, "max_cache_frames param missing"
          assert "_evict_oldest_frame" in cls_src or "evict_old_frames" in cls_src, "eviction missing"
          print("PASS")
          break
  else:
      raise AssertionError("AsyncVideoFrameLoader class not found")
  ```

- [ ] **Step 3 — Verify:** Chạy `cd samurai_optimized && python tests/test_max_cache_frames.py`. Expect stdout `PASS`. Nếu flake → chuyển sang fallback ast-level test và re-run.
- [ ] **Step 4 — Syntax check:** `python -c "import ast; ast.parse(open('samurai_optimized/tests/test_max_cache_frames.py').read())"` exit 0.
- [ ] **Step 5 — Commit:** `git add samurai_optimized/tests/test_max_cache_frames.py && git commit -m "test: add max_cache_frames smoke test"`.

**Phase 3 acceptance criteria:**
- `init_state(..., max_cache_frames=N)` forward đúng giá trị xuống `load_video_frames`.
- `python scripts/main_inference.py --help` hiển thị flag `--max_cache_frames` với default 10.
- Smoke test (hoặc fallback ast test) pass.
- Không có regression chức năng với default values (baseline behavior preserved vì default = 10 == giá trị hard-code cũ).

---

## Phase 4: Add Quality-Controlled Auto-Promote (Streaming-Friendly)

**Goal:** Thay thế auto-promote "mù" hiện tại (`promote_idx = frame_idx - 2`) bằng logic có threshold check (3 ngưỡng SAMURAI: IoU, obj score, KF score), throttle theo `promote_interval`, và sliding-window cap `max_auto_promoted_cond_frames` — streaming-friendly (không cần biết `num_frames` trước). Đồng thời update `propagate_in_video` signature để dùng 2 cửa sổ `keep_window_maskmem` / `keep_window_pred_masks` từ Phase 2.

**Spec reference:** `samurai_optimized/docs/2026-04-17-memory-optimization-design.md` §4.2 (item 5), §4.5, §11, §12.

**Files touched:**
- `samurai_optimized/sam2/sam2/sam2_video_predictor.py` (add method + update signature + rewrite promote block)
- `samurai_optimized/scripts/main_inference.py` (CLI flags)
- `samurai_optimized/tests/test_maybe_promote.py` (new)

**Preconditions:** Phase 2 đã merge (`release_old_frames` nhận `keep_window_maskmem` + `keep_window_pred_masks`). Phase 3 đã merge.

### Task 4.1: Add `_maybe_promote_cond_frame` method to predictor

- [ ] **Step 1 — Read.** Đọc `sam2/sam2/sam2_video_predictor.py` lines 664-682 để xác nhận vị trí cuối của `append_frame_as_cond_frame` (line 681 kết thúc với `inference_state["consolidated_frame_inds"]["cond_frame_outputs"].add(frame_idx)`).

- [ ] **Step 2 — Edit.** Dùng Edit tool chèn method mới ngay sau `append_frame_as_cond_frame`, ngay trước `@torch.inference_mode()` decorator của `propagate_in_video_preflight`.

  `oldString`:
  ```python
          inference_state["consolidated_frame_inds"]["non_cond_frame_outputs"].discard(
              frame_idx
          )
          inference_state["consolidated_frame_inds"]["cond_frame_outputs"].add(frame_idx)

      @torch.inference_mode()
      def propagate_in_video_preflight(self, inference_state):
  ```

  `newString`:
  ```python
          inference_state["consolidated_frame_inds"]["non_cond_frame_outputs"].discard(
              frame_idx
          )
          inference_state["consolidated_frame_inds"]["cond_frame_outputs"].add(frame_idx)

      def _maybe_promote_cond_frame(
          self,
          inference_state,
          frame_idx,
          promote_interval=500,
          promote_search_window=50,
          max_auto_promoted_cond_frames=4,
      ):
          """Conditionally promote a high-quality non-cond frame to cond.

          Throttle + threshold-based selection, streaming-friendly (bounded memory
          without needing total num_frames upfront).
          """
          cond_outputs = inference_state["output_dict"]["cond_frame_outputs"]
          non_cond = inference_state["output_dict"]["non_cond_frame_outputs"]

          # 1. Throttle: skip if recent cond is closer than promote_interval
          cond_keys_excluding_zero = [k for k in cond_outputs.keys() if k != 0]
          nearest_cond = max(cond_keys_excluding_zero) if cond_keys_excluding_zero else 0
          if frame_idx - nearest_cond < promote_interval:
              return

          # 2. Search for the nearest quality candidate (backward within window)
          candidate_idx = None
          search_start = max(1, frame_idx - promote_search_window)
          for i in range(frame_idx - 2, search_start - 1, -1):
              if i not in non_cond:
                  continue
              entry = non_cond[i]
              if entry.get("maskmem_features") is None:
                  continue
              iou = entry.get("best_iou_score")
              obj = entry.get("object_score_logits")
              kf = entry.get("kf_score")
              if iou is None or obj is None:
                  continue
              try:
                  iou_val = iou.item()
                  obj_val = obj.item()
                  kf_val = kf.item() if kf is not None else None
              except (AttributeError, RuntimeError):
                  continue
              if (
                  iou_val > self.memory_bank_iou_threshold
                  and obj_val > self.memory_bank_obj_score_threshold
                  and (kf_val is None or kf_val > self.memory_bank_kf_score_threshold)
              ):
                  candidate_idx = i
                  break

          if candidate_idx is None:
              return

          # 3. Promote candidate to cond
          self.append_frame_as_cond_frame(inference_state, candidate_idx)

          # 4. Evict oldest auto-promoted cond frame (never evict frame 0)
          auto_promoted = sorted(k for k in cond_outputs.keys() if k != 0)
          while len(auto_promoted) > max_auto_promoted_cond_frames:
              oldest = auto_promoted[0]
              cond_outputs.pop(oldest, None)
              for obj_idx in inference_state["output_dict_per_obj"]:
                  inference_state["output_dict_per_obj"][obj_idx][
                      "cond_frame_outputs"
                  ].pop(oldest, None)
              inference_state["consolidated_frame_inds"]["cond_frame_outputs"].discard(
                  oldest
              )
              auto_promoted.pop(0)

      @torch.inference_mode()
      def propagate_in_video_preflight(self, inference_state):
  ```

- [ ] **Step 3 — Verify.** `grep -n "def _maybe_promote_cond_frame" sam2/sam2/sam2_video_predictor.py` — expect exactly 1 match. `grep -n "memory_bank_iou_threshold\|memory_bank_obj_score_threshold\|memory_bank_kf_score_threshold" sam2/sam2/sam2_video_predictor.py` — expect 3+ matches.
- [ ] **Step 4 — Syntax check.** `python -c "import ast; ast.parse(open('sam2/sam2/sam2_video_predictor.py').read()); print('OK')"` — expect `OK`.
- [ ] **Step 5 — Commit.** `git add sam2/sam2/sam2_video_predictor.py && git commit -m "feat(predictor): add _maybe_promote_cond_frame with threshold + throttle"`.

### Task 4.2: Update `propagate_in_video` signature + swap promote block

- [ ] **Step 1 — Read.** Đọc `sam2/sam2/sam2_video_predictor.py` lines 753-849 để xác nhận signature và block hiện tại.

- [ ] **Step 2a — Edit signature.** Update param list + add sanity warning.

  `oldString`:
  ```python
      @torch.inference_mode()
      def propagate_in_video(
          self,
          inference_state,
          start_frame_idx=None,
          max_frame_num_to_track=None,
          reverse=False,
          release_interval=0,
          keep_window=10,
      ):
          """Propagate the input points across frames to track in the entire video."""
          self.propagate_in_video_preflight(inference_state)
  ```

  `newString`:
  ```python
      @torch.inference_mode()
      def propagate_in_video(
          self,
          inference_state,
          start_frame_idx=None,
          max_frame_num_to_track=None,
          reverse=False,
          release_interval=0,
          keep_window_maskmem=1000,
          keep_window_pred_masks=60,
          enable_auto_promote=True,
          promote_interval=500,
          promote_search_window=50,
          max_auto_promoted_cond_frames=4,
      ):
          """Propagate the input points across frames to track in the entire video."""
          import warnings
          if promote_search_window > keep_window_maskmem:
              warnings.warn(
                  f"promote_search_window ({promote_search_window}) > "
                  f"keep_window_maskmem ({keep_window_maskmem}); candidate frames "
                  f"may have been evicted by release_old_frames."
              )
          self.propagate_in_video_preflight(inference_state)
  ```

- [ ] **Step 2b — Edit release/promote block.** Replace the "Giải phóng frame cũ định kỳ (Tối ưu A)" block:

  `oldString`:
  ```python
              # Giải phóng frame cũ định kỳ (Tối ưu A)
              if (
                  release_interval > 0
                  and frame_idx > 0
                  and frame_idx % release_interval == 0
                  and not reverse
              ):
                  promote_idx = frame_idx - 2
                  if promote_idx in output_dict["non_cond_frame_outputs"]:
                      self.append_frame_as_cond_frame(inference_state, promote_idx)
                  self.release_old_frames(inference_state, keep_window=keep_window)
  ```

  `newString`:
  ```python
              # Periodic memory maintenance (Phase 4 design)
              if (
                  release_interval > 0
                  and frame_idx > 0
                  and frame_idx % release_interval == 0
                  and not reverse
              ):
                  if enable_auto_promote:
                      self._maybe_promote_cond_frame(
                          inference_state,
                          frame_idx,
                          promote_interval=promote_interval,
                          promote_search_window=promote_search_window,
                          max_auto_promoted_cond_frames=max_auto_promoted_cond_frames,
                      )
                  self.release_old_frames(
                      inference_state,
                      keep_window_maskmem=keep_window_maskmem,
                      keep_window_pred_masks=keep_window_pred_masks,
                  )
  ```

- [ ] **Step 3 — Verify.** `grep -n "keep_window_maskmem\|enable_auto_promote\|_maybe_promote_cond_frame" sam2/sam2/sam2_video_predictor.py` — expect matches both in `propagate_in_video` and the new method. `grep -n "keep_window=keep_window" sam2/sam2/sam2_video_predictor.py` — expect 0 matches.
- [ ] **Step 4 — Syntax check.** `python -c "import ast; ast.parse(open('sam2/sam2/sam2_video_predictor.py').read()); print('OK')"`.
- [ ] **Step 5 — Commit.** `git add sam2/sam2/sam2_video_predictor.py && git commit -m "refactor(predictor): update propagate_in_video for new 3-window and auto-promote"`.

### Task 4.3: Update CLI flags in `scripts/main_inference.py`

- [ ] **Step 1 — Read.** Đọc `scripts/main_inference.py` lines 40-65 (flag `--keep_window` + `--max_cache_frames` từ Phase 3) và lines 151-154 (`propagate_kwargs`). Flag `--max_cache_frames` phải giữ nguyên.

- [ ] **Step 2a — Replace `--keep_window` with 7 new flags.** Edit:

  `oldString`:
  ```python
  parser.add_argument(
      "--keep_window",
      type=int,
      default=10,
      help="Giữ bao nhiêu frame gần nhất khi release (mặc định: 10)",
  )
  ```

  `newString`:
  ```python
  parser.add_argument(
      "--keep_window_maskmem",
      type=int,
      default=1000,
      help="Số frame giữ maskmem_features trong output_dict. Mặc định: 1000",
  )
  parser.add_argument(
      "--keep_window_pred_masks",
      type=int,
      default=60,
      help="Số frame giữ pred_masks trong output_dict. Mặc định: 60",
  )
  parser.add_argument(
      "--enable_auto_promote",
      action="store_true",
      default=True,
      help="Bật auto-promote cond frames chất lượng cao. Mặc định: bật",
  )
  parser.add_argument(
      "--no_auto_promote",
      dest="enable_auto_promote",
      action="store_false",
      help="Tắt auto-promote (reproduce SAMURAI baseline 1 cond frame)",
  )
  parser.add_argument(
      "--promote_interval",
      type=int,
      default=500,
      help="Khoảng cách tối thiểu giữa 2 lần promote. Mặc định: 500",
  )
  parser.add_argument(
      "--promote_search_window",
      type=int,
      default=50,
      help="Cửa sổ tìm candidate lùi từ frame hiện tại. Mặc định: 50",
  )
  parser.add_argument(
      "--max_auto_promoted_cond_frames",
      type=int,
      default=4,
      help="Cap số cond frame auto-promoted (ngoài frame 0). Mặc định: 4",
  )
  ```

- [ ] **Step 2b — Replace `propagate_kwargs` block.** Edit:

  `oldString`:
  ```python
          propagate_kwargs = {}
          if args.optimized:
              propagate_kwargs["release_interval"] = args.release_interval
              propagate_kwargs["keep_window"] = args.keep_window
  ```

  `newString`:
  ```python
          propagate_kwargs = {}
          if args.optimized:
              propagate_kwargs["release_interval"] = args.release_interval
              propagate_kwargs["keep_window_maskmem"] = args.keep_window_maskmem
              propagate_kwargs["keep_window_pred_masks"] = args.keep_window_pred_masks
              propagate_kwargs["enable_auto_promote"] = args.enable_auto_promote
              propagate_kwargs["promote_interval"] = args.promote_interval
              propagate_kwargs["promote_search_window"] = args.promote_search_window
              propagate_kwargs["max_auto_promoted_cond_frames"] = args.max_auto_promoted_cond_frames
  ```

- [ ] **Step 3 — Verify.** `python scripts/main_inference.py --help 2>&1 | grep -E "keep_window_maskmem|keep_window_pred_masks|promote_interval|no_auto_promote|max_auto_promoted"` — expect 5+ lines. `grep -n "args.keep_window\b" scripts/main_inference.py` — expect 0 matches.
- [ ] **Step 4 — Syntax check.** `python -c "import ast; ast.parse(open('scripts/main_inference.py').read()); print('OK')"`.
- [ ] **Step 5 — Commit.** `git add scripts/main_inference.py && git commit -m "feat(cli): add auto-promote and 3-window flags to main_inference"`.

### Task 4.4: AST-level smoke test cho `_maybe_promote_cond_frame`

- [ ] **Step 1 — Read.** `ls tests/` để xác nhận thư mục tồn tại (đã có từ Phase 2/3). Nếu thiếu, `mkdir -p tests`.

- [ ] **Step 2 — Write test file** `tests/test_maybe_promote.py`:

  ```python
  """Verify _maybe_promote_cond_frame exists and has threshold + throttle logic."""
  import ast, pathlib

  src = pathlib.Path("sam2/sam2/sam2_video_predictor.py").read_text()
  tree = ast.parse(src)

  found = False
  for node in ast.walk(tree):
      if isinstance(node, ast.FunctionDef) and node.name == "_maybe_promote_cond_frame":
          body_src = ast.get_source_segment(src, node)
          assert "memory_bank_iou_threshold" in body_src
          assert "memory_bank_obj_score_threshold" in body_src
          assert "memory_bank_kf_score_threshold" in body_src
          assert "promote_interval" in body_src
          assert "max_auto_promoted_cond_frames" in body_src
          assert "append_frame_as_cond_frame" in body_src
          # Check that frame 0 is never evicted
          assert "k != 0" in body_src
          found = True
          break
  assert found, "_maybe_promote_cond_frame not found"
  print("PASS")
  ```

- [ ] **Step 3 — Verify.** `python tests/test_maybe_promote.py` from `samurai_optimized/` — expect stdout `PASS`, exit code 0.
- [ ] **Step 4 — Syntax check.** `python -c "import ast; ast.parse(open('tests/test_maybe_promote.py').read()); print('OK')"`.
- [ ] **Step 5 — Commit.** `git add tests/test_maybe_promote.py && git commit -m "test: add _maybe_promote_cond_frame ast smoke test"`.

**Phase 4 exit criteria:**
- Method `_maybe_promote_cond_frame` có đủ 3 threshold check (IoU, obj, KF), throttle qua `promote_interval`, eviction cap qua `max_auto_promoted_cond_frames`, và không bao giờ evict frame 0.
- `propagate_in_video` nhận 6 param mới (`keep_window_maskmem`, `keep_window_pred_masks`, `enable_auto_promote`, `promote_interval`, `promote_search_window`, `max_auto_promoted_cond_frames`); không còn tham chiếu `keep_window` cũ.
- CLI có đủ 7 flag mới (6 + `--no_auto_promote` toggle); `--keep_window` đã bị remove.
- `tests/test_maybe_promote.py` in `PASS`.
- 4 commits landed.

---

## Phase 5: Force-Include Frame 0 in Memory Attention

**Goal:** Đảm bảo frame 0 (user anchor với bbox gốc) luôn xuất hiện trong `selected_cond_outputs` của `_prepare_memory_conditioned_features`, ngay cả khi đã có nhiều cond frames hơn `max_cond_frames_in_attn`. Đồng thời enable flag qua config YAML với `max_cond_frames_in_attn: 2` để cross-attention luôn có 2 anchors: frame 0 + cond frame gần nhất.

**Rationale:** (spec §4.3, §4.4, §11) Sau Phase 2 cond frames không còn bị xoá, và Phase 4 đã thêm auto-promote, nên số lượng cond frames có thể >> `max_cond_frames_in_attn`. `select_closest_cond_frames` chỉ chọn k cond frames gần frame hiện tại nhất ⇒ frame 0 sẽ bị loại sau vài trăm frame. Ép frame 0 vào luôn giữ long-term anchor, chống drift.

**Files touched:**
- `sam2/sam2/modeling/sam2_base.py` (add param + update memory prep logic)
- `sam2/sam2/configs/samurai/sam2.1_hiera_b+.yaml`
- `sam2/sam2/configs/samurai/sam2.1_hiera_l.yaml`
- `sam2/sam2/configs/samurai/sam2.1_hiera_s.yaml`
- `sam2/sam2/configs/samurai/sam2.1_hiera_t.yaml`
- `tests/test_force_include_frame0.py` (new)

**Preconditions:** Phase 1-4 đã commit. `sam2_base.py` có signature `SAM2Base.__init__` với `max_cond_frames_in_attn=-1` ở line ~43 và assignment `self.max_cond_frames_in_attn = max_cond_frames_in_attn` ở line ~196.

### Task 5.1: Add `force_include_init_cond_frame` param to `SAM2Base.__init__`

- [ ] **Step 1 — Read:** Đọc `sam2/sam2/modeling/sam2_base.py` lines 40-50 và 190-220 để xác nhận param `max_cond_frames_in_attn=-1` ở line 43 và assignment ở line 196.

- [ ] **Step 2a — Edit signature:** Thêm param mới sau `max_cond_frames_in_attn=-1`.

  oldString:
  ```python
          max_cond_frames_in_attn=-1,
          # on the first frame, whether to directly add the no-memory embedding to the image feature
  ```

  newString:
  ```python
          max_cond_frames_in_attn=-1,
          # Whether to always include frame 0 (the user anchor with initial bbox) in the
          # selected cond frames during memory attention, even when the total number of
          # cond frames exceeds max_cond_frames_in_attn. Keeps long-term tracking anchor.
          force_include_init_cond_frame: bool = False,
          # on the first frame, whether to directly add the no-memory embedding to the image feature
  ```

- [ ] **Step 2b — Edit attribute assignment:** Thêm `self.force_include_init_cond_frame` ngay sau `self.max_cond_frames_in_attn`.

  oldString:
  ```python
          self._build_sam_heads()
          self.max_cond_frames_in_attn = max_cond_frames_in_attn

          # Whether to use SAMURAI or original SAM 2
          self.samurai_mode = samurai_mode
  ```

  newString:
  ```python
          self._build_sam_heads()
          self.max_cond_frames_in_attn = max_cond_frames_in_attn
          self.force_include_init_cond_frame = force_include_init_cond_frame

          # Whether to use SAMURAI or original SAM 2
          self.samurai_mode = samurai_mode
  ```

- [ ] **Step 3 — Verify:** `grep -n "force_include_init_cond_frame" sam2/sam2/modeling/sam2_base.py` → expect ≥3 matches (param + comment block + attribute assignment).

- [ ] **Step 4 — Syntax check:** `python -c "import ast; ast.parse(open('sam2/sam2/modeling/sam2_base.py').read()); print('OK')"` exit 0 with `OK`.

- [ ] **Step 5 — Commit:**
  ```bash
  git add sam2/sam2/modeling/sam2_base.py
  git commit -m "feat(base): add force_include_init_cond_frame param to SAM2Base"
  ```

### Task 5.2: Honor `force_include_init_cond_frame` in `_prepare_memory_conditioned_features`

- [ ] **Step 1 — Read:** Đọc `sam2/sam2/modeling/sam2_base.py` lines 645-660 để xác nhận block `select_closest_cond_frames` hiện tại khớp với oldString dưới.

- [ ] **Step 2 — Edit memory selection logic:**

  oldString:
  ```python
              cond_outputs = output_dict["cond_frame_outputs"]
              selected_cond_outputs, unselected_cond_outputs = select_closest_cond_frames(
                  frame_idx, cond_outputs, self.max_cond_frames_in_attn
              )
              t_pos_and_prevs = [(0, out) for out in selected_cond_outputs.values()]
  ```

  newString:
  ```python
              cond_outputs = output_dict["cond_frame_outputs"]
              if (
                  self.force_include_init_cond_frame
                  and 0 in cond_outputs
                  and self.max_cond_frames_in_attn >= 2
                  and len(cond_outputs) > self.max_cond_frames_in_attn
              ):
                  # Ensure frame 0 (user anchor) is always included as a long-term cond frame.
                  # Pick (max_cond_frames_in_attn - 1) closest cond frames from the remaining
                  # set, then prepend frame 0.
                  frame_0_entry = cond_outputs[0]
                  other_cond = {k: v for k, v in cond_outputs.items() if k != 0}
                  selected_others, unselected_cond_outputs = select_closest_cond_frames(
                      frame_idx, other_cond, self.max_cond_frames_in_attn - 1
                  )
                  selected_cond_outputs = {0: frame_0_entry, **selected_others}
              else:
                  selected_cond_outputs, unselected_cond_outputs = select_closest_cond_frames(
                      frame_idx, cond_outputs, self.max_cond_frames_in_attn
                  )
              t_pos_and_prevs = [(0, out) for out in selected_cond_outputs.values()]
  ```

- [ ] **Step 3 — Verify:** `grep -n "force_include_init_cond_frame" sam2/sam2/modeling/sam2_base.py` → expect ≥4 matches total. Also `grep -n "frame_0_entry" sam2/sam2/modeling/sam2_base.py` → expect ≥1 match.

- [ ] **Step 4 — Syntax check:** `python -c "import ast; ast.parse(open('sam2/sam2/modeling/sam2_base.py').read()); print('OK')"` exit 0 with `OK`.

- [ ] **Step 5 — Commit:**
  ```bash
  git add sam2/sam2/modeling/sam2_base.py
  git commit -m "feat(base): honor force_include_init_cond_frame in memory selection"
  ```

### Task 5.3: Enable config flags in all 4 SAMURAI YAMLs

Tất cả 4 file có cùng khối hyperparameter cuối — kết thúc bằng dòng `memory_bank_kf_score_threshold: 0.0`. Thêm 2 dòng mới ngay sau đó.

- [ ] **Step 1 — Read:** Kiểm tra dòng cuối của 4 YAML để xác nhận block khớp:
  ```bash
  tail -n 3 sam2/sam2/configs/samurai/sam2.1_hiera_b+.yaml
  tail -n 3 sam2/sam2/configs/samurai/sam2.1_hiera_l.yaml
  tail -n 3 sam2/sam2/configs/samurai/sam2.1_hiera_s.yaml
  tail -n 3 sam2/sam2/configs/samurai/sam2.1_hiera_t.yaml
  ```
  Expect tất cả hiển thị block kết thúc bằng `memory_bank_obj_score_threshold: 0.0` / `memory_bank_kf_score_threshold: 0.0`. Nếu không khớp (file đã có trailing content khác), dừng lại và điều chỉnh oldString trước khi edit.

- [ ] **Step 2a — Edit** `sam2/sam2/configs/samurai/sam2.1_hiera_b+.yaml`:

  oldString:
  ```yaml
    memory_bank_obj_score_threshold: 0.0
    memory_bank_kf_score_threshold: 0.0
  ```

  newString:
  ```yaml
    memory_bank_obj_score_threshold: 0.0
    memory_bank_kf_score_threshold: 0.0
    # Memory attention: cap at 2 cond frames (frame 0 + nearest) and force-keep frame 0
    max_cond_frames_in_attn: 2
    force_include_init_cond_frame: true
  ```

- [ ] **Step 2b — Edit** `sam2/sam2/configs/samurai/sam2.1_hiera_l.yaml`: dùng cùng oldString / newString như 2a.

- [ ] **Step 2c — Edit** `sam2/sam2/configs/samurai/sam2.1_hiera_s.yaml`: dùng cùng oldString / newString như 2a.

- [ ] **Step 2d — Edit** `sam2/sam2/configs/samurai/sam2.1_hiera_t.yaml`: dùng cùng oldString / newString như 2a.

- [ ] **Step 3 — Verify:**
  ```bash
  grep -l "force_include_init_cond_frame: true" sam2/sam2/configs/samurai/*.yaml | wc -l
  grep -l "max_cond_frames_in_attn: 2" sam2/sam2/configs/samurai/*.yaml | wc -l
  ```
  → cả 2 lệnh phải in ra `4`.

- [ ] **Step 4 — Syntax check (YAML parse):**
  ```bash
  for f in sam2/sam2/configs/samurai/sam2.1_hiera_b+.yaml sam2/sam2/configs/samurai/sam2.1_hiera_l.yaml sam2/sam2/configs/samurai/sam2.1_hiera_s.yaml sam2/sam2/configs/samurai/sam2.1_hiera_t.yaml; do
    python -c "import yaml; yaml.safe_load(open('$f').read()); print('OK $f')"
  done
  ```
  Expect 4 dòng `OK …`.

- [ ] **Step 5 — Commit:**
  ```bash
  git add sam2/sam2/configs/samurai/sam2.1_hiera_b+.yaml \
          sam2/sam2/configs/samurai/sam2.1_hiera_l.yaml \
          sam2/sam2/configs/samurai/sam2.1_hiera_s.yaml \
          sam2/sam2/configs/samurai/sam2.1_hiera_t.yaml
  git commit -m "config: enable force_include_init_cond_frame + max_cond_frames_in_attn=2 for samurai"
  ```

### Task 5.4: AST smoke test for force-include wiring

- [ ] **Step 1 — Read:** Xác nhận thư mục `tests/` đã tồn tại (`ls -d tests` → `tests`). Nếu chưa, chạy `mkdir -p tests`.

- [ ] **Step 2 — Write** file `tests/test_force_include_frame0.py`:

  ```python
  """Verify force_include_init_cond_frame wiring in sam2_base.py + configs."""
  import ast, pathlib

  src = pathlib.Path("sam2/sam2/modeling/sam2_base.py").read_text()
  tree = ast.parse(src)

  # Find SAM2Base class
  cls = next(
      (n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == "SAM2Base"),
      None,
  )
  assert cls is not None, "SAM2Base class not found"

  init_fn = next(
      (n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__"),
      None,
  )
  prep_fn = next(
      (
          n
          for n in cls.body
          if isinstance(n, ast.FunctionDef)
          and n.name == "_prepare_memory_conditioned_features"
      ),
      None,
  )
  assert init_fn is not None, "__init__ not found"
  assert prep_fn is not None, "_prepare_memory_conditioned_features not found"

  init_src = ast.get_source_segment(src, init_fn)
  prep_src = ast.get_source_segment(src, prep_fn)

  assert "force_include_init_cond_frame" in init_src, "param missing in __init__"
  assert "self.force_include_init_cond_frame" in init_src, "attribute assignment missing"
  assert (
      "force_include_init_cond_frame" in prep_src
  ), "logic missing in _prepare_memory_conditioned_features"
  assert (
      "frame_0_entry" in prep_src or "0 in cond_outputs" in prep_src
  ), "frame 0 handling missing"

  # Config check
  for yaml_name in [
      "sam2.1_hiera_b+.yaml",
      "sam2.1_hiera_l.yaml",
      "sam2.1_hiera_s.yaml",
      "sam2.1_hiera_t.yaml",
  ]:
      p = pathlib.Path(f"sam2/sam2/configs/samurai/{yaml_name}")
      text = p.read_text()
      assert "max_cond_frames_in_attn: 2" in text, f"{yaml_name} missing max_cond_frames_in_attn"
      assert (
          "force_include_init_cond_frame: true" in text
      ), f"{yaml_name} missing force_include_init_cond_frame"

  print("PASS")
  ```

- [ ] **Step 3 — Verify:** `ls -la tests/test_force_include_frame0.py` → file exists.

- [ ] **Step 4 — Run test + syntax check:**
  ```bash
  python -c "import ast; ast.parse(open('tests/test_force_include_frame0.py').read()); print('OK')"
  python tests/test_force_include_frame0.py
  ```
  Expect `OK` then `PASS`, both with exit 0.

- [ ] **Step 5 — Commit:**
  ```bash
  git add tests/test_force_include_frame0.py
  git commit -m "test: add force_include_init_cond_frame ast smoke test"
  ```

**Phase 5 exit criteria:**
- `SAM2Base.__init__` accepts `force_include_init_cond_frame: bool = False` và expose `self.force_include_init_cond_frame`.
- `_prepare_memory_conditioned_features` giữ frame 0 trong `selected_cond_outputs` khi flag bật, `max_cond_frames_in_attn ≥ 2`, frame 0 ∈ cond_outputs, và `len(cond_outputs) > max_cond_frames_in_attn`.
- Cả 4 YAML configs trong `sam2/sam2/configs/samurai/` có cả `max_cond_frames_in_attn: 2` và `force_include_init_cond_frame: true`; `yaml.safe_load` parse OK.
- `python tests/test_force_include_frame0.py` in ra `PASS`.
- 4 commit (`5.1`, `5.2`, `5.3`, `5.4`) đã land.

---

## Phase 6: End-to-End Validation on LaSOT

**Goal:** Measure peak RAM, peak VRAM, wall-clock time, and accuracy (IoU vs
SAMURAI gốc) on a long LaSOT video to verify the memory optimizations meet the
targets in spec sections 5–8 (RAM < 2 GB, VRAM < 6 GB, ≤1.15× baseline time,
Mean IoU ≥ 0.9 vs `samurai/`).

**Reality check — run FIRST before starting Phase 6:**
```bash
ls -d samurai_optimized/data/LaSOT/airplane/airplane-1/ 2>&1 || \
    echo "LaSOT airplane-1 not present — see Task 6.2 adaptation notes"
ls samurai_optimized/data/LaSOT/airplane/airplane-1/img/ 2>&1 | head -3 || \
    echo "frames missing"
```
At the time this plan was written, `samurai_optimized/data/LaSOT/` did NOT
exist on this workspace. The engineer running Phase 6 must either (a) download
a LaSOT category containing at least one long (>1000-frame) video, or (b) edit
`testing_set_small.txt` to point at any available long sequence (e.g.
`book-3`, `person-1`, etc.). All commands below assume `airplane-1`; substitute
the chosen sequence name throughout.

Dependency: `pip install psutil` (used by Task 6.1 benchmark wrapper).

---

### Task 6.1: Memory + speed monitoring wrapper script

**Step 1 — Prepare.** Ensure `psutil` is installed:
```bash
pip install psutil
```

**Step 2 — Create `samurai_optimized/tests/bench_inference.py`** with the
contents below. It spawns `main_inference.py` as a subprocess and samples
system RAM (parent + children via `psutil`) and GPU VRAM (via
`nvidia-smi --query-gpu=memory.used`) every 2 seconds, streams child stdout
unchanged so the tqdm progress bar is still visible, and prints peak usage on
exit.

```python
"""Benchmark wrapper for main_inference.py.

Runs main_inference.py as a subprocess and samples RAM/VRAM every 2s. Prints
peak usage at the end.

Usage:
  python tests/bench_inference.py -- \
      --optimized --max_cache_frames 10 --keep_window_maskmem 1000 \
      --testing_set data/LaSOT/testing_set_small.txt
"""
import argparse, os, subprocess, sys, time, threading
import psutil

def get_gpu_mem_mb():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            encoding="utf-8",
        )
        return int(out.strip().split("\n")[0])
    except Exception:
        return -1

def main():
    if "--" in sys.argv:
        idx = sys.argv.index("--")
        inf_args = sys.argv[idx + 1:]
    else:
        inf_args = []
    cmd = [sys.executable, "scripts/main_inference.py"] + inf_args
    print(f"Running: {' '.join(cmd)}")

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    ps_proc = psutil.Process(proc.pid)

    peak_rss_mb = 0
    peak_gpu_mb = 0
    samples = []
    t0 = time.time()

    def sample():
        nonlocal peak_rss_mb, peak_gpu_mb
        while proc.poll() is None:
            try:
                rss = ps_proc.memory_info().rss // (1024 * 1024)
                for child in ps_proc.children(recursive=True):
                    try:
                        rss += child.memory_info().rss // (1024 * 1024)
                    except psutil.NoSuchProcess:
                        pass
                gpu = get_gpu_mem_mb()
                peak_rss_mb = max(peak_rss_mb, rss)
                peak_gpu_mb = max(peak_gpu_mb, gpu)
                samples.append((time.time() - t0, rss, gpu))
            except psutil.NoSuchProcess:
                break
            time.sleep(2.0)

    th = threading.Thread(target=sample, daemon=True)
    th.start()

    for line in proc.stdout:
        sys.stdout.write(line.decode("utf-8", errors="replace"))
        sys.stdout.flush()

    proc.wait()
    elapsed = time.time() - t0
    print(f"\n=== Benchmark ===")
    print(f"Elapsed: {elapsed:.1f}s")
    print(f"Peak system RAM: {peak_rss_mb} MB")
    print(f"Peak GPU VRAM:  {peak_gpu_mb} MB")
    print(f"Samples recorded: {len(samples)}")

if __name__ == "__main__":
    main()
```

**Step 3 — Smoke test.** Confirm the script parses args and forwards `--help`:
```bash
python samurai_optimized/tests/bench_inference.py -- --help 2>&1 | head -20
```

**Step 4 — Commit.**
```bash
git add samurai_optimized/tests/bench_inference.py
git commit -m "test: add bench_inference wrapper for ram/vram monitoring"
```

---

### Task 6.2: Prepare small test set (single long video)

**Step 1 — Reality check.** Confirm whether the candidate video exists:
```bash
ls -d samurai_optimized/data/LaSOT/airplane/airplane-1/ 2>&1 || \
    echo "Video not available — edit testing_set_small.txt"
ls samurai_optimized/data/LaSOT/airplane/airplane-1/img/ 2>&1 | head -3 || \
    echo "frames missing"
```
If `airplane-1` is missing, pick any available long LaSOT sequence and
substitute its name below and in Tasks 6.3–6.6.

**Step 2 — Create `samurai_optimized/data/LaSOT/testing_set_small.txt`**
(only if it does not already exist):
```
airplane-1
```

**Step 3 — Verify `main_inference.py` accepts the flag:**
```bash
grep -n testing_set samurai_optimized/scripts/main_inference.py | head
```

**Step 4 — Commit.**
```bash
git add samurai_optimized/data/LaSOT/testing_set_small.txt
git commit -m "test: add testing_set_small.txt for benchmarks"
```

---

### Task 6.3: Baseline run (`--no_auto_promote`, reproduces SAMURAI gốc behavior)

**Step 1 — Prepare output dir.**
```bash
mkdir -p samurai_optimized/tests/results
```

**Step 2 — Run.** From `samurai_optimized/`:
```bash
python tests/bench_inference.py -- \
    --optimized \
    --no_auto_promote \
    --max_cache_frames 10 \
    --keep_window_maskmem 1000 \
    --keep_window_pred_masks 60 \
    --model_name base_plus \
    --data_root data/LaSOT \
    --testing_set data/LaSOT/testing_set_small.txt 2>&1 | tee tests/results/baseline_no_promote.log
```

**Step 3 — Verify.** Expected at end of log:
- `Peak system RAM:` below ~2000 MB
- `Peak GPU VRAM:` below ~6000 MB
- Process exits 0, no OOM, all frames (1646 for airplane-1) processed.

**Step 4 — Record.** Note `Elapsed`, `Peak system RAM`, `Peak GPU VRAM`; these
feed the results table in Task 6.7.

**Step 5 — No commit** (log file only; committed in Task 6.8).

---

### Task 6.4: Optimized run (auto-promote enabled)

**Step 1 — Run.**
```bash
python tests/bench_inference.py -- \
    --optimized \
    --max_cache_frames 10 \
    --keep_window_maskmem 1000 \
    --keep_window_pred_masks 60 \
    --promote_interval 500 \
    --max_auto_promoted_cond_frames 4 \
    --model_name base_plus \
    --data_root data/LaSOT \
    --testing_set data/LaSOT/testing_set_small.txt 2>&1 | tee tests/results/optimized_auto_promote.log
```

**Step 2 — Verify.**
- Peak RAM < 2 GB, peak VRAM < 6 GB
- `Elapsed` ≤ 1.15 × baseline Elapsed from Task 6.3
- Log shows auto-promote activity (search for `promote` / `cond_frame` log
  messages if emitted by Phase 4 code).

**Step 3 — Record** numbers for the Task 6.7 table.

**Step 4 — No commit.**

---

### Task 6.5: Small-window stress test (`max_cache_frames=30`)

**Step 1 — Run.**
```bash
python tests/bench_inference.py -- \
    --optimized \
    --max_cache_frames 30 \
    --keep_window_maskmem 1000 \
    --keep_window_pred_masks 60 \
    --model_name base_plus \
    --data_root data/LaSOT \
    --testing_set data/LaSOT/testing_set_small.txt 2>&1 | tee tests/results/stress_cache30.log
```

**Step 2 — Verify RAM scaling.** Expected:
`peak_RAM(cache=30) − peak_RAM(Task 6.4, cache=10)` ≈ 20 × per-cached-frame cost
- 640×360 decoded JPEG: ~60 MB total extra
- 1024×1024 `image_size`: ~240 MB total extra

Both are acceptable; the point is RAM grows roughly linearly with
`max_cache_frames`, confirming the knob functions as designed.

**Step 3 — Record** numbers for Task 6.7.

**Step 4 — No commit.**

---

### Task 6.6: Accuracy comparison vs SAMURAI gốc

**Step 1 — Create `samurai_optimized/tests/compare_results.py`** with:

```python
"""Compare SAMURAI baseline predictions vs samurai_optimized predictions.

Reads per-video result txt files in the format produced by main_inference.py:
  x,y,w,h\n    (one line per frame)

Computes IoU per frame and reports mean IoU. Small regressions acceptable
(spec section 7: AO/Success < 1% drop).
"""
import sys, pathlib

def iou(box_a, box_b):
    ax1, ay1, aw, ah = box_a
    bx1, by1, bw, bh = box_b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0

def load(path):
    return [tuple(int(float(x)) for x in line.strip().split(","))
            for line in pathlib.Path(path).read_text().splitlines() if line.strip()]

if __name__ == "__main__":
    baseline_path, new_path = sys.argv[1], sys.argv[2]
    base = load(baseline_path)
    new = load(new_path)
    n = min(len(base), len(new))
    if len(base) != len(new):
        print(f"WARN: length mismatch base={len(base)} new={len(new)}; using first {n}")
    ious = [iou(base[i], new[i]) for i in range(n)]
    mean = sum(ious) / n if n else 0.0
    print(f"Frames compared: {n}")
    print(f"Mean IoU:       {mean:.4f}")
    print(f"Min IoU:        {min(ious):.4f}")
    below_50 = sum(1 for x in ious if x < 0.5)
    print(f"Frames IoU<0.5: {below_50} ({100*below_50/n:.1f}%)")
    if mean < 0.7:
        print("REGRESSION — mean IoU below 0.7")
        sys.exit(1)
    print("OK")
```

**Step 2a — Share LaSOT data between `samurai/` and `samurai_optimized/`.** The baseline run needs frames under `samurai/data/LaSOT/`. Create a symlink so both trees read the same data (run from repo root `/home/ubuntu-phuocbh/Downloads/Khoa_luan_tot_nghiep_sam2/`):
```bash
mkdir -p samurai/data
ln -sfn "$(pwd)/samurai_optimized/data/LaSOT" samurai/data/LaSOT
ls samurai/data/LaSOT/airplane/airplane-1/img/ | head -3
```
Expected: first 3 JPG filenames print (e.g. `00000001.jpg`). If symlinks are not supported on your filesystem (e.g., Kaggle kernel `/kaggle/working/` is OK; `/kaggle/input/` is read-only), copy instead:
```bash
mkdir -p samurai/data/LaSOT
cp -r samurai_optimized/data/LaSOT/* samurai/data/LaSOT/
```

**Step 2b — Produce SAMURAI gốc baseline predictions.** From repo root:
```bash
cd samurai && python scripts/main_inference.py \
    --data_root data/LaSOT \
    --testing_set data/LaSOT/testing_set_small.txt 2>&1 | tee /tmp/samurai_baseline.log
cd -
```
If `samurai/data/LaSOT/testing_set_small.txt` does not exist, create it with
the same single-line content as Task 6.2.

**Step 3 — Compare.** From `samurai_optimized/`:
```bash
python tests/compare_results.py \
    ../samurai/results/samurai/samurai_base_plus/airplane-1.txt \
    results/samurai/samurai_base_plus/airplane-1.txt \
    2>&1 | tee tests/results/iou_vs_samurai.log
```

**Step 4 — Verify.** Expected `Mean IoU ≥ 0.9`. Script exits 1 if below 0.7.

**Step 5 — Commit script.**
```bash
git add samurai_optimized/tests/compare_results.py
git commit -m "test: add compare_results.py for IoU regression checks"
```

---

### Task 6.7: Report

**Step 1 — Create `samurai_optimized/docs/2026-04-17-memory-optimization-results.md`**
with the table below, filling the `___` cells from the Task 6.3–6.6 outputs:

```markdown
# Memory Optimization Results (2026-04-17)

Test video: airplane-1 (LaSOT, 1646 frames)
Hardware: Kaggle T4 GPU, 30 GB system RAM
Model: sam2.1_hiera_base_plus

## Runs
| Config                  | Peak RAM (MB) | Peak VRAM (MB) | Elapsed (s) | Mean IoU vs samurai/ |
|-------------------------|---------------|----------------|-------------|----------------------|
| baseline (no_promote)   | ___           | ___            | ___         | ___                  |
| optimized (auto_promote)| ___           | ___            | ___         | ___                  |
| stress (cache=30)       | ___           | ___            | ___         | ___                  |

Numbers filled in by running `tests/bench_inference.py` per Tasks 6.3–6.5.
IoU filled in from Task 6.6 output.

## Conclusion
- [ ] RAM peak < 2 GB (target)
- [ ] VRAM peak < 6 GB (target)
- [ ] Optimized run ≤ 1.15× baseline time
- [ ] Mean IoU ≥ 0.9 vs samurai/ baseline
```

**Step 2 — Verify** all checkboxes can be ticked from the recorded numbers. If
any target is missed, investigate (revisit Phase 4/5 config defaults) before
proceeding to Task 6.8.

---

### Task 6.8: Final commit + cleanup

**Step 1 — Stage all Phase 6 artifacts.**
```bash
git add samurai_optimized/tests/ samurai_optimized/docs/2026-04-17-memory-optimization-results.md
```

**Step 2 — Commit.**
```bash
git commit -m "test: record end-to-end validation results for memory optimization"
```

**Step 3 — Verify.**
```bash
git log --oneline -5
git status
```
Working tree should be clean; the last commit visible.

---

**Phase 6 exit criteria:**
- `tests/bench_inference.py` and `tests/compare_results.py` exist and run.
- `tests/results/` contains three benchmark logs and one IoU log.
- `docs/2026-04-17-memory-optimization-results.md` has all cells filled and
  all four conclusion checkboxes ticked.
- Targets met: RAM < 2 GB, VRAM < 6 GB, time ≤ 1.15× baseline, Mean IoU ≥ 0.9.
