# processing.py

import torch
import cv2
import numpy as np
import json
from matplotlib.colors import to_rgb
import time
from ultralytics import YOLO
import sys
from pathlib import Path
import os
import torchvision.ops as ops
from PIL import Image
# <<< THÊM MỚI: Import cần thiết cho tạo mask chênh lệch >>>
from skimage.metrics import structural_similarity as ssim

try:
    from doctr.models import ocr_predictor
    DOCTR_AVAILABLE = True
except ImportError:
    DOCTR_AVAILABLE = False
try:
    from manga_ocr import MangaOcr
    MANGA_OCR_AVAILABLE = True
except ImportError:
    MANGA_OCR_AVAILABLE = False
from numba import jit
from numba.typed import List

# --- PHẦN KHỞI TẠO ---
FILE = Path(__file__).resolve(); ROOT = FILE.parents[0]
if str(ROOT) not in sys.path: sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))
try:
    from models.common import DetectMultiBackend
    from utils.general import non_max_suppression, scale_boxes
    from utils.augmentations import letterbox
    from utils.torch_utils import select_device
    YOLOV9_AVAILABLE = True
except ImportError: YOLOV9_AVAILABLE = False

OCR_MODELS = {}
MANGA_OCR_MODELS = {}

# --- CÁC HÀM TIỆN ÍCH VÀ VẼ (Giữ nguyên) ---
def hex_to_bgr(hex_color):
    try:
        rgb_float = to_rgb(hex_color); return (int(rgb_float[2] * 255), int(rgb_float[1] * 255), int(rgb_float[0] * 255))
    except (ValueError, TypeError): return (0, 255, 0)
def update_color_preview(colors_json_str):
    try:
        colors_dict = json.loads(colors_json_str)
        if not isinstance(colors_dict, dict): return "<p style='color: red;'>Lỗi: JSON phải là một đối tượng.</p>"
        html_output = ""
        for class_name, color_code in colors_dict.items():
            swatch_style = f"display: inline-block; width: 20px; height: 20px; background-color: {color_code}; border: 1px solid #ccc; vertical-align: middle; margin-right: 10px;"
            text_style = "font-family: monospace; font-weight: bold;"
            html_output += f"<div><span style='{swatch_style}'></span><span style='{text_style}'>{class_name}</span></div>"
        return html_output if html_output else "<p>Chưa có màu nào.</p>"
    except Exception as e: return f"<p style='color: red;'>Lỗi cú pháp JSON: {e}</p>"
def draw_custom_boxes(image_np, boxes, model_names, box_thickness, class_colors_str):
    img_draw = image_np.copy()
    try: custom_colors = json.loads(class_colors_str)
    except: custom_colors = {}
    default_colors = {}
    for cls_id, name in model_names.items():
        np.random.seed(hash(name) & 0xFFFFFFFF); default_colors[name] = [int(c) for c in np.random.randint(50, 255, size=3)]
    for box in boxes:
        x1, y1, x2, y2, conf, cls_id = box; x1, y1, x2, y2, cls_id = map(int, [x1, y1, x2, y2, cls_id])
        if cls_id not in model_names: continue
        class_name = model_names[cls_id]; label = f'{class_name} {conf:.2f}'
        color_str = custom_colors.get(class_name); color_bgr = hex_to_bgr(color_str) if color_str else default_colors.get(class_name, (0, 255, 0))
        cv2.rectangle(img_draw, (x1, y1), (x2, y2), color_bgr, thickness=box_thickness)
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        outside = y1 - h - 5 > 0; text_bg_y1, text_bg_y2 = (y1 - h - 5, y1) if outside else (y1, y1 + h + 5)
        cv2.rectangle(img_draw, (x1, text_bg_y1), (x1 + w, text_bg_y2), color_bgr, -1)
        cv2.putText(img_draw, label, (x1, y1 - 4 if outside else y1 + h + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
    return img_draw

@jit(nopython=True, cache=True)
def _get_box_area_numba(box):
    return (box[2] - box[0]) * (box[3] - box[1])

@jit(nopython=True, cache=True)
def calculate_iou_numba(box1, box2):
    x1_inter = max(box1[0], box2[0]); y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2]); y2_inter = min(box1[3], box2[3])
    inter_area = max(0.0, x2_inter - x1_inter) * max(0.0, y2_inter - y1_inter)
    if inter_area == 0.0: return 0.0
    box1_area = _get_box_area_numba(box1); box2_area = _get_box_area_numba(box2)
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0.0 else 0.0

@jit(nopython=True, cache=True)
def _check_containment_numba(box1, box2, thr):
    area1, area2 = _get_box_area_numba(box1), _get_box_area_numba(box2)
    if area1 == 0.0 or area2 == 0.0: return False
    bigger, smaller = (box1, box2) if area1 > area2 else (box2, box1)
    inter_x1, inter_y1 = max(bigger[0], smaller[0]), max(bigger[1], smaller[1])
    inter_x2, inter_y2 = min(bigger[2], smaller[2]), min(bigger[3], smaller[3])
    inter_area = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
    containment_ratio = inter_area / _get_box_area_numba(smaller)
    return containment_ratio >= thr

@jit(nopython=True, cache=True)
def _is_box_a_inside_b_numba(box_a, box_b, containment_threshold=0.8):
    x1_inter = max(box_a[0], box_b[0]); y1_inter = max(box_a[1], box_b[1])
    x2_inter = min(box_a[2], box_b[2]); y2_inter = min(box_a[3], box_b[3])
    inter_area = max(0.0, x2_inter - x1_inter) * max(0.0, y2_inter - y1_inter)
    if inter_area == 0.0: return False
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    if area_a == 0.0: return False
    return (inter_area / area_a) >= containment_threshold

# --- KẾT THÚC THÊM MỚI ---

class AdvancedDetectionPipeline:
    def __init__(self, class_mapping, config, image_size):
        self.class_mapping = class_mapping; self.reverse_class_mapping = {v: k for k, v in class_mapping.items()}; self.config = config
        self.image_width, self.image_height = image_size; self._map_names_to_ids()
    def _map_names_to_ids(self):
        groups = self.config.get('class_groups', {}); hier_config = self.config.get('intra_class_hierarchical_logic', {}); similar_groups_config = self.config.get('semantically_similar_groups', [])
        self.group_b_ids = {self.reverse_class_mapping.get(n) for n in groups.get('group_b', [])} - {None}
        self.hierarchical_class_ids = {self.reverse_class_mapping.get(n) for n in hier_config.get('enabled_classes', [])} - {None}
        self.similarity_map_numba = List()
        temp_map = {}
        for group in similar_groups_config:
            group_ids = {self.reverse_class_mapping.get(name) for name in group if name in self.reverse_class_mapping}
            for cls_id in group_ids: temp_map[cls_id] = group_ids
        for k, v_set in temp_map.items():
            if k is not None:
                v_list = List([float(item) for item in v_set if item is not None])
                self.similarity_map_numba.append((float(k), v_list))
        self.intra_group_filterable_ids = {self.reverse_class_mapping.get(cn) for gn in self.config.get('intra_group_filtering_enabled_groups', []) if gn in groups for cn in groups[gn] if cn in self.reverse_class_mapping}
        self.class_id_to_group_name = {self.reverse_class_mapping.get(cn): gn for gn in self.config.get('intra_group_filtering_enabled_groups', []) if gn in groups for cn in groups[gn] if cn in self.reverse_class_mapping}
        # --- BẮT ĐẦU THÊM MỚI ---
        # Lấy ID cho các class cần thiết cho luật thưởng/phạt
        self.text_id = self.reverse_class_mapping.get('text', -1)
        self.text2_id = self.reverse_class_mapping.get('text2', -1)
        self.group_a_ids = {self.reverse_class_mapping.get(n) for n in groups.get('group_a', [])} - {None}
        # --- KẾT THÚC THÊM MỚI ---
    def _custom_fusion(self, raw_results, iou_matrix_gpu):
        params = self.config['current_settings']; iou_thr = params['wbf_iou_thr']; containment_thr = params.get('parent_child_containment_thr', 0.8)
        all_boxes_list = List()
        model_index = 0
        for model_result in raw_results:
            for box in model_result:
                if box[4] >= params.get('wbf_skip_box_thr', 0.0):
                    all_boxes_list.append(np.array([box[0], box[1], box[2], box[3], box[4], float(box[5]), float(model_index)], dtype=np.float64))
            model_index += 1
        num_models_active = len(raw_results)
        if not all_boxes_list: return []
        iou_matrix_cpu = iou_matrix_gpu.cpu().numpy()
        clusters = _cluster_boxes_smart_numba(all_boxes_list, iou_thr, containment_thr, self.similarity_map_numba, iou_matrix_cpu)
        fused_boxes = List()
        for c_indices in clusters:
            if not c_indices: continue
            cluster = List([all_boxes_list[i] for i in c_indices])
            fused_box = _process_cluster_adaptive_numba(cluster, num_models_active)
            if fused_box is not None:
                fused_boxes.append(fused_box)
        return [[b[0], b[1], b[2], b[3], b[4], int(b[5])] for b in fused_boxes]

    # --- BẮT ĐẦU THÊM MỚI ---
    def _apply_containment_bonus_penalty(self, predictions):
        """Áp dụng luật thưởng/phạt dựa trên vị trí tương đối."""
        if not predictions or not self.group_a_ids or (self.text_id == -1 and self.text2_id == -1):
            return predictions

        # Phân loại các box để tối ưu hóa việc kiểm tra
        group_a_boxes, text_boxes, text2_boxes, other_boxes = [], [], [], []
        for p in predictions:
            class_id = p[5]
            if class_id in self.group_a_ids:
                group_a_boxes.append(p)
            elif class_id == self.text_id:
                text_boxes.append(p)
            elif class_id == self.text2_id:
                text2_boxes.append(p)
            else:
                other_boxes.append(p)

        if not group_a_boxes:
            return predictions

        modified_predictions = []

        # Xử lý text2 (bị phạt nếu nằm trong)
        for t2_box in text2_boxes:
            is_inside = False
            for a_box in group_a_boxes:
                if _is_box_a_inside_b_numba(np.array(t2_box[:4]), np.array(a_box[:4])):
                    t2_box[4] *= 0.7  # Phạt 30%
                    is_inside = True
                    break
            modified_predictions.append(t2_box)
            
        # Xử lý text (được thưởng nếu nằm trong)
        for t_box in text_boxes:
            is_inside = False
            for a_box in group_a_boxes:
                if _is_box_a_inside_b_numba(np.array(t_box[:4]), np.array(a_box[:4])):
                    t_box[4] *= 1.3  # Thưởng 30%
                    # Giới hạn điểm tin cậy không vượt quá 1.0
                    if t_box[4] > 1.0: t_box[4] = 1.0
                    is_inside = True
                    break
            modified_predictions.append(t_box)

        # Gộp tất cả các box lại
        return modified_predictions + group_a_boxes + other_boxes
    # --- KẾT THÚC THÊM MỚI ---

    def _filter_inter_class_overlap(self, predictions, iou_matrix_gpu):
        inter_class_iou_thr = self.config['current_settings']['ensemble_inter_class_iou_thr']
        preds_sorted = sorted(predictions, key=lambda x: x[4], reverse=True)
        if not preds_sorted: return []
        boxes_tensor = torch.tensor([p[:4] for p in preds_sorted], dtype=torch.float32, device=iou_matrix_gpu.device)
        class_ids = torch.tensor([p[5] for p in preds_sorted], device=iou_matrix_gpu.device)
        iou_matrix = ops.box_iou(boxes_tensor, boxes_tensor)
        num_boxes = len(preds_sorted)
        keep_mask = torch.ones(num_boxes, dtype=torch.bool, device=iou_matrix_gpu.device)
        group_b_ids_tensor = torch.tensor(list(self.group_b_ids), dtype=class_ids.dtype, device=class_ids.device)
        for i in range(num_boxes):
            if not keep_mask[i]: continue
            if torch.any(class_ids[i] == group_b_ids_tensor): continue
            is_not_group_b_j = ~torch.any(class_ids[i+1:, None] == group_b_ids_tensor, dim=1)
            is_overlapped_j = iou_matrix[i, i+1:] > inter_class_iou_thr
            suppress_mask = is_overlapped_j & is_not_group_b_j
            keep_mask[i+1:][suppress_mask] = False
        kept_indices = torch.where(keep_mask)[0].cpu().numpy()
        return [preds_sorted[i] for i in kept_indices]
    def _filter_intra_group_overlap(self, predictions):
        iou_thr = self.config['current_settings']['ensemble_intra_group_iou_thr']
        preds_sorted = sorted(predictions, key=lambda x: x[4], reverse=True); to_keep = [True] * len(preds_sorted)
        for i in range(len(preds_sorted)):
            if not to_keep[i]: continue
            box_i = preds_sorted[i]; group_i = self.class_id_to_group_name.get(box_i[5])
            if group_i is None: continue
            for j in range(i + 1, len(preds_sorted)):
                if not to_keep[j]: continue
                box_j = preds_sorted[j]; group_j = self.class_id_to_group_name.get(box_j[5])
                if group_j != group_i: continue
                if calculate_iou_numba(np.array(box_i[:4], dtype=np.float64), np.array(box_j[:4], dtype=np.float64)) > iou_thr: to_keep[j] = False
        return [preds_sorted[i] for i in range(len(preds_sorted)) if to_keep[i]]

    def process(self, raw_results, device_mode):
        logs = []; t_start = time.perf_counter()
        
        device = 'cuda' if device_mode == 'GPU' and torch.cuda.is_available() else 'cpu'

        t_step_start = time.perf_counter()
        all_boxes_for_iou = [box for res in raw_results for box in res]
        iou_matrix_gpu = torch.empty(0, 0, device=device)
        if len(all_boxes_for_iou) > 1:
            boxes_tensor = torch.tensor([b[:4] for b in all_boxes_for_iou], dtype=torch.float32).to(device)
            iou_matrix_gpu = ops.box_iou(boxes_tensor, boxes_tensor)
        logs.append(f"- Tính toán ma trận IoU ({len(all_boxes_for_iou)} boxes) trên {device.upper()}: {time.perf_counter() - t_step_start:.4f}s")
        
        t_step_start = time.perf_counter()
        fused_preds = self._custom_fusion(raw_results, iou_matrix_gpu)
        logs.append(f"- Fusion & Gom cụm (Numba + Pre-computed IoU): {time.perf_counter() - t_step_start:.4f}s")
        
        # --- BẮT ĐẦU THÊM MỚI ---
        t_step_start = time.perf_counter()
        bonus_penalty_preds = self._apply_containment_bonus_penalty(fused_preds)
        logs.append(f"- Áp dụng Luật Thưởng/Phạt Vị trí (CPU Numba): {time.perf_counter() - t_step_start:.4f}s")
        # --- KẾT THÚC THÊM MỚI ---

        t_step_start = time.perf_counter()
        inter_class_filtered_preds = []
        if bonus_penalty_preds: # <-- Sửa: Dùng kết quả từ bước thưởng/phạt
             fused_boxes_tensor = torch.tensor([p[:4] for p in bonus_penalty_preds], dtype=torch.float32).to(device)
             fused_iou_matrix_gpu = ops.box_iou(fused_boxes_tensor, fused_boxes_tensor)
             inter_class_filtered_preds = self._filter_inter_class_overlap(bonus_penalty_preds, fused_iou_matrix_gpu) # <-- Sửa
        logs.append(f"- Lọc GIỮA các Nhóm ({device.upper()}-accelerated): {time.perf_counter() - t_step_start:.4f}s")
        
        t_step_start = time.perf_counter()
        intra_group_filtered_preds = self._filter_intra_group_overlap(inter_class_filtered_preds)
        logs.append(f"- Lọc TRONG Nội Nhóm (CPU Numba): {time.perf_counter() - t_step_start:.4f}s")
        
        t_step_start = time.perf_counter()
        final_conf_thr = self.config['current_settings'].get('final_conf_threshold', 0.0)
        final_preds = [p for p in intra_group_filtered_preds if p[4] >= final_conf_thr]
        logs.append(f"- Lọc Tin cậy Cuối cùng: {time.perf_counter() - t_step_start:.4f}s")
        logs.insert(0, f"<b>Tổng thời gian Pipeline: {time.perf_counter() - t_start:.4f}s</b>")
        return final_preds, logs

# (Các hàm JIT và xử lý model đơn được giữ nguyên)
@jit(nopython=True, cache=True)
def _are_boxes_related_numba(box1_idx, box2_idx, boxes, iou_thr, containment_thr, similarity_map, iou_matrix):
    box1 = boxes[box1_idx]; box2 = boxes[box2_idx]; cls1, cls2 = int(box1[5]), int(box2[5])
    is_class_related = (cls1 == cls2)
    if not is_class_related:
        for k, v_list in similarity_map:
            if k == cls1:
                for v in v_list:
                    if v == cls2: is_class_related = True; break
                break
    if not is_class_related: return False
    if iou_matrix[box1_idx, box2_idx] >= iou_thr: return True
    if _check_containment_numba(box1[:4], box2[:4], containment_thr): return True
    return False
@jit(nopython=True, cache=True)
def _cluster_boxes_smart_numba(boxes, iou_thr, containment_thr, similarity_map, iou_matrix):
    n_boxes = len(boxes); clusters = List(); visited = np.zeros(n_boxes, dtype=np.bool_)
    for i in range(n_boxes):
        if visited[i]: continue
        current_cluster_indices = List(); q = List(); q.append(i); visited[i] = True
        while len(q) > 0:
            j = q.pop(0); current_cluster_indices.append(j)
            for k in range(n_boxes):
                if not visited[k] and _are_boxes_related_numba(j, k, boxes, iou_thr, containment_thr, similarity_map, iou_matrix):
                    visited[k] = True; q.append(k)
        clusters.append(current_cluster_indices)
    return clusters
@jit(nopython=True, cache=True)
def _process_cluster_adaptive_numba(cluster, num_models_active):
    if len(cluster) == 0: return None
    class_votes_keys = List(); class_votes_values = List()
    for box in cluster:
        cls_id = int(box[5]); score = box[4]; found = False
        for i, k in enumerate(class_votes_keys):
            if k == cls_id: class_votes_values[i] += score; found = True; break
        if not found: class_votes_keys.append(cls_id); class_votes_values.append(score)
    if len(class_votes_keys) == 0: return None
    max_vote = -1.0; winning_class = -1
    for i in range(len(class_votes_values)):
        if class_votes_values[i] > max_vote:
            max_vote = class_votes_values[i]; winning_class = class_votes_keys[i]
    winning_boxes = List()
    for b in cluster:
        if int(b[5]) == winning_class: winning_boxes.append(b)
    if len(winning_boxes) == 0: return None
    min_x1, min_y1, max_x2, max_y2 = winning_boxes[0][0], winning_boxes[0][1], winning_boxes[0][2], winning_boxes[0][3]
    for i in range(1, len(winning_boxes)):
        box = winning_boxes[i]
        min_x1 = min(min_x1, box[0]); min_y1 = min(min_y1, box[1])
        max_x2 = max(max_x2, box[2]); max_y2 = max(max_y2, box[3])
    fused_box_coords = [min_x1, min_y1, max_x2, max_y2]
    unique_models = List()
    for b in winning_boxes:
        model_idx = int(b[6]); is_in = False
        for um in unique_models:
            if um == model_idx: is_in = True; break
        if not is_in: unique_models.append(model_idx)
    num_contributing_models = len(unique_models); total_score_for_conf = 0.0
    for b in winning_boxes: total_score_for_conf += b[4]
    avg_score = total_score_for_conf / len(winning_boxes); final_score = 0.0
    if num_contributing_models == 1: final_score = avg_score * 0.4
    elif num_contributing_models == 2: final_score = avg_score * 0.8
    else: final_score = avg_score * 1.0
    return np.array([fused_box_coords[0], fused_box_coords[1], fused_box_coords[2], fused_box_coords[3], final_score, float(winning_class)])
def _filter_boxes_gpu(predictions_tensor, iou_diff_class_threshold):
    if predictions_tensor is None or predictions_tensor.shape[0] == 0: return []
    sorted_indices = predictions_tensor[:, 4].argsort(descending=True); preds_sorted = predictions_tensor[sorted_indices]
    boxes, scores, classes = preds_sorted[:, :4], preds_sorted[:, 4], preds_sorted[:, 5]
    keep_indices = torch.ones(len(preds_sorted), dtype=torch.bool, device=preds_sorted.device)
    for i in range(len(preds_sorted)):
        if not keep_indices[i]: continue
        iou = ops.box_iou(boxes[i:i+1], boxes[i+1:]); different_class_mask = classes[i] != classes[i+1:]
        overlap_mask = iou[0] > iou_diff_class_threshold; suppress_mask = different_class_mask & overlap_mask
        keep_indices[i+1:][suppress_mask] = False
    return preds_sorted[keep_indices].cpu().tolist()
def inference_and_draw(model_object, is_yolov9_original, image_pil, image_bgr, params):
    conf_threshold, iou_same_class_threshold, iou_diff_class_threshold, _, box_thickness, class_colors_str = params; t_total_start = time.perf_counter()
    boxes_after_nms_tensor, names = None, model_object.names
    if is_yolov9_original:
        stride, pt, imgsz = model_object.stride, model_object.pt, (640, 640)
        im = letterbox(image_bgr, imgsz, stride=stride, auto=pt)[0]; im = im.transpose((2, 0, 1))[::-1]; im = np.ascontiguousarray(im)
        im = torch.from_numpy(im).to(model_object.device).float() / 255.0
        if len(im.shape) == 3: im = im[None]
        pred = model_object(im, augment=False, visualize=False)[0]
        pred = non_max_suppression(pred, conf_threshold, iou_same_class_threshold, max_det=1000)[0]
        if pred is not None and len(pred):
            pred[:, :4] = scale_boxes(im.shape[2:], pred[:, :4], image_bgr.shape).round(); boxes_after_nms_tensor = pred
        else: boxes_after_nms_tensor = torch.empty(0, 6, device=model_object.device)
    else:
        results = model_object(image_pil, conf=conf_threshold, iou=iou_same_class_threshold, verbose=False)
        boxes_after_nms_tensor = results[0].boxes.data if len(results[0].boxes) > 0 else torch.empty(0, 6, device=model_object.device)
    final_boxes_for_single_model = _filter_boxes_gpu(boxes_after_nms_tensor, iou_diff_class_threshold)
    annotated_image = draw_custom_boxes(image_bgr, final_boxes_for_single_model, names, box_thickness, class_colors_str)
    return annotated_image, f"Tổng cộng: {time.perf_counter() - t_total_start:.3f} giây", final_boxes_for_single_model, names
def process_single_model(model_obj, is_original, image_pil, image_bgr, params, result_list, index):
    if model_obj is None: result_list[index] = (None, "Chưa chạy", [], {}); return
    try:
        result_list[index] = inference_and_draw(model_obj, is_original, image_pil, image_bgr, params)
    except Exception as e:
        print(f"Lỗi khi suy luận model {index+1}: {e}"); result_list[index] = (image_bgr.copy(), f"Lỗi: {e}", [], {})
        
# --- LOGIC OCR ---
def get_ocr_model(device_mode):
    global OCR_MODELS
    device_key = 'cuda' if device_mode == 'GPU' and torch.cuda.is_available() else 'cpu'
    if device_key in OCR_MODELS: return OCR_MODELS[device_key]
    if not DOCTR_AVAILABLE: raise ImportError("Thư viện 'python-doctr' chưa được cài đặt.")
    print(f"--- Lần đầu tải model OCR (docTR) cho thiết bị '{device_key}', vui lòng chờ... ---")
    model = ocr_predictor(det_arch='db_resnet50', reco_arch='crnn_vgg16_bn', pretrained=True).to(device_key)
    OCR_MODELS[device_key] = model
    print(f"--- Tải model OCR (docTR) cho '{device_key}' hoàn tất. ---")
    return model

def get_manga_ocr_model(device_mode):
    global MANGA_OCR_MODELS
    device_key = 'GPU' if device_mode == 'GPU' and torch.cuda.is_available() else 'CPU'
    if device_key in MANGA_OCR_MODELS: return MANGA_OCR_MODELS[device_key]
    if not MANGA_OCR_AVAILABLE: raise ImportError("Thư viện 'manga-ocr' chưa được cài đặt.")
    force_cpu_flag = (device_key == 'CPU')
    print(f"--- Lần đầu tải model OCR (Manga-OCR) cho thiết bị '{device_key}' (force_cpu={force_cpu_flag}), vui lòng chờ... ---")
    model = MangaOcr(force_cpu=force_cpu_flag)
    MANGA_OCR_MODELS[device_key] = model
    print(f"--- Tải model OCR (Manga-OCR) cho '{device_key}' hoàn tất. ---")
    return model

def _is_word_in_box(word_box, text_box, threshold=0.5):
    """Kiểm tra xem một 'word_box' có nằm trong 'text_box' hay không."""
    x1_inter = max(word_box[0], text_box[0]); y1_inter = max(word_box[1], text_box[1])
    x2_inter = min(word_box[2], text_box[2]); y2_inter = min(word_box[3], text_box[3])
    inter_area = max(0.0, x2_inter - x1_inter) * max(0.0, y2_inter - y1_inter)
    word_area = (word_box[2] - word_box[0]) * (word_box[3] - word_box[1])
    if word_area == 0: return False
    return (inter_area / word_area) > threshold

def run_ocr_and_refine_boxes(image_np_bgr, final_boxes_from_ensemble, model_names, device_mode):
    t_start = time.perf_counter(); logs = []
    try: ocr_model = get_ocr_model(device_mode)
    except Exception as e:
        logs.append(f"<p style='color: red;'>Lỗi khi tải model docTR OCR: {e}</p>")
        return final_boxes_from_ensemble, image_np_bgr.copy(), "", logs
    image_np_rgb = cv2.cvtColor(image_np_bgr, cv2.COLOR_BGR2RGB); t_step = time.perf_counter()
    ocr_result = ocr_model([image_np_rgb]); logs.append(f"- Thời gian OCR (docTR): {time.perf_counter() - t_step:.4f}s")
    h, w, _ = image_np_bgr.shape; all_words = []
    for page in ocr_result.pages:
        for block in page.blocks:
            for line in block.lines:
                for word in line.words:
                    x_min, y_min = word.geometry[0]; x_max, y_max = word.geometry[1]
                    abs_box = [int(x_min * w), int(y_min * h), int(x_max * w), int(y_max * h)]
                    all_words.append({'box': abs_box, 'text': word.value})
    text_class_id = next((cls_id for cls_id, name in model_names.items() if name == 'text'), -1)
    if text_class_id == -1:
        logs.append("- Không tìm thấy class 'text', bỏ qua tinh chỉnh."); return final_boxes_from_ensemble, image_np_bgr.copy(), "", logs
    refined_boxes, ocr_full_text_output = [], []
    yolo_text_boxes = [b for b in final_boxes_from_ensemble if b[5] == text_class_id]
    other_boxes = [b for b in final_boxes_from_ensemble if b[5] != text_class_id]
    for text_box in yolo_text_boxes:
        contained_words = [word for word in all_words if _is_word_in_box(word['box'], text_box[:4])]
        if contained_words:
            word_coords = np.array([w['box'] for w in contained_words])
            min_x, min_y = np.min(word_coords[:, [0, 1]], axis=0); max_x, max_y = np.max(word_coords[:, [2, 3]], axis=0)
            refined_boxes.append([min_x, min_y, max_x, max_y, text_box[4], text_box[5]])
            ocr_full_text_output.append(f"Box ({min_x},{min_y})-({max_x},{max_y}):\n{' '.join([w['text'] for w in contained_words])}\n")
        else: refined_boxes.append(text_box)
    final_refined_list = refined_boxes + other_boxes
    ocr_visualization_image = image_np_bgr.copy()
    for word in all_words: b = word['box']; cv2.rectangle(ocr_visualization_image, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 1)
    logs.insert(0, f"<b>Tổng thời gian OCR Pipeline (docTR): {time.perf_counter() - t_start:.4f}s</b>")
    return final_refined_list, ocr_visualization_image, "\n---\n".join(ocr_full_text_output), logs

def run_manga_ocr_on_boxes(image_np_bgr, final_boxes_from_ensemble, model_names, device_mode):
    t_start = time.perf_counter(); logs = []
    try: ocr_model = get_manga_ocr_model(device_mode)
    except Exception as e:
        logs.append(f"<p style='color: red;'>Lỗi khi tải model Manga-OCR: {e}</p>"); print(f"!!! LỖI TẢI MANGA-OCR: {e}")
        return final_boxes_from_ensemble, image_np_bgr.copy(), "", logs
    text_class_id = next((cls_id for cls_id, name in model_names.items() if name == 'text'), -1)
    if text_class_id == -1:
        logs.append("- Không tìm thấy class 'text', bỏ qua OCR."); return final_boxes_from_ensemble, image_np_bgr.copy(), "", logs
    ocr_full_text_output = []; image_pil = Image.fromarray(cv2.cvtColor(image_np_bgr, cv2.COLOR_BGR2RGB))
    ocr_visualization_image = image_np_bgr.copy()
    yolo_text_boxes = [box for box in final_boxes_from_ensemble if box[5] == text_class_id]
    for text_box in yolo_text_boxes:
        x1, y1, x2, y2, _, _ = map(int, text_box); x1, y1 = max(0, x1), max(0, y1); x2, y2 = min(image_pil.width, x2), min(image_pil.height, y2)
        if x1 >= x2 or y1 >= y2: continue
        cropped_pil = image_pil.crop((x1, y1, x2, y2))
        try:
            recognized_text = ocr_model(cropped_pil)
            ocr_full_text_output.append(f"Box ({x1},{y1})-({x2},{y2}):\n{recognized_text}\n")
            cv2.rectangle(ocr_visualization_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        except Exception as ocr_err:
            print(f"!!! LỖI KHI CHẠY MANGA-OCR trên box ({x1},{y1})-({x2},{y2}): {ocr_err}")
            ocr_full_text_output.append(f"Box ({x1},{y1})-({x2},{y2}):\nLỗi OCR: {ocr_err}\n")
            cv2.rectangle(ocr_visualization_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    logs.insert(0, f"<b>Tổng thời gian OCR Pipeline (Manga-OCR): {time.perf_counter() - t_start:.4f}s</b>")
    return final_boxes_from_ensemble, ocr_visualization_image, "\n---\n".join(ocr_full_text_output), logs

def create_mask_from_bboxes(image_shape, bboxes):
    """Tạo một mask nhị phân từ danh sách bounding box."""
    mask = np.zeros((image_shape[0], image_shape[1]), dtype=np.uint8)
    if not bboxes:
        return mask
    for box in bboxes:
        x1, y1, x2, y2 = map(int, box[:4])
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
    return mask

def create_difference_mask(img_raw, img_clean, blur_level, min_area, cleanup_level):
    """Tạo mask từ sự khác biệt pixel (Logic từ test.py)."""
    h, w, _ = img_raw.shape
    img_clean_resized = cv2.resize(img_clean, (w, h))
    gray_raw = cv2.cvtColor(img_raw, cv2.COLOR_BGR2GRAY)
    gray_clean = cv2.cvtColor(img_clean_resized, cv2.COLOR_BGR2GRAY)
    blur_level = blur_level + 1 if blur_level % 2 == 0 else blur_level
    blurred_raw = cv2.GaussianBlur(gray_raw, (blur_level, blur_level), 0)
    blurred_clean = cv2.GaussianBlur(gray_clean, (blur_level, blur_level), 0)
    (_, diff) = ssim(blurred_raw, blurred_clean, full=True)
    diff = (diff * 255).astype("uint8")
    thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    kernel_open = np.ones((3, 3), np.uint8)
    mask_opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_open, iterations=2)
    cleanup_level = cleanup_level + 1 if cleanup_level % 2 == 0 else cleanup_level
    kernel_close = np.ones((cleanup_level, cleanup_level), np.uint8)
    mask_closed = cv2.morphologyEx(mask_opened, cv2.MORPH_CLOSE, kernel_close, iterations=3)
    contours, _ = cv2.findContours(mask_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    final_mask = np.zeros_like(mask_closed)
    for cnt in contours:
        if cv2.contourArea(cnt) > min_area:
            cv2.drawContours(final_mask, [cnt], -1, (255), -1)
    return final_mask

def feather_mask(mask, expand_size, feather_amount):
    """Làm mịn và mở rộng viền mask (Logic từ test.py)."""
    expand_size = expand_size + 1 if expand_size % 2 == 0 else expand_size
    feather_amount = feather_amount + 1 if feather_amount % 2 == 0 else feather_amount
    if expand_size > 1:
        kernel = np.ones((expand_size, expand_size), np.uint8)
        dilated_mask = cv2.dilate(mask, kernel, iterations=1)
    else:
        dilated_mask = mask
    if feather_amount > 1:
        feathered_mask = cv2.GaussianBlur(dilated_mask, (feather_amount, feather_amount), 0)
    else:
        feathered_mask = dilated_mask
    return feathered_mask

def create_feathered_overlay(img_raw, img_clean, feathered_mask):
    """Tạo ảnh overlay từ mask đã làm mịn (Logic từ test.py)."""
    h, w, _ = img_raw.shape
    img_clean_resized = cv2.resize(img_clean, (w, h))
    alpha = cv2.cvtColor(feathered_mask, cv2.COLOR_GRAY2BGR).astype(float) / 255.0
    img_raw_float = img_raw.astype(float)
    img_clean_float = img_clean_resized.astype(float)
    result_float = cv2.multiply(alpha, img_clean_float) + cv2.multiply(1.0 - alpha, img_raw_float)
    return result_float.astype(np.uint8)


# <<< --- BẮT ĐẦU CẬP NHẬT --- >>>
def run_mask_generation_pipeline(raw_img_np, clean_img_np, selected_bboxes, unselected_bboxes, params):
    """
    Hàm điều phối chính cho việc tạo mask kết hợp với logic loại trừ thông minh.
    """
    # 1. Tạo mask từ bbox của các class ĐƯỢC CHỌN
    yolo_mask_selected = create_mask_from_bboxes(raw_img_np.shape, selected_bboxes)

    # 2. Tạo mask từ sự khác biệt pixel (như cũ)
    blur, min_area, cleanup = params['blur'], params['min_area'], params['cleanup']
    diff_mask = create_difference_mask(raw_img_np, clean_img_np, blur, min_area, cleanup)

    # 3. BƯỚC MỚI: TẠO MASK LOẠI TRỪ TỪ CÁC CLASS KHÔNG ĐƯỢC CHỌN
    # Tạo mask từ các bbox không được chọn
    yolo_mask_unselected = create_mask_from_bboxes(raw_img_np.shape, unselected_bboxes)
    # Đảo ngược mask này: vùng không được chọn -> đen, còn lại -> trắng
    exclusion_mask = cv2.bitwise_not(yolo_mask_unselected)
    
    # Áp dụng loại trừ: chỉ giữ lại chênh lệch pixel NẾU nó không nằm trong vùng không được chọn
    diff_mask_exclusive = cv2.bitwise_and(diff_mask, exclusion_mask)

    # 4. ÁP DỤNG PHÉP TOÁN BITWISE AND (giữa mask được chọn và mask chênh lệch đã loại trừ)
    combined_mask_unfeathered = cv2.bitwise_and(yolo_mask_selected, diff_mask_exclusive)

    # 5. Áp dụng hiệu chỉnh (làm mịn, mở rộng) trên mask đã kết hợp
    expand, feather = params['expand'], params['feather']
    final_mask_feathered = feather_mask(combined_mask_unfeathered, expand, feather)

    # 6. Tạo ảnh overlay cuối cùng để so sánh
    overlay_image = create_feathered_overlay(raw_img_np, clean_img_np, final_mask_feathered)

    # Trả về các ảnh để hiển thị (thay diff_mask bằng diff_mask_exclusive)
    return yolo_mask_selected, diff_mask_exclusive, final_mask_feathered, overlay_image
