# config_manager.py

import json
import os

CONFIG_FILE = 'ensemble_config.json'

def get_default_config():
    """Trả về cấu trúc config mặc định nếu file không tồn tại."""
    return {
        "single_model_defaults": {
            "conf_threshold": 0.25,
            "iou_same_class_threshold": 0.45,
            "iou_diff_class_threshold": 0.7 
        },
        "ensemble_defaults": {
            "wbf_iou_thr": 0.55,
            "wbf_skip_box_thr": 0.01,
            "final_conf_threshold": 0.3,
            "parent_child_containment_thr": 0.8,
            "ensemble_inter_class_iou_thr": 0.5,
            "ensemble_intra_group_iou_thr": 0.6 # <<< THÊM MỚI: Ngưỡng lọc nội nhóm
        },
        "class_groups": { 
            "group_a": ["b1", "b2", "b3", "b4", "b5"],
            "group_b": ["text"],
            "group_c_peer": ["text2", "text"]
        },
        # <<< THÊM MỚI: Định nghĩa các nhóm sẽ áp dụng lọc nội bộ >>>
        "intra_group_filtering_enabled_groups": [
            "group_a", "group_c_peer" 
        ],
        "semantically_similar_groups": [ ["text2", "text3"] ],
        "intra_class_hierarchical_logic": {
            "enabled_classes": ["text", "text2", "text3"],
            "parent_min_score_for_discard": 0.2, "child_min_score_for_override": 0.6
        },
        "presets": {
            "Mặc định (Cân bằng)": {
                "wbf_iou_thr": 0.55, "wbf_skip_box_thr": 0.01,
                "final_conf_threshold": 0.3, "ensemble_inter_class_iou_thr": 0.5,
                "ensemble_intra_group_iou_thr": 0.6
            },
            "Ưu tiên Chính xác (Strict)": {
                "wbf_iou_thr": 0.5, "wbf_skip_box_thr": 0.05,
                "final_conf_threshold": 0.5, "ensemble_inter_class_iou_thr": 0.4,
                "ensemble_intra_group_iou_thr": 0.5
            }
        }
    }

def load_config():
    if not os.path.exists(CONFIG_FILE):
        default_config = get_default_config(); save_config(default_config); return default_config
    try:
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f: return json.load(f)
    except (json.JSONDecodeError, IOError): return get_default_config()

def save_config(data):
    try:
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f: json.dump(data, f, indent=4, ensure_ascii=False); return True
    except IOError: return False