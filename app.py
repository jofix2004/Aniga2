# app.py

import gradio as gr
import torch
import cv2
import numpy as np
import os
import threading
import time
import shutil
import filecmp
import importlib

# Import các hàm xử lý và quản lý config
import processing
from processing import *
import config_manager

# Tải lại để chắc chắn có phiên bản mới nhất
importlib.reload(config_manager)
importlib.reload(processing)

# --- TẢI CONFIG VÀ TẠO THƯ MỤC CACHE ---
CONFIG = config_manager.load_config()
MODEL_CACHE_DIR = "_model_cache"
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)


# --- CÁC HÀM XỬ LÝ CHO GIAO DIỆN ---
def save_preset(name, wbf_iou, wbf_conf, final_conf, inter_iou, intra_iou):
    global CONFIG
    if not name.strip():
        gr.Warning("Tên Preset không được để trống!"); return gr.update(choices=list(CONFIG['presets'].keys()))
    CONFIG['presets'][name] = {
        "wbf_iou_thr": wbf_iou, "wbf_skip_box_thr": wbf_conf,
        "final_conf_threshold": final_conf, "ensemble_inter_class_iou_thr": inter_iou,
        "ensemble_intra_group_iou_thr": intra_iou
    }
    if config_manager.save_config(CONFIG): gr.Info(f"Đã lưu Preset '{name}'!")
    else: gr.Error("Lưu Preset thất bại!")
    return gr.update(choices=list(CONFIG['presets'].keys()), value=name)

def load_preset(name):
    global CONFIG
    preset_values = CONFIG['presets'].get(name)
    if preset_values:
        return (
            preset_values.get('wbf_iou_thr', 0.55), preset_values.get('wbf_skip_box_thr', 0.01),
            preset_values.get('final_conf_threshold', 0.3), preset_values.get('ensemble_inter_class_iou_thr', 0.5),
            preset_values.get('ensemble_intra_group_iou_thr', 0.6)
        )
    return gr.update(), gr.update(), gr.update(), gr.update(), gr.update()

# --- HÀM ĐIỀU PHỐI CHÍNH ---
def main_predict_flow(click_start_time, *args):
    log_messages = []
    t_internal_start = time.perf_counter()
    *inputs, state_model1, state_model2, state_model3, state_image, state_loaded_model1, state_loaded_model2, state_loaded_model3 = args
    (model_path1, model_path2, model_path3, input_image,
     conf_threshold, iou_same_class_threshold, iou_diff_class_threshold,
     device_mode, box_thickness, class_colors_str, is_yolov9_original,
     enable_ensemble, wbf_iou_thr, wbf_skip_box_thr, final_conf_threshold, 
     ensemble_inter_class_iou_thr, ensemble_intra_group_iou_thr,
     ocr_mode
    ) = inputs
    
    model_files = [m.name if m else s for m, s in zip([model_path1, model_path2, model_path3], [state_model1, state_model2, state_model3])]
    current_image_pil = input_image if input_image else state_image
    
    if current_image_pil is None:
        empty_times = ["Chưa có ảnh đầu vào"] * 4
        yield (None, None, None, None, *empty_times, "Vui lòng cung cấp ảnh đầu vào.", *model_files, current_image_pil, *[state_loaded_model1, state_loaded_model2, state_loaded_model3], None, "")
        return

    current_image_np_bgr = cv2.cvtColor(np.array(current_image_pil), cv2.COLOR_RGB2BGR)
    log_messages.append(f"<b>[Bước 1] Chuẩn bị & Tải ảnh:</b> {time.perf_counter() - t_internal_start:.4f}s")
    
    params_single_model = (conf_threshold, iou_same_class_threshold, iou_diff_class_threshold, device_mode, box_thickness, class_colors_str)
    
    t_step_start = time.perf_counter()
    loaded_models = [state_loaded_model1, state_loaded_model2, state_loaded_model3]
    models_to_run = [None, None, None]
    for i, temp_model_path in enumerate(model_files):
        if not (temp_model_path and os.path.exists(temp_model_path)): continue
        cached_model = loaded_models[i]
        try:
            device_str = 'cuda:0' if device_mode == 'GPU' and torch.cuda.is_available() else 'cpu'
            target_device_obj = torch.device(device_str)
            if cached_model is None or cached_model.get('temp_path') != temp_model_path or cached_model.get('device') != device_mode:
                model_filename = os.path.basename(temp_model_path)
                persistent_model_path = os.path.join(MODEL_CACHE_DIR, f"cached_model_{i+1}_{model_filename}")
                if not os.path.exists(persistent_model_path) or not filecmp.cmp(temp_model_path, persistent_model_path, shallow=False):
                    shutil.copy(temp_model_path, persistent_model_path)
                
                log_messages.append(f"- Tải/Di chuyển Model {i+1} vào bộ nhớ ({device_mode})")
                t_load_start = time.perf_counter()
                is_original = (i == 1 and is_yolov9_original)
                if is_original:
                    model_obj = DetectMultiBackend(persistent_model_path, device=target_device_obj, data=ROOT / 'data/coco.yaml')
                else:
                    model_obj = YOLO(persistent_model_path)
                    model_obj.to(target_device_obj)
                loaded_models[i] = {'path': persistent_model_path, 'temp_path': temp_model_path, 'model': model_obj, 'device': device_mode}
                log_messages.append(f"  + Hoàn thành sau: {time.perf_counter() - t_load_start:.4f}s")

            if loaded_models[i]: models_to_run[i] = loaded_models[i]['model']
        except Exception as e:
            log_messages.append(f"<p style='color: red;'>- Lỗi nghiêm trọng ở Model {i+1}: {e}</p>")
            loaded_models[i] = None; continue
    log_messages.append(f"<b>[Bước 2] Tổng thời gian Tải/Cache/Di chuyển Models:</b> {time.perf_counter() - t_step_start:.4f}s")

    t_step_start = time.perf_counter()
    final_results = [(None, "Chưa chạy", [], {})] * 3
    model_infos = [(models_to_run[0], False), (models_to_run[1], is_yolov9_original), (models_to_run[2], False)]
    threads = [threading.Thread(target=process_single_model, args=(*info, current_image_pil, current_image_np_bgr, params_single_model, final_results, i)) for i, info in enumerate(model_infos)]
    for t in threads: t.start()
    for t in threads: t.join()
    log_messages.append(f"<b>[Bước 3] Tổng thời gian chạy 3 luồng suy luận:</b> {time.perf_counter() - t_step_start:.4f}s")

    ensemble_image, ensemble_time_str = None, "Ensemble không được bật"
    ocr_viz_image, ocr_text_output = None, "OCR không được bật."
    
    if enable_ensemble:
        t_ensemble_start = time.perf_counter()
        filtered_results_for_ensemble = [res[2] for res in final_results if res[2]]
        combined_names = {k: v for res in final_results for k, v in res[3].items()}
        
        if len(filtered_results_for_ensemble) > 1 and combined_names:
            h, w, _ = current_image_np_bgr.shape
            runtime_config = CONFIG.copy()
            runtime_config['current_settings'] = { "wbf_iou_thr": wbf_iou_thr, "wbf_skip_box_thr": wbf_skip_box_thr, "final_conf_threshold": final_conf_threshold, "parent_child_containment_thr": CONFIG['ensemble_defaults']['parent_child_containment_thr'], "ensemble_inter_class_iou_thr": ensemble_inter_class_iou_thr, "ensemble_intra_group_iou_thr": ensemble_intra_group_iou_thr }
            pipeline = AdvancedDetectionPipeline(class_mapping=combined_names, config=runtime_config, image_size=(w, h))
            
            final_boxes, pipeline_logs = pipeline.process(filtered_results_for_ensemble, device_mode)
            
            log_messages.append("--- Log Chi tiết Pipeline ---"); log_messages.extend(pipeline_logs)
            ensemble_time_str = pipeline_logs[0].replace("<b>", "").replace("</b>", "")

            if ocr_mode != "Không bật":
                log_messages.append(f"--- Log Chi tiết OCR ({ocr_mode}) ---")
                if ocr_mode == "Tiếng Anh (Tinh chỉnh box)":
                    final_boxes, ocr_viz_image, ocr_text_output, ocr_logs = run_ocr_and_refine_boxes(current_image_np_bgr, final_boxes, combined_names, device_mode)
                elif ocr_mode == "Tiếng Nhật (Chỉ trích xuất)":
                    final_boxes, ocr_viz_image, ocr_text_output, ocr_logs = run_manga_ocr_on_boxes(current_image_np_bgr, final_boxes, combined_names, device_mode)
                log_messages.extend(ocr_logs)

            ensemble_image = draw_custom_boxes(current_image_np_bgr, final_boxes, combined_names, box_thickness, class_colors_str)
        else: 
            ensemble_time_str = "Cần ít nhất 2 model có kết quả"
        log_messages.append(f"<b>[Bước 4] Tổng thời gian Ensemble & OCR:</b> {time.perf_counter() - t_ensemble_start:.4f}s")
    else:
        if ocr_mode != "Không bật":
            log_messages.append(f"<b style='color: orange;'>[Bước 4] CẢNH BÁO: Chế độ OCR '{ocr_mode}' đã được chọn nhưng Ensemble đang TẮT. => Chức năng OCR đã bị bỏ qua.</b>")
        else:
            log_messages.append("<b>[Bước 4] Ensemble & OCR:</b> Không được bật")

    images = [cv2.cvtColor(res[0], cv2.COLOR_BGR2RGB) if res and res[0] is not None else None for res in final_results]
    images.append(cv2.cvtColor(ensemble_image, cv2.COLOR_BGR2RGB) if ensemble_image is not None else None)
    times = [res[1] for res in final_results]; times.append(ensemble_time_str)
    
    internal_duration = time.perf_counter() - t_internal_start
    total_user_wait_time = time.perf_counter() - click_start_time
    
    log_header = (f"<h2>Tổng thời gian chờ (UI): {total_user_wait_time:.4f}s</h2>" f"<i>(Thời gian xử lý Backend là: {internal_duration:.4f}s.)</i>")
    log_messages.insert(0, log_header)
    final_log_str = "<br>".join(log_messages)
    
    final_ocr_viz_image = cv2.cvtColor(ocr_viz_image, cv2.COLOR_BGR2RGB) if ocr_viz_image is not None else None
    yield (*images, *times, final_log_str, *model_files, current_image_pil, *loaded_models, final_ocr_viz_image, ocr_text_output)


def main_flow_wrapper(*args):
    click_start_time = time.perf_counter()
    yield from main_predict_flow(click_start_time, *args)

# ---- GIAO DIỆN GRADIO ----
# <<< SỬA LỖI: Di chuyển `is_gpu_available` ra khỏi khối `with gr.Blocks` >>>
is_gpu_available = torch.cuda.is_available()
# <<< KẾT THÚC SỬA LỖI >>>

with gr.Blocks(theme=gr.themes.Soft(), css="footer {display: none !important;}") as demo:
    state_model1, state_model2, state_model3, state_image = gr.State(), gr.State(), gr.State(), gr.State()
    state_loaded_model1, state_loaded_model2, state_loaded_model3 = gr.State(), gr.State(), gr.State()
    default_colors_json = '{"b1":"#d62728","b2":"#1f77b4","b3":"#2ca02c","b4":"#ff7f0e","b5":"#9467bd","text":"#8c564b","text2":"#e377c2","text3":"#17becf"}'
    gr.Markdown("# Chương trình Test, So sánh và Ensemble các mô hình YOLO")
    with gr.Tabs():
        with gr.TabItem("Cài đặt chung & Xử lý đơn model"):
            single_model_defaults = CONFIG['single_model_defaults']
            with gr.Row():
                with gr.Column(scale=2):
                    gr.Markdown("#### **Thông số xử lý cho từng model**"); conf_threshold = gr.Slider(0.0, 1.0, value=single_model_defaults['conf_threshold'], step=0.05, label="Ngưỡng tự tin (Confidence)"); iou_same_class_threshold = gr.Slider(0.0, 1.0, value=single_model_defaults['iou_same_class_threshold'], step=0.05, label="Ngưỡng NMS (Cùng Class)"); iou_diff_class_threshold = gr.Slider(0.0, 1.0, value=single_model_defaults['iou_diff_class_threshold'], step=0.05, label="Ngưỡng lọc (Khác Class)")
                with gr.Column(scale=1):
                    gr.Markdown("#### **Cài đặt hệ thống**"); device_mode = gr.Radio(["CPU", "GPU"], value="GPU" if is_gpu_available else "CPU", label="Chế độ xử lý", interactive=is_gpu_available); is_yolov9_original = gr.Checkbox(label="Model 2 là YOLOv9 gốc"); box_thickness = gr.Slider(1, 20, 2, step=1, label="Độ đậm của Box")
            with gr.Row():
                with gr.Column(scale=2): class_colors_str = gr.Textbox(label="Bảng màu tùy chỉnh (JSON)", value=default_colors_json, lines=3)
                with gr.Column(scale=1): color_preview_display = gr.Markdown(label="Xem trước màu", value=processing.update_color_preview(default_colors_json))
        with gr.TabItem("Chế độ Ensemble"):
            ensemble_defaults = CONFIG['ensemble_defaults']
            enable_ensemble = gr.Checkbox(label="Bật Chế độ Ensemble", info="Kết hợp kết quả bằng Pipeline nâng cao.")
            ocr_mode = gr.Radio(
                ["Không bật", "Tiếng Anh (Tinh chỉnh box)", "Tiếng Nhật (Chỉ trích xuất)"],
                label="Chế độ OCR", value="Không bật", info="Sử dụng OCR để xử lý các box 'text'. Yêu cầu bật Ensemble."
            )
            with gr.Row():
                with gr.Column(scale=2):
                    gr.Markdown("#### **Thông số Pipeline Ensemble**"); wbf_iou_thr = gr.Slider(0.0, 1.0, value=ensemble_defaults['wbf_iou_thr'], step=0.05, label="Ngưỡng IoU cho WBF (Gom cụm)"); wbf_skip_box_thr = gr.Slider(0.0, 1.0, value=ensemble_defaults['wbf_skip_box_thr'], step=0.01, label="Ngưỡng Tin cậy để gộp (Skip Thr)"); final_conf_threshold = gr.Slider(0.0, 1.0, value=ensemble_defaults['final_conf_threshold'], step=0.05, label="Ngưỡng Tin cậy CUỐI CÙNG"); ensemble_inter_class_iou_thr = gr.Slider(0.0, 1.0, value=ensemble_defaults['ensemble_inter_class_iou_thr'], step=0.05, label="Ngưỡng Lọc GIỮA các Nhóm"); ensemble_intra_group_iou_thr = gr.Slider(0.0, 1.0, value=ensemble_defaults['ensemble_intra_group_iou_thr'], step=0.05, label="Ngưỡng Lọc TRONG Nội Nhóm")
                with gr.Column(scale=1):
                    gr.Markdown("#### **Quản lý Preset**"); preset_name = gr.Textbox(label="Tên Preset để lưu"); save_preset_btn = gr.Button("Lưu Preset hiện tại", variant="primary"); preset_dropdown = gr.Dropdown(label="Tải một Preset", choices=list(CONFIG['presets'].keys()), interactive=True)
        with gr.TabItem("Kết quả OCR") as ocr_tab:
            gr.Markdown("Tab này hiển thị kết quả từ pipeline OCR. Ảnh bên trái hiển thị các vùng được xử lý bởi OCR. Textbox bên phải hiển thị nội dung văn bản được trích xuất từ các box 'text'.")
            with gr.Row():
                output_image_ocr = gr.Image(label="Ảnh trực quan hóa OCR")
                ocr_text_output = gr.Textbox(label="Nội dung văn bản được trích xuất", lines=15, interactive=True)
    with gr.Row():
        input_image = gr.Image(type="pil", label="Ảnh đầu vào")
        with gr.Column():
            model_path1 = gr.File(label="Mô hình 1")
            model_path2 = gr.File(label="Mô hình 2")
            model_path3 = gr.File(label="Mô hình 3")
    run_button = gr.Button("Chạy Phân Tích", variant="primary")
    gr.Markdown("--- \n ### **Kết quả dự đoán**")
    with gr.Row():
        output_image1 = gr.Image(label="Kết quả 1"); time_label1 = gr.Label(label="Model 1 Time")
        output_image2 = gr.Image(label="Kết quả 2"); time_label2 = gr.Label(label="Model 2 Time")
        output_image3 = gr.Image(label="Kết quả 3"); time_label3 = gr.Label(label="Model 3 Time")
        with gr.Column(visible=False) as ensemble_col: 
            output_image4 = gr.Image(label="Kết quả Ensemble (Pipeline)"); time_label4 = gr.Label(label="Pipeline Time")
    with gr.Accordion("Báo cáo Thời gian Chi tiết", open=False): log_output = gr.Markdown("Nhấn 'Chạy Phân Tích' để xem báo cáo.")
    
    # Event Handlers
    enable_ensemble.change(lambda x: gr.update(visible=x), inputs=enable_ensemble, outputs=ensemble_col)
    class_colors_str.change(fn=processing.update_color_preview, inputs=class_colors_str, outputs=color_preview_display)
    save_preset_btn.click(fn=save_preset, inputs=[preset_name, wbf_iou_thr, wbf_skip_box_thr, final_conf_threshold, ensemble_inter_class_iou_thr, ensemble_intra_group_iou_thr], outputs=[preset_dropdown])
    preset_dropdown.change(fn=load_preset, inputs=[preset_dropdown], outputs=[wbf_iou_thr, wbf_skip_box_thr, final_conf_threshold, ensemble_inter_class_iou_thr, ensemble_intra_group_iou_thr])
    
    inputs_list = [ model_path1, model_path2, model_path3, input_image, conf_threshold, iou_same_class_threshold, iou_diff_class_threshold, device_mode, box_thickness, class_colors_str, is_yolov9_original, enable_ensemble, wbf_iou_thr, wbf_skip_box_thr, final_conf_threshold, ensemble_inter_class_iou_thr, ensemble_intra_group_iou_thr, ocr_mode ]
    states_list = [state_model1, state_model2, state_model3, state_image, state_loaded_model1, state_loaded_model2, state_loaded_model3]
    outputs_list = [ output_image1, output_image2, output_image3, output_image4, time_label1, time_label2, time_label3, time_label4, log_output, *states_list, output_image_ocr, ocr_text_output ]
    run_button.click(fn=main_flow_wrapper, inputs=inputs_list + states_list, outputs=outputs_list)

if __name__ == "__main__":
    demo.launch(share=True, debug=True)