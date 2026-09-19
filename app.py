"""
PCB缺陷检测系统 - Flask后端API
提供模型训练、推理、数据集统计、系统监控等功能
"""

import os
import io
import json
import time
import uuid
import threading
import base64
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import psutil
from PIL import Image
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename

from train import (
    analyze_dataset,
    analyze_defect_sizes,
    compute_class_weights,
    train_model,
    validate_model,
    export_model,
    predict_image,
    get_training_metrics,
    get_best_model_path,
    list_pretrained_models,
    pretrain_default_model,
    CLASS_NAMES,
    CLASS_LABELS_CN,
    CLASS_SEVERITY_WEIGHTS,
    RUNS_DIR,
    DATASET_DIR,
    BASE_DIR,
    EXPORT_DIR,
    PRETRAINED_DIR,
)

app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)

UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

# 缓存数据集统计信息，避免重复扫描
_dataset_stats_cache = {"data": None, "timestamp": 0}
_CACHE_DURATION = 300  # 5分钟缓存

# 全局训练状态
training_state = {
    "is_training": False,
    "progress": 0,
    "current_epoch": 0,
    "total_epochs": 0,
    "project_path": None,
    "project_name": None,
    "start_time": None,
    "status": "idle",
    "message": "",
    "model_size": "",
    "best_map50": 0,
    "best_map": 0,
}

# 当前激活的模型路径
active_model_path = None

# 告警记录
alert_log = []

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def allowed_file(filename):
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


# ============ 页面路由 ============

@app.route("/")
def index():
    return render_template("index.html")


# ============ 数据集API ============

@app.route("/api/dataset/stats", methods=["GET"])
def dataset_stats():
    """获取数据集统计信息（使用缓存）"""
    global _dataset_stats_cache
    
    # 检查缓存是否仍然有效
    current_time = time.time()
    if _dataset_stats_cache["data"] is not None and (current_time - _dataset_stats_cache["timestamp"]) < _CACHE_DURATION:
        return jsonify({"success": True, "data": _dataset_stats_cache["data"]})
    
    # 重新计算统计信息
    stats = analyze_dataset()
    _dataset_stats_cache["data"] = stats
    _dataset_stats_cache["timestamp"] = current_time
    
    return jsonify({"success": True, "data": stats})


@app.route("/api/dataset/samples", methods=["GET"])
def dataset_samples():
    """获取数据集样本图片（Base64编码）"""
    split = request.args.get("split", "train")
    count = min(int(request.args.get("count", 12)), 24)
    defect_type = request.args.get("defect_type", "all")

    img_dir = DATASET_DIR / split / "images"
    lbl_dir = DATASET_DIR / split / "labels"

    if not img_dir.exists():
        return jsonify({"success": False, "message": "数据集目录不存在"})

    img_files = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))

    if defect_type != "all":
        img_files = [f for f in img_files if defect_type in f.stem]

    # 均匀采样
    if len(img_files) > count:
        step = len(img_files) // count
        img_files = img_files[::step][:count]

    samples = []
    for img_path in img_files:
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        # 读取对应标签
        lbl_path = lbl_dir / (img_path.stem + ".txt")
        boxes = []
        if lbl_path.exists():
            h, w = img.shape[:2]
            with open(lbl_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls_id = int(parts[0])
                        cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                        x1 = int((cx - bw / 2) * w)
                        y1 = int((cy - bh / 2) * h)
                        x2 = int((cx + bw / 2) * w)
                        y2 = int((cy + bh / 2) * h)
                        boxes.append({
                            "class_id": cls_id,
                            "class_name": CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else "unknown",
                            "bbox": [x1, y1, x2, y2],
                        })
                        # 在图上画框
                        colors = [
                            (0, 255, 0), (255, 0, 0), (0, 0, 255),
                            (255, 255, 0), (0, 255, 255), (255, 0, 255),
                        ]
                        color = colors[cls_id % len(colors)]
                        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                        label = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else str(cls_id)
                        cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        _, buffer = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        img_b64 = base64.b64encode(buffer).decode("utf-8")

        samples.append({
            "filename": img_path.name,
            "image": img_b64,
            "boxes": boxes,
        })

    return jsonify({"success": True, "data": samples})


# ============ 模型训练API ============

@app.route("/api/train/start", methods=["POST"])
def start_training():
    """启动模型训练"""
    if training_state["is_training"]:
        return jsonify({"success": False, "message": "训练正在进行中"})

    params = request.get_json(silent=True) or {}
    model_size = params.get("model_size", "n")
    epochs = int(params.get("epochs", 100))
    batch_size = int(params.get("batch_size", 16))
    img_size = int(params.get("img_size", 640))
    lr0 = float(params.get("lr0", 0.01))
    optimizer = params.get("optimizer", "AdamW")
    patience = int(params.get("patience", 20))
    augment = params.get("augment", True)
    use_pretrained = params.get("use_pretrained", None)
    cos_lr = params.get("cos_lr", True)
    label_smoothing = float(params.get("label_smoothing", 0.05))
    multi_scale = params.get("multi_scale", False)

    training_state.update({
        "is_training": True,
        "progress": 0,
        "current_epoch": 0,
        "total_epochs": epochs,
        "start_time": datetime.now().isoformat(),
        "status": "training",
        "message": "正在初始化训练...",
        "model_size": model_size,
        "best_map50": 0,
        "best_map": 0,
    })

    def train_thread():
        try:
            model, results, proj_path, proj_name = train_model(
                model_size=model_size,
                epochs=epochs,
                batch_size=batch_size,
                img_size=img_size,
                lr0=lr0,
                optimizer=optimizer,
                patience=patience,
                augment=augment,#把数据增强开关传给训练。
                use_pretrained=use_pretrained,
                cos_lr=cos_lr,
                label_smoothing=label_smoothing,
                multi_scale=multi_scale,#把多尺度开关传给训练函数
            )
            training_state.update({
                "is_training": False,
                "progress": 100,
                "current_epoch": epochs,
                "project_path": proj_path,
                "project_name": proj_name,
                "status": "completed",
                "message": "训练完成！",
            })

            # 添加告警
            alert_log.append({
                "id": str(uuid.uuid4()),
                "time": datetime.now().isoformat(),
                "level": "info",
                "type": "training",
                "message": f"YOLOv8{model_size} 训练完成，共 {epochs} 轮",
            })
        except Exception as e:
            training_state.update({
                "is_training": False,
                "status": "error",
                "message": f"训练出错: {str(e)}",
            })
            alert_log.append({
                "id": str(uuid.uuid4()),
                "time": datetime.now().isoformat(),
                "level": "error",
                "type": "training",
                "message": f"训练错误: {str(e)}",
            })

    thread = threading.Thread(target=train_thread, daemon=True)
    thread.start()

    return jsonify({"success": True, "message": "训练已启动"})


@app.route("/api/train/stop", methods=["POST"])
def stop_training():
    """停止训练"""
    if not training_state["is_training"]:
        return jsonify({"success": False, "message": "当前没有训练在运行"})
    training_state.update({"is_training": False, "status": "stopped", "message": "训练已手动停止"})
    return jsonify({"success": True, "message": "训练停止信号已发送"})


@app.route("/api/train/status", methods=["GET"])
def train_status():
    """获取训练状态"""
    state = dict(training_state)

    # 尝试读取最新metrics
    if state["project_path"] and state["project_name"]:
        metrics = get_training_metrics(state["project_path"], state["project_name"])
        if metrics:
            state["current_epoch"] = len(metrics)
            state["progress"] = min(100, int(len(metrics) / max(state["total_epochs"], 1) * 100))
            latest = metrics[-1]
            state["latest_metrics"] = latest
    return jsonify({"success": True, "data": state})


@app.route("/api/train/metrics", methods=["GET"])
def train_metrics():
    """获取训练历史指标"""
    proj_path = request.args.get("project_path", training_state.get("project_path"))
    proj_name = request.args.get("project_name", training_state.get("project_name"))

    if not proj_path or not proj_name:
        # 尝试查找最新的训练目录
        detect_dir = RUNS_DIR / "detect"
        if detect_dir.exists():
            subdirs = sorted(detect_dir.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True)
            for sd in subdirs:
                csv_file = sd / "results.csv"
                if csv_file.exists():
                    proj_path = str(detect_dir)
                    proj_name = sd.name
                    break

    if not proj_path or not proj_name:
        return jsonify({"success": False, "message": "未找到训练记录"})

    metrics = get_training_metrics(proj_path, proj_name)
    if not metrics:
        return jsonify({"success": False, "message": "暂无训练数据"})

    return jsonify({"success": True, "data": metrics})


@app.route("/api/train/history", methods=["GET"])
def train_history():
    """获取所有训练历史"""
    detect_dir = RUNS_DIR / "detect"
    history = []

    if detect_dir.exists():
        for sd in sorted(detect_dir.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
            if not sd.is_dir():
                continue
            csv_file = sd / "results.csv"
            best_pt = sd / "weights" / "best.pt"
            entry = {
                "name": sd.name,
                "path": str(sd),
                "has_results": csv_file.exists(),
                "has_best": best_pt.exists(),
                "best_model": str(best_pt) if best_pt.exists() else None,
                "modified": datetime.fromtimestamp(sd.stat().st_mtime).isoformat(),
            }
            if csv_file.exists():
                metrics = get_training_metrics(str(detect_dir), sd.name)
                if metrics:
                    last = metrics[-1]
                    entry["epochs"] = len(metrics)
                    entry["last_metrics"] = last
            history.append(entry)

    return jsonify({"success": True, "data": history})


# ============ 推理API ============

@app.route("/api/detect", methods=["POST"])
def detect():
    """对上传图片进行缺陷检测"""
    if "image" not in request.files:
        return jsonify({"success": False, "message": "请上传图片"})

    file = request.files["image"]
    if not file or not allowed_file(file.filename):
        return jsonify({"success": False, "message": "不支持的文件格式"})

    conf = float(request.form.get("conf", 0.25))
    iou = float(request.form.get("iou", 0.45))
    model_path = request.form.get("model_path", None)

    # 查找可用模型
    if not model_path:
        model_path = _find_best_model()
    if not model_path or not Path(model_path).exists():
        return jsonify({"success": False, "message": "未找到可用模型，请先训练模型"})

    # 保存上传文件
    filename = secure_filename(file.filename)
    save_path = UPLOAD_DIR / f"{uuid.uuid4().hex}_{filename}"
    file.save(str(save_path))

    try:
        from ultralytics import YOLO
        model = YOLO(model_path)
        results = model.predict(source=str(save_path), conf=conf, iou=iou, imgsz=640)

        detections = []
        img = cv2.imread(str(save_path))
        h, w = img.shape[:2]

        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                confidence = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                class_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"class_{cls_id}"
                detections.append({
                    "class_id": cls_id,
                    "class_name": class_name,
                    "confidence": round(confidence, 4),
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                })

                # 画检测框
                colors = [
                    (0, 255, 0), (255, 0, 0), (0, 0, 255),
                    (255, 255, 0), (0, 255, 255), (255, 0, 255),
                ]
                color = colors[cls_id % len(colors)]
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                label = f"{class_name} {confidence:.2f}"
                cv2.putText(img, label, (int(x1), int(y1) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        _, buffer = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        result_b64 = base64.b64encode(buffer).decode("utf-8")

        # 生成告警
        if detections:
            high_severity = [d for d in detections if d["class_name"] in ("short", "open_circuit")]
            if high_severity:
                alert = {
                    "id": str(uuid.uuid4()),
                    "time": datetime.now().isoformat(),
                    "level": "critical",
                    "type": "defect",
                    "message": f"严重缺陷检测: {', '.join(d['class_name'] for d in high_severity)}",
                    "details": high_severity,
                }
                alert_log.append(alert)
            else:
                alert = {
                    "id": str(uuid.uuid4()),
                    "time": datetime.now().isoformat(),
                    "level": "warning",
                    "type": "defect",
                    "message": f"检测到 {len(detections)} 个缺陷: {', '.join(d['class_name'] for d in detections)}",
                    "details": detections,
                }
                alert_log.append(alert)

        return jsonify({
            "success": True,
            "data": {
                "detections": detections,
                "count": len(detections),
                "image": result_b64,
                "model": model_path,
                "image_size": [w, h],
            },
        })
    except Exception as e:
        return jsonify({"success": False, "message": f"检测失败: {str(e)}"})
    finally:
        if save_path.exists():
            save_path.unlink()


def _find_best_model():
    """查找最佳可用模型（优先预训练模型 > 训练模型）"""
    global active_model_path
    # 优先使用手动设置的激活模型
    if active_model_path and Path(active_model_path).exists():
        return active_model_path

    # 其次查找预训练模型目录
    if PRETRAINED_DIR.exists():
        pt_files = sorted(PRETRAINED_DIR.glob("*.pt"), key=lambda x: x.stat().st_mtime, reverse=True)
        if pt_files:
            return str(pt_files[0])

    # 最后查找训练输出
    detect_dir = RUNS_DIR / "detect"
    if detect_dir.exists():
        for sd in sorted(detect_dir.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
            best = sd / "weights" / "best.pt"
            if best.exists():
                return str(best)
    return None


# ============ 模型导出API ============
@app.route("/api/export", methods=["POST"])
def export_model_api():
    """导出模型"""
    params = request.get_json(silent=True) or {}
    model_path = params.get("model_path", _find_best_model())
    formats = params.get("formats", ["onnx"])

    if not model_path or not Path(model_path).exists():
        return jsonify({"success": False, "message": "未找到可用模型"})

    try:
        exported = export_model(model_path, formats=formats)
        return jsonify({"success": True, "data": exported})
    except Exception as e:
        return jsonify({"success": False, "message": f"导出失败: {str(e)}"})


# ============ 系统监控API ============

@app.route("/api/system/status", methods=["GET"])
def system_status():
    """获取系统状态"""
    try:
        cpu_percent = psutil.cpu_percent(interval=None)
        mem = psutil.virtual_memory()
        
        # 获取磁盘信息，添加错误处理
        try:
            disk = psutil.disk_usage(str(BASE_DIR))
            disk_info = {
                "total": round(disk.total / 1024**3, 2) if disk.total > 0 else 0,
                "used": round(disk.used / 1024**3, 2) if disk.used >= 0 else 0,
                "percent": round(disk.percent, 1) if disk.percent >= 0 else 0,
            }
        except Exception:
            disk_info = {"total": 0, "used": 0, "percent": 0}

        gpu_info = None
        try:
            import torch
            if torch.cuda.is_available():
                gpu_info = {
                    "name": torch.cuda.get_device_name(0),
                    "memory_total": round(torch.cuda.get_device_properties(0).total_mem / 1024**3, 2),
                    "memory_used": round(torch.cuda.memory_allocated(0) / 1024**3, 2),
                    "memory_free": round((torch.cuda.get_device_properties(0).total_mem - torch.cuda.memory_allocated(0)) / 1024**3, 2),
                }
        except Exception:
            pass

        return jsonify({
            "success": True,
            "data": {
                "cpu": {"percent": cpu_percent, "cores": psutil.cpu_count()},
                "memory": {
                    "total": round(mem.total / 1024**3, 2),
                    "used": round(mem.used / 1024**3, 2),
                    "percent": mem.percent,
                },
                "disk": disk_info,
                "gpu": gpu_info,
                "training": training_state.get("is_training", False),
            },
        })
    except Exception as e:
        return jsonify({
            "success": True,
            "data": {
                "cpu": {"percent": 0, "cores": 0},
                "memory": {"total": 0, "used": 0, "percent": 0},
                "disk": {"total": 0, "used": 0, "percent": 0},
                "gpu": None,
                "training": training_state.get("is_training", False),
            },
        })


# ============ 故障诊断API ============

@app.route("/api/diagnosis/analyze", methods=["POST"])
def diagnosis_analyze():
    """故障诊断分析"""
    params = request.get_json(silent=True) or {}
    detections = params.get("detections", [])

    if not detections:
        return jsonify({"success": True, "data": {"risk_level": "normal", "suggestions": ["当前无缺陷检出，系统运行正常。"]}})

    # 缺陷分析
    defect_counts = {}
    for d in detections:
        name = d.get("class_name", "unknown")
        defect_counts[name] = defect_counts.get(name, 0) + 1

    severity_map = {
        "short": {"level": "critical", "desc": "短路", "suggestion": "立即停止生产线，检查铜箔连接区域，可能导致PCB功能失效。建议使用AOI重新扫描确认范围。"},
        "open_circuit": {"level": "critical", "desc": "开路", "suggestion": "立即检查断裂走线，可能导致信号中断。建议进行电气连通性测试，必要时返工修复。"},
        "missing_hole": {"level": "warning", "desc": "缺孔", "suggestion": "检查钻孔工序参数设置，确认CNC钻孔程序和钻头磨损状态。"},
        "mouse_bite": {"level": "warning", "desc": "鼠咬痕", "suggestion": "检查V-Cut/拼板分割工艺参数，调整分板机刀具间距。"},
        "spur": {"level": "info", "desc": "毛刺", "suggestion": "检查蚀刻液浓度和温度参数，调整蚀刻时间。属于轻微缺陷，可通过后续工序修正。"},
        "spurious_copper": {"level": "info", "desc": "残铜", "suggestion": "检查蚀刻工序，确认干膜/湿膜附着力。清理蚀刻槽并更换蚀刻液。"},
    }

    issues = []
    risk_level = "normal"
    suggestions = []
    root_causes = []

    for defect_name, count in defect_counts.items():
        info = severity_map.get(defect_name, {"level": "info", "desc": defect_name, "suggestion": "请人工复核。"})
        issues.append({"defect": defect_name, "description": info["desc"], "count": count, "severity": info["level"]})
        suggestions.append(f"[{info['desc']}×{count}] {info['suggestion']}")

        if info["level"] == "critical":
            risk_level = "critical"
            root_causes.append(f"{info['desc']}可能由工艺参数异常或设备老化导致")
        elif info["level"] == "warning" and risk_level != "critical":
            risk_level = "warning"
            root_causes.append(f"{info['desc']}通常与加工精度下降有关")

    return jsonify({
        "success": True,
        "data": {
            "risk_level": risk_level,
            "issues": issues,
            "suggestions": suggestions,
            "root_causes": root_causes,
            "defect_summary": defect_counts,
            "total_defects": sum(defect_counts.values()),
            "timestamp": datetime.now().isoformat(),
        },
    })


@app.route("/api/diagnosis/report", methods=["GET"])
def diagnosis_report():
    """生成诊断报告"""
    # 收集所有可用数据
    stats = analyze_dataset()
    training_info = dict(training_state)
    recent_alerts = alert_log[-20:] if alert_log else []

    report = {
        "generated_at": datetime.now().isoformat(),
        "dataset": stats,
        "training": training_info,
        "recent_alerts": recent_alerts,
        "system_health": {
            "cpu": psutil.cpu_percent(),
            "memory": psutil.virtual_memory().percent,
        },
    }
    return jsonify({"success": True, "data": report})


# ============ 告警管理API ============

@app.route("/api/alerts", methods=["GET"])
def get_alerts():
    """获取告警列表"""
    limit = int(request.args.get("limit", 50))
    level = request.args.get("level", None)

    alerts = list(reversed(alert_log))
    if level:
        alerts = [a for a in alerts if a["level"] == level]

    return jsonify({"success": True, "data": alerts[:limit]})


@app.route("/api/alerts/clear", methods=["POST"])
def clear_alerts():
    """清除告警"""
    alert_log.clear()
    return jsonify({"success": True, "message": "告警已清除"})


# ============ 模型管理API ============

@app.route("/api/models", methods=["GET"])
def list_models():
    """列出所有可用模型"""
    models = []
    detect_dir = RUNS_DIR / "detect"

    if detect_dir.exists():
        for sd in sorted(detect_dir.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
            best = sd / "weights" / "best.pt"
            last = sd / "weights" / "last.pt"
            if best.exists():
                models.append({
                    "name": sd.name,
                    "path": str(best),
                    "type": "best",
                    "size_mb": round(best.stat().st_size / 1024**2, 2),
                    "modified": datetime.fromtimestamp(best.stat().st_mtime).isoformat(),
                })
            if last.exists():
                models.append({
                    "name": f"{sd.name} (last)",
                    "path": str(last),
                    "type": "last",
                    "size_mb": round(last.stat().st_size / 1024**2, 2),
                    "modified": datetime.fromtimestamp(last.stat().st_mtime).isoformat(),
                })

    # 导出模型
    if EXPORT_DIR.exists():
        for f in EXPORT_DIR.iterdir():
            if f.suffix in (".onnx", ".engine", ".tflite"):
                models.append({
                    "name": f.name,
                    "path": str(f),
                    "type": "exported",
                    "size_mb": round(f.stat().st_size / 1024**2, 2),
                    "modified": datetime.fromtimestamp(f.stat().st_mtime).isoformat(),
                })

    return jsonify({"success": True, "data": models})


# ============ 预训练模型管理API ============

@app.route("/api/pretrained", methods=["GET"])
def get_pretrained_models():
    """列出所有预训练模型"""
    models = list_pretrained_models()
    return jsonify({"success": True, "data": models})


@app.route("/api/pretrained/load", methods=["POST"])
def load_pretrained_model():
    """加载指定的预训练模型为当前激活模型"""
    global active_model_path
    params = request.get_json(silent=True) or {}
    model_path = params.get("model_path")

    if not model_path or not Path(model_path).exists():
        return jsonify({"success": False, "message": "模型文件不存在"})

    active_model_path = model_path
    alert_log.append({
        "id": str(uuid.uuid4()),
        "time": datetime.now().isoformat(),
        "level": "info",
        "type": "model",
        "message": f"已加载预训练模型: {Path(model_path).name}",
    })
    return jsonify({"success": True, "message": f"已加载模型: {Path(model_path).name}", "model_path": model_path})


@app.route("/api/pretrained/current", methods=["GET"])
def get_current_model():
    """获取当前激活的模型信息"""
    model_path = _find_best_model()
    if model_path:
        p = Path(model_path)
        return jsonify({
            "success": True,
            "data": {
                "path": model_path,
                "name": p.stem,
                "size_mb": round(p.stat().st_size / 1024**2, 2),
                "source": "pretrained" if str(PRETRAINED_DIR) in model_path else "trained",
            }
        })
    return jsonify({"success": True, "data": None, "message": "暂无可用模型"})


@app.route("/api/dataset/defect-sizes", methods=["GET"])
def defect_sizes():
    """获取缺陷尺寸分析数据"""
    sizes = analyze_defect_sizes()
    return jsonify({"success": True, "data": sizes})


@app.route("/api/dataset/class-weights", methods=["GET"])
def class_weights():
    """获取类别平衡权重"""
    weights = compute_class_weights()
    return jsonify({"success": True, "data": dict(zip(CLASS_NAMES, weights))})


@app.route("/api/innovation", methods=["GET"])
def innovation_info():
    """返回模型创新点介绍"""
    innovations = [
        {
            "id": 1,
            "title": "CBAM注意力机制 (Channel & Spatial Attention)",
            "description": "通过通道注意力和空间注意力双重增强，提升小目标缺陷（如鼠咬痕、毛刺）的特征表达能力，有效提高微小缺陷检测精度。",
            "category": "模型结构",
        },
        {
            "id": 2,
            "title": "自适应PCB缺陷锚框优化",
            "description": "基于训练集标注自动分析缺陷尺寸分布，使用K-Means聚类生成PCB缺陷专用锚框，提升检测框的回归精度。",
            "category": "先验知识",
        },
        {
            "id": 3,
            "title": "类别平衡Focal Loss + 严重等级加权",
            "description": "针对PCB缺陷数据集6类分布不均衡问题，结合逆频率加权和缺陷严重等级（短路/开路权重更高）动态调节分类损失。",
            "category": "损失函数",
        },
        {
            "id": 4,
            "title": "多尺度训练策略 (Multi-Scale Training)",
            "description": "训练时随机切换480/640/800分辨率，增强模型对不同尺寸PCB板和不同距离拍摄图像的适应性。",
            "category": "训练策略",
        },
        {
            "id": 5,
            "title": "PCB工业场景专用数据增强",
            "description": "针对铜箔特征的HSV色调增强、亮度抖动（模拟不同光照），Mosaic/MixUp/Copy-Paste缺陷增强管线。",
            "category": "数据增强",
        },
        {
            "id": 6,
            "title": "模型轻量化 + INT8量化边缘部署",
            "description": "训练后自动剪枝+INT8量化，适配Ubuntu AI边缘开发板，实现端侧实时检测（>30FPS）。",
            "category": "部署优化",
        },
    ]
    return jsonify({"success": True, "data": innovations})


# 训练结果图片
@app.route("/api/train/plots/<path:filename>")
def get_train_plot(filename):
    """获取训练结果图表"""
    detect_dir = RUNS_DIR / "detect"
    if not detect_dir.exists():
        return jsonify({"success": False}), 404

    # 查找最新训练目录
    for sd in sorted(detect_dir.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
        plot_path = sd / filename
        if plot_path.exists():
            return send_from_directory(str(sd), filename)

    return jsonify({"success": False}), 404


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  PCB缺陷检测智能系统 v1.0")
    print("  2026广东省大学生计算机设计大赛")
    print("=" * 60)
    print(f"  数据集路径: {DATASET_DIR}")
    print(f"  类别数: {len(CLASS_NAMES)}")
    print(f"  类别: {', '.join(CLASS_NAMES)}")
    print("=" * 60)
    print("  访问地址: http://localhost:5000")
    print("=" * 60 + "\n")
    app.run(host="0.0.0.0", port=5000, debug=True)
