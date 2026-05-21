"""后台 YOLO 训练：路径、设备、日志监控、子进程调度。"""
import csv
import re
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from werkzeug.utils import secure_filename

from app import db
from app.models import TrainingJob

def get_training_root(cfg) -> Path:
    """训练目录 backend/training（可由环境变量 TRAINING_ROOT 覆盖）。"""
    backend_root = Path(cfg.get('BACKEND_ROOT') or cfg.get('PROJECT_ROOT') or '.')
    raw = cfg.get('TRAINING_ROOT')
    if raw:
        p = Path(str(raw))
        if p.is_absolute():
            return p.resolve()
        return (backend_root / p).resolve()
    return (backend_root / 'training').resolve()

def resolve_training_weights_file(
    upload_root: Path,
    rel: str | None,
    training_root: Path,
) -> Path:
    """
    rel 为空：抛出错误，须由管理员显式选择或上传训练起点权重。
    rel 为相对路径：training/bases/<文件名>
    不得使用 uploads/detect_model/（仅在线推理）。
    """
    upload_root = upload_root.resolve()
    training_root = training_root.resolve()

    if not rel or not str(rel).strip():
        raise ValueError('请选择训练初始权重（上传 .pt / .pth 后在下拉框中选择）')

    rel = str(rel).strip().replace('\\', '/').lstrip('/')
    if '..' in rel or rel.startswith(('/', '\\')):
        raise ValueError('非法权重路径')

    suf = Path(rel).suffix.lower()
    if suf not in ('.pt', '.pth'):
        raise ValueError('仅支持 .pt / .pth 权重文件')

    if rel.startswith('detect_model/'):
        raise ValueError('detect_model/ 为在线推理权重目录，不能用作训练起点')

    if rel.startswith('training/bases/'):
        full = (training_root / 'bases' / rel[len('training/bases/') :]).resolve()
        try:
            full.relative_to((training_root / 'bases').resolve())
        except ValueError:
            raise ValueError('路径超出 training/bases') from None
        if not full.is_file():
            raise ValueError('请先上传模型权重文件')
        return full

    raise ValueError('仅允许 training/bases/ 中的训练权重')

def training_run_name(base_weights_rel: str, job_id: int) -> str:
    """运行名：{权重文件名stem}_{training_job.id}，如 best_12。"""
    name = Path(str(base_weights_rel or '').replace('\\', '/')).name or 'weights.pt'
    stem = secure_filename(Path(name).stem or 'weights') or 'weights'
    return f'{stem}_{int(job_id)}'

def run_name_for_job(job: TrainingJob) -> str:
    """从 log_path / output_weights_path 解析运行名；否则用 {stem}_{job.id}。"""
    raw = (job.log_path or '').strip()
    if raw:
        return Path(raw).stem
    out_raw = str(job.output_weights_path or '').strip().replace('\\', '/')
    if out_raw:
        out_p = Path(out_raw)
        if out_p.parent.name.lower() == 'weights' and out_p.parent.parent.name:
            return out_p.parent.parent.name
    if job.id:
        return training_run_name(job.base_weights_rel or '', job.id)
    return ''

def _results_csv_candidates(train_project_dir: Path, job: TrainingJob) -> list[Path]:
    """列出 results.csv 路径：training/runs/<权重名>_<序号>/results.csv。"""
    run_names: list[str] = []
    log_stem = run_name_for_job(job)
    if log_stem:
        run_names.append(log_stem)

    out_raw = str(job.output_weights_path or '').strip().replace('\\', '/')
    if out_raw:
        out_p = Path(out_raw)
        if out_p.parent.name.lower() == 'weights' and out_p.parent.parent.name:
            run_names.append(out_p.parent.parent.name)

    seen: set[str] = set()
    paths: list[Path] = []
    for name in run_names:
        if not name or name in seen:
            continue
        seen.add(name)
        paths.append(train_project_dir / name / 'results.csv')
    return paths

def resolve_results_csv_for_job(train_project_dir: Path, job: TrainingJob) -> Path | None:
    for p in _results_csv_candidates(train_project_dir, job):
        if p.is_file():
            return p
    return None

def enrich_training_job_dict(job: TrainingJob, cfg) -> dict[str, Any]:
    """为 API 返回补充运行名、是否可读指标等字段。"""
    d = job.to_dict()
    train_dir = resolve_train_project_dir(cfg)
    run = run_name_for_job(job)
    csv_path = resolve_results_csv_for_job(train_dir, job)
    d['run_name'] = run
    d['has_metrics'] = csv_path is not None
    if csv_path:
        d['metrics_run_name'] = csv_path.parent.name
    else:
        d['metrics_run_name'] = ''
    base = str(d.get('base_weights_rel') or '').replace('\\', '/')
    d['base_weights_name'] = Path(base).name if base else ''
    return d

"""训练用设备探测（与发起训练同一 Python 环境）。"""

def collect_training_device_info() -> dict:
    out: dict = {
        'torch_installed': False,
        'torch_version': None,
        'cuda_available': False,
        'cuda_device_count': 0,
        'cuda_devices': [],
        'mps_available': False,
        'error': None,
    }
    try:
        import torch

        out['torch_installed'] = True
        out['torch_version'] = str(getattr(torch, '__version__', '') or '')
        out['cuda_available'] = bool(torch.cuda.is_available())
        n = int(torch.cuda.device_count()) if out['cuda_available'] else 0
        out['cuda_device_count'] = n
        devices = []
        for i in range(n):
            try:
                name = str(torch.cuda.get_device_name(i) or '')
            except Exception:
                name = ''
            devices.append({'index': i, 'name': name})
        out['cuda_devices'] = devices
        mps = getattr(torch.backends, 'mps', None)
        if mps is not None and callable(getattr(mps, 'is_available', None)):
            out['mps_available'] = bool(mps.is_available())
    except Exception as e:
        out['error'] = str(e)
    out['default_device'] = _recommended_training_device(out)
    return out

def _recommended_training_device(info: dict) -> str:
    """与训练 device 参数一致：优先首张 CUDA GPU，其次 MPS，否则 cpu。"""
    devices = info.get('cuda_devices') or []
    if devices:
        return str(devices[0].get('index', 0))
    if info.get('mps_available'):
        return 'mps'
    return 'cpu'

_ANSI = re.compile(r'\x1b\[[0-?]*[ -/]*[@-~]')

def strip_ansi(s: str) -> str:
    return _ANSI.sub('', s)

def read_file_tail(path: Path, max_bytes: int = 98304) -> str:
    if not path.is_file():
        return ''
    with open(path, 'rb') as f:
        f.seek(0, 2)
        sz = f.tell()
        f.seek(max(0, sz - max_bytes))
        raw = f.read()
    return raw.decode('utf-8', errors='replace')

def parse_log_epoch(log_tail: str, total_epochs: int) -> dict:
    """从日志尾部解析当前轮次与阶段。"""
    out = {
        'phase': 'unknown',
        'epoch': None,
        'epochs': total_epochs,
    }
    if not log_tail or total_epochs < 1:
        return out

    text = strip_ansi(log_tail)
    if 'Starting training for' in text:
        out['phase'] = 'training'
    elif 'Scanning' in text and 'labels' in text:
        out['phase'] = 'scanning'
    elif 'engine\\trainer' in text or 'engine/trainer' in text:
        out['phase'] = 'init'

    epoch_pat = re.compile(r'\s+(\d+)/(\d+)\s+\d+(?:\.\d+)?G\s+[\d.]+')
    em = list(epoch_pat.finditer(text))
    if em:
        out['epoch'] = int(em[-1].group(1))
        out['epochs'] = int(em[-1].group(2)) or total_epochs
    return out

def _read_csv_last_epoch(csv_path: Path) -> int | None:
    if not csv_path.is_file():
        return None
    try:
        with open(csv_path, 'r', encoding='utf-8', newline='') as f:
            rows = list(csv.DictReader(f))
    except OSError:
        return None
    if not rows:
        return None
    try:
        return int(float(rows[-1].get('epoch') or 0))
    except (TypeError, ValueError):
        return None

def _resolve_current_epoch(log_epoch: int | None, csv_epoch: int | None, total_epochs: int) -> int | None:
    """日志中的当前轮次优先；验证阶段日志无 epoch 行时回退 csv 已完成轮次。"""
    total_epochs = max(int(total_epochs or 0), 1)
    if log_epoch and log_epoch > 0:
        return min(int(log_epoch), total_epochs)
    if csv_epoch and csv_epoch > 0:
        return min(int(csv_epoch), total_epochs)
    return None

def _compute_progress_and_eta(
    current_epoch: int,
    total_epochs: int,
    started_at: datetime | None,
) -> tuple[float | None, float | None]:
    """
    进度% = (当前轮数 / 总轮数) * 100
    每轮平均时间 = 已用总时间 / 当前轮数
    剩余总时间 = 每轮平均时间 × (总轮数 - 当前轮数)
    """
    total_epochs = max(int(total_epochs or 0), 1)
    current_epoch = max(1, min(int(current_epoch), total_epochs))
    progress_pct = round(100.0 * current_epoch / float(total_epochs), 2)

    if not started_at:
        return progress_pct, None
    elapsed = (datetime.utcnow() - started_at).total_seconds()
    if elapsed <= 0:
        return progress_pct, None

    avg_epoch_sec = elapsed / float(current_epoch)
    remaining_epochs = max(0, total_epochs - current_epoch)
    eta_seconds = max(0.0, avg_epoch_sec * remaining_epochs)
    return progress_pct, eta_seconds

def resolve_train_project_dir(cfg) -> Path:
    pd = Path(cfg['TRAIN_PROJECT_DIR'])
    if pd.is_absolute():
        return pd.resolve()
    training_root = get_training_root(cfg)
    return (training_root / pd).resolve()

def _training_log_roots(cfg) -> list[Path]:
    roots: list[Path] = []
    training_root = get_training_root(cfg)
    logs_rel = Path(cfg.get('TRAIN_LOGS_DIR') or 'logs')
    if logs_rel.is_absolute():
        logs = logs_rel.resolve()
    else:
        logs = (training_root / logs_rel).resolve()
    roots.append(logs)
    roots.append((training_root / 'logs').resolve())
    seen = set()
    out: list[Path] = []
    for r in roots:
        key = str(r)
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out

def _safe_resolve_log_path(cfg, log_path: str | None) -> Path | None:
    if not log_path:
        return None
    p = Path(log_path)
    if not p.is_absolute():
        p = (get_training_root(cfg) / p).resolve()
    else:
        p = p.resolve()
    for allowed in _training_log_roots(cfg):
        try:
            p.relative_to(allowed)
        except ValueError:
            continue
        return p
    return None

def build_training_live_payload(project_root: Path, cfg, job) -> dict:
    """供轮询：解析当前轮次、进度与 ETA。"""
    status = str(getattr(job, 'status', '') or '')

    tail_full = ''
    log_path = _safe_resolve_log_path(cfg, job.log_path)
    if log_path:
        tail_full = read_file_tail(log_path, 98304)
    try:
        epochs = int(job.epochs or cfg.get('TRAIN_DEFAULT_EPOCHS') or 30)
    except (TypeError, ValueError):
        epochs = 30

    log_info = parse_log_epoch(tail_full, epochs)
    train_project_dir = resolve_train_project_dir(cfg)
    csv_path = resolve_results_csv_for_job(train_project_dir, job)
    csv_epoch = _read_csv_last_epoch(csv_path) if csv_path else None

    progress = {
        'phase': log_info.get('phase', 'unknown'),
        'epoch': log_info.get('epoch'),
        'epochs': epochs,
        'progress_pct': None,
        'eta_seconds': None,
    }

    if status in ('succeeded', 'failed', 'terminated'):
        if status == 'succeeded':
            progress['epoch'] = epochs
            progress['progress_pct'] = 100.0
            progress['eta_seconds'] = 0.0
        return {'progress': progress}

    if status not in ('running', 'queued'):
        return {'progress': progress}

    current_epoch = _resolve_current_epoch(log_info.get('epoch'), csv_epoch, epochs)
    if current_epoch:
        pct, eta = _compute_progress_and_eta(current_epoch, epochs, job.started_at)
        progress['epoch'] = current_epoch
        progress['progress_pct'] = pct
        progress['eta_seconds'] = eta

    return {'progress': progress}

def _pick_float(row: dict, *keys) -> float | None:
    for k in keys:
        if k in row and row[k] not in (None, ''):
            try:
                return float(row[k])
            except (TypeError, ValueError):
                continue
    return None

def _row_to_metrics(row: dict, csv_path: Path) -> dict:
    map50 = _pick_float(row, 'metrics/mAP50(B)', 'mAP50(B)', 'metrics/mAP_0.5')
    prec = _pick_float(row, 'metrics/precision(B)', 'precision(B)')
    rec = _pick_float(row, 'metrics/recall(B)', 'recall(B)')

    tb = _pick_float(row, 'train/box_loss')
    tc = _pick_float(row, 'train/cls_loss')
    td = _pick_float(row, 'train/dfl_loss')
    vb = _pick_float(row, 'val/box_loss')
    vc = _pick_float(row, 'val/cls_loss')
    vd = _pick_float(row, 'val/dfl_loss')

    train_sum = None
    if tb is not None and tc is not None and td is not None:
        train_sum = tb + tc + td
    val_sum = None
    if vb is not None and vc is not None and vd is not None:
        val_sum = vb + vc + vd

    epoch_val = None
    if row.get('epoch') not in (None, ''):
        try:
            epoch_val = int(float(row['epoch']))
        except (TypeError, ValueError):
            epoch_val = None

    return {
        'epoch': epoch_val,
        'mAP50': map50,
        'precision': prec,
        'recall': rec,
        'train_box_loss': tb,
        'train_cls_loss': tc,
        'train_dfl_loss': td,
        'train_loss_sum': train_sum,
        'val_box_loss': vb,
        'val_cls_loss': vc,
        'val_dfl_loss': vd,
        'val_loss_sum': val_sum,
        'run_name': csv_path.parent.name,
        'results_csv': str(csv_path).replace('\\', '/'),
    }

def load_training_metrics_best_row(project_root: Path, train_project_dir: Path, job: TrainingJob) -> dict | None:
    """读取 results.csv 中 mAP@0.5 最高的一轮指标（与 best.pt 对应）。"""
    _ = project_root
    csv_path = resolve_results_csv_for_job(train_project_dir, job)
    if not csv_path:
        return None
    try:
        with open(csv_path, 'r', encoding='utf-8', newline='') as f:
            rows = list(csv.DictReader(f))
    except OSError:
        return None
    if not rows:
        return None

    best_row = rows[-1]
    best_map = -1.0
    for row in rows:
        map50 = _pick_float(row, 'metrics/mAP50(B)', 'mAP50(B)', 'metrics/mAP_0.5')
        if map50 is not None and map50 > best_map:
            best_map = map50
            best_row = row

    return _row_to_metrics(best_row, csv_path)

_train_lock = threading.Lock()
_proc_lock = threading.Lock()
_active_job_id: int | None = None
_active_proc: subprocess.Popen | None = None
_user_abort_job_ids: set[int] = set()

def _popen_kwargs():
    """本机以 Windows 为主：新建进程组便于 taskkill /T 结束子进程树。"""
    if sys.platform == 'win32':
        return {'creationflags': subprocess.CREATE_NEW_PROCESS_GROUP}
    return {}

def _terminate_pid(pid: int):
    """按 PID 结束进程树（Windows 以 taskkill 为主）。"""
    if not pid or pid <= 0:
        return
    if sys.platform == 'win32':
        subprocess.run(
            ['taskkill', '/F', '/T', '/PID', str(pid)],
            capture_output=True,
            timeout=30,
        )
        return
    else:
        try:
            subprocess.run(['kill', '-TERM', str(pid)], capture_output=True, timeout=5)
        except (FileNotFoundError, OSError):
            pass

def _terminate_process_tree(proc: subprocess.Popen):
    if proc.poll() is not None:
        return
    _terminate_pid(proc.pid)

def _read_pid_from_train_log(log_path: Path) -> int | None:
    """从训练日志头部读取 TRAIN_SUBPROCESS_PID=（与 .pid 文件互为兜底）。"""
    if not log_path.is_file():
        return None
    try:
        head = log_path.read_text(encoding='utf-8', errors='replace')[:16384]
    except OSError:
        return None
    for line in head.splitlines():
        m = re.match(r'^TRAIN_SUBPROCESS_PID=(\d+)\s*$', line.strip())
        if m:
            try:
                return int(m.group(1))
            except ValueError:
                return None
    return None

def _subprocess_no_window_kwargs() -> dict:
    if sys.platform == 'win32' and hasattr(subprocess, 'CREATE_NO_WINDOW'):
        return {'creationflags': subprocess.CREATE_NO_WINDOW}
    return {}

def _find_train_pids_by_run_name(run_name: str) -> list[int]:
    """无 .pid 时按命令行匹配 train.py 与本次 --name（如 best_12，12 为 job.id）。"""
    marker = str(run_name or '').strip()
    if not marker:
        return []
    ps_script = (
        "Get-CimInstance Win32_Process -ErrorAction SilentlyContinue | "
        "Where-Object { $_.CommandLine -and "
        "($_.CommandLine -like '*train.py*') -and "
        f"($_.CommandLine -like '*{marker}*') }} | "
        "Select-Object -ExpandProperty ProcessId"
    )
    try:
        r = subprocess.run(
            ['powershell', '-NoProfile', '-NonInteractive', '-Command', ps_script],
            capture_output=True,
            text=True,
            timeout=45,
            **_subprocess_no_window_kwargs(),
        )
    except (subprocess.TimeoutExpired, OSError, FileNotFoundError):
        return []
    if r.returncode != 0:
        return []
    pids: list[int] = []
    for line in (r.stdout or '').strip().splitlines():
        s = line.strip()
        if not s:
            continue
        try:
            pids.append(int(s))
        except ValueError:
            continue
    return sorted(set(pids))

def _resolve_pid_sidecar_paths(job: TrainingJob, training_root: Path) -> tuple[Path | None, Path | None]:
    """与任务记录中 log_path 同目录的 .pid；log_path 相对 training/ 目录。"""
    raw = (job.log_path or '').strip()
    if not raw:
        return None, None
    log_path = Path(raw)
    if not log_path.is_absolute():
        log_path = (training_root / log_path).resolve()
    else:
        log_path = log_path.resolve()
    pid_path = log_path.with_suffix('.pid')
    return log_path, pid_path

def _kill_train_job_subprocess(training_root: Path, job: TrainingJob) -> int:
    """
    按 .pid / 日志 TRAIN_SUBPROCESS_PID /（Windows）命令行匹配结束训练子进程。
    返回尝试终止的进程数（不含是否已退出）。
    """
    log_path, pid_path = _resolve_pid_sidecar_paths(job, training_root)
    pid: int | None = None
    if pid_path and pid_path.is_file():
        try:
            pid = int(pid_path.read_text(encoding='utf-8').strip())
        except (ValueError, OSError):
            pid = None
    if pid is None and log_path:
        pid = _read_pid_from_train_log(log_path)

    pids_to_kill: list[int] = []
    if pid is not None and pid > 0:
        pids_to_kill = [pid]
    elif sys.platform == 'win32':
        pids_to_kill = _find_train_pids_by_run_name(run_name_for_job(job))

    for p in pids_to_kill:
        _terminate_pid(p)
    if pid_path and pid_path.is_file():
        try:
            pid_path.unlink(missing_ok=True)
        except OSError:
            pass
    return len(pids_to_kill)

def reclaim_stale_training_jobs(app):
    """
    应用启动时：上一进程遗留的 running / queued 任务已无工作线程，结束可能仍存活的子进程并更新状态。
    running：杀子进程 → terminated；queued：无子进程 → failed。
    """
    training_root = get_training_root(app.config)
    with app.app_context():
        rows = TrainingJob.query.filter(TrainingJob.status.in_(['running', 'queued'])).all()
        if not rows:
            return
        now = datetime.utcnow()
        for job in rows:
            if job.status == 'running':
                _kill_train_job_subprocess(training_root, job)
                job.status = 'terminated'
                job.message = '服务重启已中断'
            else:
                job.status = 'failed'
                job.message = '服务重启，任务未执行'
            job.finished_at = now
        db.session.commit()

def _stop_training_by_pid_file(app, job_id: int) -> tuple[bool, str]:
    """
    内存中无 Popen 句柄时：读 pid → taskkill；并将 running 记为 terminated。
    """
    training_root = get_training_root(app.config)
    with app.app_context():
        job = TrainingJob.query.get(job_id)
        if not job:
            return False, '任务不存在'
        if job.status != 'running':
            log_path, pid_path = _resolve_pid_sidecar_paths(job, training_root)
            if pid_path and pid_path.is_file():
                try:
                    pid_path.unlink()
                except OSError:
                    pass
            return False, '当前没有训练中的任务'

        n = _kill_train_job_subprocess(training_root, job)
        if n <= 0:
            return False, '未找到训练进程'

        job = TrainingJob.query.get(job_id)
        if job and job.status == 'running':
            job.status = 'terminated'
            job.message = '已手动停止'
            job.finished_at = datetime.utcnow()
            db.session.commit()
    return True, '已停止'

def stop_training_job(app, job_id: int) -> tuple[bool, str]:
    """
    终止训练子进程：优先本进程内登记的 Popen；否则读 .pid / 日志 / 命令行匹配（适配调试重载）。
    """
    proc = None
    with _proc_lock:
        global _active_job_id, _active_proc
        if _active_job_id == job_id and _active_proc is not None:
            proc = _active_proc
            _user_abort_job_ids.add(job_id)
    if proc is not None:
        _terminate_process_tree(proc)
        return True, '已停止'
    return _stop_training_by_pid_file(app, job_id)

def _arg_path(path: Path, base: Path) -> str:
    """传给 train.py 的路径：在 base 下则用相对路径，否则用绝对路径。"""
    path = path.resolve()
    root = base.resolve()
    try:
        return str(path.relative_to(root)).replace('\\', '/')
    except ValueError:
        return str(path)

def start_training_job_async(app, job_id: int):
    """单任务队列：同一时刻仅一个训练线程。"""
    training_root = get_training_root(app.config)
    upload_root = Path(app.config['UPLOAD_FOLDER']).resolve()

    def worker():
        global _active_job_id, _active_proc
        logs_dir = Path(app.config.get('TRAIN_LOGS_DIR') or 'logs')
        if not logs_dir.is_absolute():
            logs_dir = (training_root / logs_dir).resolve()
        else:
            logs_dir = logs_dir.resolve()
        logs_dir.mkdir(parents=True, exist_ok=True)

        def _cleanup_pid_file(pid_path: Path | None):
            if not pid_path:
                return
            try:
                pid_path.unlink(missing_ok=True)
            except OSError:
                pass

        pid_file: Path | None = None
        try:
            with _train_lock:
                train_py = Path(app.config.get('TRAIN_SCRIPT', training_root / 'train.py'))
                if not train_py.is_absolute():
                    train_py = (training_root / train_py).resolve()
                data_yaml = Path(app.config['TRAIN_DATA_YAML'])
                if not data_yaml.is_absolute():
                    data_yaml = (training_root / data_yaml).resolve()

                with app.app_context():
                    job = TrainingJob.query.get(job_id)
                    if not job:
                        return
                    if not train_py.is_file():
                        job.status = 'failed'
                        job.message = '训练失败'
                        job.finished_at = datetime.utcnow()
                        db.session.commit()
                        return
                    if not data_yaml.is_file():
                        job.status = 'failed'
                        job.message = '训练失败'
                        job.finished_at = datetime.utcnow()
                        db.session.commit()
                        return

                    try:
                        weights = resolve_training_weights_file(
                            upload_root,
                            job.base_weights_rel,
                            training_root,
                        )
                    except ValueError:
                        job.status = 'failed'
                        job.message = '训练失败'
                        job.finished_at = datetime.utcnow()
                        db.session.commit()
                        return

                    batch = job.batch_size if job.batch_size is not None else int(app.config.get('TRAIN_BATCH', 8))
                    imgsz = job.imgsz if job.imgsz is not None else int(app.config.get('TRAIN_DEFAULT_IMGSZ', 640))
                    patience = job.patience if job.patience is not None else int(app.config.get('TRAIN_DEFAULT_PATIENCE', 25))
                    device = (job.device or app.config.get('TRAIN_DEVICE') or 'cpu').strip()

                    run_name = training_run_name(job.base_weights_rel or '', job.id)
                    log_file = logs_dir / f'{run_name}.log'
                    pid_file = logs_dir / f'{run_name}.pid'

                    job.status = 'running'
                    job.started_at = datetime.utcnow()
                    try:
                        job.log_path = str(log_file.relative_to(training_root)).replace('\\', '/')
                    except ValueError:
                        job.log_path = str(log_file).replace('\\', '/')
                    db.session.commit()

                    project_dir = Path(app.config['TRAIN_PROJECT_DIR'])
                    if not project_dir.is_absolute():
                        project_dir = (training_root / project_dir).resolve()

                    cmd = [
                        sys.executable,
                        str(train_py.resolve()),
                        '--data', _arg_path(data_yaml, training_root),
                        '--weights', _arg_path(weights, training_root),
                        '--epochs', str(job.epochs),
                        '--project', _arg_path(project_dir, training_root),
                        '--name', run_name,
                        '--batch', str(max(1, min(int(batch), 128))),
                        '--imgsz', str(max(320, min(int(imgsz), 1280))),
                        '--device', device[:32],
                        '--patience', str(max(1, min(int(patience), 200))),
                    ]

                    proc = None
                    rc = None
                    user_aborted = False
                    try:
                        with open(log_file, 'w', encoding='utf-8') as logf:
                            logf.write(f'cwd={training_root}\n')
                            logf.write(f'weights={weights}\n')
                            logf.write(f'cmd={" ".join(cmd)}\n\n')
                            logf.flush()
                            proc = subprocess.Popen(
                                cmd,
                                cwd=str(training_root),
                                stdout=logf,
                                stderr=subprocess.STDOUT,
                                **_popen_kwargs(),
                            )
                            logf.write(f'TRAIN_SUBPROCESS_PID={proc.pid}\n')
                            logf.flush()
                            try:
                                pid_file.parent.mkdir(parents=True, exist_ok=True)
                                pid_file.write_text(str(proc.pid), encoding='utf-8')
                            except OSError:
                                pass
                            with _proc_lock:
                                _active_job_id = job_id
                                _active_proc = proc
                            try:
                                timeout = int(app.config.get('TRAIN_TIMEOUT_SEC', 86400))
                                t0 = time.time()
                                while proc.poll() is None:
                                    if time.time() - t0 > timeout:
                                        _terminate_process_tree(proc)
                                        with _proc_lock:
                                            _active_job_id = None
                                            _active_proc = None
                                        _user_abort_job_ids.discard(job_id)
                                        job = TrainingJob.query.get(job_id)
                                        if job:
                                            job.status = 'failed'
                                            job.message = '训练超时'
                                            job.finished_at = datetime.utcnow()
                                            db.session.commit()
                                        return
                                    time.sleep(1.0)
                                rc = proc.returncode
                            finally:
                                with _proc_lock:
                                    _active_job_id = None
                                    _active_proc = None
                                user_aborted = job_id in _user_abort_job_ids
                                _user_abort_job_ids.discard(job_id)
                    except Exception:
                        with _proc_lock:
                            _active_job_id = None
                            _active_proc = None
                        _user_abort_job_ids.discard(job_id)
                        job = TrainingJob.query.get(job_id)
                        if job:
                            job.status = 'failed'
                            job.message = '训练异常'
                            job.finished_at = datetime.utcnow()
                            db.session.commit()
                        return

                    job = TrainingJob.query.get(job_id)
                    if not job:
                        return
                    db.session.refresh(job)
                    if job.status == 'terminated':
                        return
                    best_pt = project_dir / run_name / 'weights' / 'best.pt'

                    if user_aborted:
                        job.status = 'terminated'
                        job.message = '已手动停止'
                        job.finished_at = datetime.utcnow()
                        db.session.commit()
                        return

                    if rc is None or rc != 0:
                        job.status = 'failed'
                        job.message = '训练失败'
                        job.finished_at = datetime.utcnow()
                        db.session.commit()
                        return

                    if not best_pt.is_file():
                        job.status = 'failed'
                        job.message = '训练失败'
                        job.finished_at = datetime.utcnow()
                        db.session.commit()
                        return

                    job.status = 'succeeded'
                    job.output_weights_path = str(best_pt).replace('\\', '/')
                    job.message = '训练完成'
                    job.finished_at = datetime.utcnow()
                    db.session.commit()

        finally:
            _cleanup_pid_file(pid_file)

    threading.Thread(target=worker, daemon=True).start()
