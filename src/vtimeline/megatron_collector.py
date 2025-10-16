#############################################
### Megatron-Core Collector for vtimeline ###
#############################################

import os
import json
import torch
import duckdb
import hashlib
import warnings
import sys
from datetime import datetime


def _log(msg: str, level: str = "INFO"):
    """统一的日志输出函数，带时间戳和级别"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    print(f"[{timestamp}][MegatronCollector][{level}] {msg}", flush=True)


def _get_cksum(data: torch.Tensor):
    byte_data = data.detach().cpu().view(torch.uint8).contiguous().numpy().tobytes()
    hasher = hashlib.sha256()
    hasher.update(byte_data)

    return hasher.hexdigest()


class MegatronCollector:
    # ===== 步骤计数器 =====
    step_ = 0
    dump_step_: int = int(os.getenv("VTIMELINE_DUMP_STEP", -1))
    dumped_training_batch_once_: bool = False
    
    # ===== 核心组件（默认为 None，等待 set_core 设置）=====
    model_ = None
    optimizer_ = None
    scheduler_ = None
    
    # ===== 进程组信息（默认为空字典，等待 set_process_group_info 设置）=====
    ranks_info_ = {}
    
    # ===== 数据库连接（默认为 None，等待 initialize 设置）=====
    db_ = None
    
    # ===== 初始化状态标记 =====
    _ranks_info_set_ = False
    _core_set_ = False
    _db_initialized_ = False
    
    # ===== 详细日志开关（通过环境变量控制）=====
    _verbose_logging_ = os.getenv("MEGATRON_COLLECTOR_VERBOSE", "1") == "1"

    def __init__(self):
        raise RuntimeError("Use initialize to init Megatron Core Collector")

    @classmethod
    def set_process_group_info(cls, ranks_info):
        """阶段1初始化：设置进程组信息"""
        cls.ranks_info_ = ranks_info
        cls._ranks_info_set_ = True
        
        if cls._verbose_logging_:
            _log(f"✓ Rank info set: dp={ranks_info.get('dp')}, tp={ranks_info.get('tp')}, "
                 f"pp={ranks_info.get('pp')}, cp={ranks_info.get('cp')}, "
                 f"ep={ranks_info.get('ep')}, etp={ranks_info.get('etp')}")
        else:
            _log(f"Rank info set: {ranks_info}")

    @classmethod
    def set_core(cls, model, optimizer, scheduler):
        """阶段2初始化：设置核心组件并初始化数据库"""
        _log(f"Setting core components... (rank dp={cls.ranks_info_.get('dp', '?')})")
        
        cls.model_ = model
        cls.optimizer_ = optimizer
        cls.scheduler_ = scheduler

        if not isinstance(cls.model_, list):
            cls.model_ = [cls.model_]

        cls._core_set_ = True
        
        if cls._verbose_logging_:
            model_count = len(cls.model_)
            _log(f"✓ Core components set: {model_count} model(s), "
                 f"optimizer={'set' if optimizer else 'None'}, "
                 f"scheduler={'set' if scheduler else 'None'}")
        
        # 自动调用数据库初始化
        cls.initialize()

    @classmethod
    def initialize(cls):
        """阶段3初始化：创建数据库连接"""
        # 防止重复初始化
        if cls._db_initialized_:
            if cls._verbose_logging_:
                _log("Database already initialized, skipping", "DEBUG")
            return
        
        _log("Initializing database connection...")
        
        # 检查前置条件
        if not cls._ranks_info_set_:
            error_msg = ("Cannot initialize: ranks_info not set. "
                        "Please call set_process_group_info() first.")
            _log(error_msg, "ERROR")
            raise RuntimeError(f"[MegatronCollector] {error_msg}")
        
        if not cls._core_set_:
            warning_msg = ("Initializing database before set_core() is called. "
                          "This may work but is not recommended.")
            _log(warning_msg, "WARNING")
            warnings.warn(f"[MegatronCollector] {warning_msg}")
        
        root_dir = os.environ.get("VTIMELINE_LOGGER_DIR", "/var/log")
        db_dir = os.path.join(root_dir, "Collector")
        os.makedirs(db_dir, exist_ok=True)

        db_path = os.path.join(
            root_dir,
            "Collector/coredump_dp{}_tp{}_pp{}_cp{}.db".format(
                cls.ranks_info_.get("dp", 0),
                cls.ranks_info_.get("tp", 0),
                cls.ranks_info_.get("pp", 0),
                cls.ranks_info_.get("cp", 0)
            ),
        )
        
        try:
            cls.db_ = duckdb.connect(db_path)
            cls.db_.execute(
                """CREATE TABLE IF NOT EXISTS coredump(
                      step INTEGER,
                      stage TEXT,
                      data JSON);"""
            )
            cls._db_initialized_ = True
            _log(f"✓ Database initialized at: {db_path}")
            
            if cls._verbose_logging_:
                _log(f"Dump will be enabled for steps 0-{cls.dump_step_} "
                     f"(VTIMELINE_DUMP_STEP={os.getenv('VTIMELINE_DUMP_STEP', '-1')})")
        except Exception as e:
            _log(f"Failed to initialize database: {e}", "ERROR")
            raise

    @classmethod
    def should_dump(cls):
        return cls.step_ <= cls.dump_step_

    @classmethod
    def _ensure_ready_for_dump(cls, operation_name: str):
        """确保 dump 操作的所有前置条件都满足，如果不满足则抛出详细错误"""
        errors = []
        
        if not cls._ranks_info_set_:
            errors.append("ranks_info not set (need to call set_process_group_info first)")
        
        if not cls._core_set_:
            errors.append("core components not set (need to call set_core first)")
        
        if not cls._db_initialized_:
            errors.append("database not initialized (initialize should be called by set_core)")
        
        if cls.model_ is None:
            errors.append("model is None")
        
        if cls.db_ is None:
            errors.append("database connection is None")
        
        if errors:
            error_msg = f"[MegatronCollector.{operation_name}] Cannot proceed: " + ", ".join(errors)
            _log(error_msg, "ERROR")
            _log(f"Detailed status:", "ERROR")
            _log(f"  Stage: {operation_name}, Step: {cls.step_}", "ERROR")
            _log(f"  _ranks_info_set_: {cls._ranks_info_set_}", "ERROR")
            _log(f"  _core_set_: {cls._core_set_}", "ERROR")
            _log(f"  _db_initialized_: {cls._db_initialized_}", "ERROR")
            _log(f"  model_ is None: {cls.model_ is None}", "ERROR")
            _log(f"  db_ is None: {cls.db_ is None}", "ERROR")
            _log(f"  ranks_info_: {cls.ranks_info_}", "ERROR")
            
            # 打印调用堆栈
            import traceback
            _log("Call stack:", "ERROR")
            for line in traceback.format_stack()[:-1]:
                _log(line.strip(), "ERROR")
            
            raise RuntimeError(error_msg)

    @classmethod
    def dump_main_grad(
        cls, param, param_name: str, stage_name: str = "main-grad-in-bwd"
    ):
        if not cls.should_dump():
            return
        
        # 确保已完全初始化
        cls._ensure_ready_for_dump("dump_main_grad")
        
        if cls._verbose_logging_:
            _log(f"Dumping main_grad: {param_name} @ {stage_name}")

        param_info = {
            "name": param_name,
            "cksum": _get_cksum(param.main_grad),
            "shape": list(param.main_grad.shape),
            "type": str(param.main_grad.type()),
        }
        param_info.update(cls.ranks_info_)

        try:
            cls.db_.execute(
                "INSERT INTO coredump VALUES (?, ?, ?);",
                (cls.step_, stage_name, json.dumps(param_info)),
            )
        except Exception as e:
            _log(f"Error inserting main_grad data into coredump: {e}", "ERROR")
            raise

    @classmethod
    def dump_main_param(cls, stage_name: str):
        if not cls.should_dump():
            return
        
        # 确保已完全初始化
        cls._ensure_ready_for_dump("dump_main_param")
        
        if cls._verbose_logging_:
            _log(f"Dumping main_param @ {stage_name}")
        
        param_count = 0
        for model in cls.model_:
            for name, param in model.named_parameters():
                main_param_exist = (
                    hasattr(param, "main_param") and param.main_param is not None
                )

                param_info = {
                    "name": name,
                    "cksum": _get_cksum(param.main_param) if main_param_exist else None,
                    "shape": list(param.main_param.shape) if main_param_exist else None,
                    "type": str(param.main_param.type())
                    if hasattr(param, "main_param") and param.main_param is not None
                    else None,
                }
                param_info.update(cls.ranks_info_)
                try:
                    cls.db_.execute(
                        "INSERT INTO coredump VALUES (?, ?, ?);",
                        (cls.step_, stage_name, json.dumps(param_info)),
                    )
                    param_count += 1
                except Exception as e:
                    _log(f"Error inserting main_param data into coredump: {e}", "ERROR")
                    raise
        
        if cls._verbose_logging_:
            _log(f"✓ Dumped {param_count} main_params @ {stage_name}")

    @classmethod
    def dump_model(cls, stage_name: str):
        if not cls.should_dump():
            return
        
        # 确保已完全初始化，如果未初始化会抛出详细错误
        try:
            cls._ensure_ready_for_dump("dump_model")
        except RuntimeError as e:
            # 添加额外的上下文信息
            _log("=" * 80, "ERROR")
            _log(f"CRITICAL: dump_model failed at stage '{stage_name}', step {cls.step_}", "ERROR")
            _log("This indicates MegatronCollector was not properly initialized before dump was called.", "ERROR")
            _log("=" * 80, "ERROR")
            raise  # 重新抛出异常，确保问题被发现
        
        if cls._verbose_logging_:
            _log(f"Dumping model parameters @ {stage_name}")
        
        param_count = 0
        grad_count = 0
        for model in cls.model_:
            for name, param in model.named_parameters():
                param_info = {
                    "name": name,
                    "cksum": _get_cksum(param),
                    "shape": list(param.shape),
                    "type": str(param.type()),
                    "requires_grad": param.requires_grad,
                    "grad_cksum": _get_cksum(param.grad)
                    if param.grad is not None
                    else None,
                    "grad_shape": list(param.grad.shape)
                    if param.grad is not None
                    else None,
                    "grad_type": str(param.grad.type())
                    if param.grad is not None
                    else None,
                }
                param_info.update(cls.ranks_info_)
                try:
                    cls.db_.execute(
                        "INSERT INTO coredump VALUES (?, ?, ?);",
                        (cls.step_, stage_name, json.dumps(param_info)),
                    )
                    param_count += 1
                    if param.grad is not None:
                        grad_count += 1
                except Exception as e:
                    _log(f"Error inserting model data into coredump: {e}", "ERROR")
                    raise
        
        if cls._verbose_logging_:
            _log(f"✓ Dumped {param_count} parameters ({grad_count} with grads) @ {stage_name}")


    @classmethod
    def dump_training_batch(cls, tokens, labels, loss_mask, attention_mask, position_ids, stage_name: str = "after-get-batch"):
        """Dump key training batch info after get_batch.

        Make the JSON data format consistent with other dumps: one row per tensor with fields
        name, cksum, shape, type, plus rank info.
        """
        # 若已在本进程 dump 过一次训练 batch，则不再重复 dump
        if cls.dumped_training_batch_once_:
            if cls._verbose_logging_:
                _log(f"Training batch already dumped once, skipping @ {stage_name}", "DEBUG")
            return
        
        if not cls.should_dump():
            return
        
        # 确保已完全初始化
        cls._ensure_ready_for_dump("dump_training_batch")
        
        if cls._verbose_logging_:
            _log(f"Dumping training batch @ {stage_name}")

        items = [
            ("tokens", tokens),
            ("labels", labels),
            ("loss_mask", loss_mask),
            ("attention_mask", attention_mask),
            ("position_ids", position_ids),
        ]

        item_count = 0
        for name, t in items:
            try:
                info = {
                    "name": name,
                    "cksum": _get_cksum(t) if t is not None else None,
                    "shape": list(t.shape) if t is not None else None,
                    "type": str(t.type()) if t is not None else None,
                }
                info.update(cls.ranks_info_)
                cls.db_.execute(
                    "INSERT INTO coredump VALUES (?, ?, ?);",
                    (cls.step_, stage_name, json.dumps(info)),
                )
                item_count += 1
            except Exception as e:
                _log(f"Error inserting batch data into coredump: {e}", "ERROR")
                raise
        
        # 标记本进程已完成一次训练 batch dump
        cls.dumped_training_batch_once_ = True
        
        if cls._verbose_logging_:
            _log(f"✓ Dumped {item_count} training batch items @ {stage_name}")

    @classmethod
    def step(cls):
        """增加步骤计数器"""
        cls.step_ += 1
        if cls._verbose_logging_:
            _log(f"Step incremented to {cls.step_}")
