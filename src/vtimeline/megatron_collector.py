#############################################
### Megatron-Core Collector for vtimeline ###
#############################################

import os
import json
import torch
import duckdb
import hashlib


def _get_cksum(data: torch.Tensor):
    byte_data = data.detach().cpu().view(torch.uint8).contiguous().numpy().tobytes()
    hasher = hashlib.sha256()
    hasher.update(byte_data)

    return hasher.hexdigest()


class MegatronCollector:
    step_ = 0
    dump_step_: int = int(os.getenv("VTIMELINE_DUMP_STEP", -1))
    # 每个进程仅 dump 一个 batch 的一次性开关
    dumped_training_batch_once_: bool = False

    def __init__(self):
        raise RuntimeError("Use initialize to init Megatron Core Collector")

    @classmethod
    def initialize(cls):
        root_dir = os.environ.get("VTIMELINE_LOGGER_DIR", "/var/log")
        db_dir = os.path.join(root_dir, "Collector")
        os.makedirs(db_dir, exist_ok=True)

        assert hasattr(cls, "ranks_info_"), "the rank information must be set"

        db_path = os.path.join(
            root_dir,
            "Collector/coredump_dp{}_tp{}_pp{}_cp{}.db".format(
                cls.ranks_info_["dp"], cls.ranks_info_["tp"],cls.ranks_info_["pp"],cls.ranks_info_["cp"]
            ),
        )
        cls.db_ = duckdb.connect(db_path)

        cls.db_.execute(
            """CREATE TABLE IF NOT EXISTS coredump(
                  step INTEGER,
                  stage TEXT,
                  data JSON);"""
        )

    @classmethod
    def set_process_group_info(cls, ranks_info):
        cls.ranks_info_ = ranks_info

    @classmethod
    def set_core(cls, model, optimizer, scheduler):
        cls.model_ = model
        cls.optimizer_ = optimizer
        cls.scheduler_ = scheduler

        if not isinstance(cls.model_, list):
            cls.model_ = [cls.model_]

        cls.initialize()

    @classmethod
    def should_dump(cls):
        return cls.step_ <= cls.dump_step_

    @classmethod
    def dump_main_grad(
        cls, param, param_name: str, stage_name: str = "main-grad-in-bwd"
    ):
        if not cls.should_dump():
            return

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
            print(f"Error inserting data into coredump: {e}")

    @classmethod
    def dump_main_param(cls, stage_name: str):
        if not cls.should_dump():
            return

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
                except Exception as e:
                    print(f"Error inserting data into coredump: {e}")

    @classmethod
    def dump_model(cls, stage_name: str):
        if not cls.should_dump():
            return

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
                except Exception as e:
                    print(f"Error inserting data into coredump: {e}")


    @classmethod
    def dump_optimizer_state(cls, stage_name: str):
        """Dump optimizer state (momentum, variance, etc.) to database.
        
        This method dumps the internal state of the optimizer (e.g., exp_avg, exp_avg_sq
        for Adam/AdamW, momentum_buffer for SGD) for each parameter.
        
        Args:
            stage_name: Stage identifier (e.g., "optimizer-state-after-step")
        """
        if not cls.should_dump():
            return
        
        if not hasattr(cls, 'optimizer_') or cls.optimizer_ is None:
            return
        
        # Access the underlying PyTorch optimizer
        # MegatronOptimizer wraps a PyTorch optimizer in self.optimizer
        pytorch_optimizer = cls.optimizer_.optimizer if hasattr(cls.optimizer_, 'optimizer') else cls.optimizer_
        
        # Build a mapping from param to name for faster lookup
        param_to_name = {}
        if hasattr(cls, 'model_'):
            for model in cls.model_:
                for name, param in model.named_parameters():
                    # Handle both regular params and main_params
                    param_to_name[id(param)] = name
                    if hasattr(param, 'main_param') and param.main_param is not None:
                        param_to_name[id(param.main_param)] = name
        
        # Iterate through optimizer state
        for group in pytorch_optimizer.param_groups:
            for param in group['params']:
                if param not in pytorch_optimizer.state:
                    continue
                
                state = pytorch_optimizer.state[param]
                
                # Get parameter name
                param_name = param_to_name.get(id(param), "unknown")
                
                # Dump each state field (exp_avg, exp_avg_sq, momentum_buffer, etc.)
                for state_key, state_value in state.items():
                    if isinstance(state_value, torch.Tensor):
                        state_info = {
                            "name": param_name,
                            "state_key": state_key,  # e.g., "exp_avg", "exp_avg_sq"
                            "cksum": _get_cksum(state_value),
                            "shape": list(state_value.shape),
                            "type": str(state_value.type()),
                        }
                        state_info.update(cls.ranks_info_)
                        
                        try:
                            cls.db_.execute(
                                "INSERT INTO coredump VALUES (?, ?, ?);",
                                (cls.step_, stage_name, json.dumps(state_info)),
                            )
                        except Exception as e:
                            print(f"Error inserting optimizer state into coredump: {e}")

    @classmethod
    def dump_training_batch(cls, tokens, labels, loss_mask, attention_mask, position_ids, stage_name: str = "after-get-batch"):
        """Dump key training batch info after get_batch.

        Make the JSON data format consistent with other dumps: one row per tensor with fields
        name, cksum, shape, type, plus rank info.
        """
        # 若已在本进程 dump 过一次训练 batch，则不再重复 dump
        if cls.dumped_training_batch_once_:
            return
        if not cls.should_dump():
            return

        items = [
            ("tokens", tokens),
            ("labels", labels),
            ("loss_mask", loss_mask),
            ("attention_mask", attention_mask),
            ("position_ids", position_ids),
        ]

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
            except Exception as e:
                print(f"Error inserting batch data into coredump: {e}")
        # 标记本进程已完成一次训练 batch dump
        cls.dumped_training_batch_once_ = True

    @classmethod
    def step(cls):
        cls.step_ += 1
