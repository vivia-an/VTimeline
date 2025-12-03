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
                # 基础信息
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
                
                # 添加分片参数标识 (用于DP分片参数互异性检查)
                # Megatron-LM 中 param.allreduce=False 表示专家并行参数（分片参数）
                # param.allreduce=True 或不存在表示全复制参数
                is_expert_parallel = not getattr(param, 'allreduce', True)
                param_info["param_sharded"] = is_expert_parallel
                param_info["param_full_replica"] = not is_expert_parallel
                
                # 添加分布统计量：min, max, quantile (用于极值/分位数一致性检查)
                try:
                    if param.requires_grad and param.dtype in (torch.float32, torch.float16, torch.bfloat16):
                        flat_param = param.detach().float().flatten()
                        param_info["min"] = float(flat_param.min().item())
                        param_info["max"] = float(flat_param.max().item())
                        # 计算分位数 (25%, 50%, 75%)
                        param_info["quantile_25"] = float(torch.quantile(flat_param, 0.25).item())
                        param_info["quantile_50"] = float(torch.quantile(flat_param, 0.50).item())
                        param_info["quantile_75"] = float(torch.quantile(flat_param, 0.75).item())
                        
                        # 添加高阶统计量：skewness (偏度) 和 kurtosis (峰度)
                        # 用于 model-before-optimizer-step阶段DP参数分布高阶统计量一致性检查
                        n = flat_param.numel()
                        if n > 2:  # 需要至少3个元素才能计算偏度和峰度
                            mean = flat_param.mean()
                            std = flat_param.std()
                            if std > 1e-10:  # 避免除零
                                # 偏度 (skewness): E[(X-μ)³] / σ³
                                skewness = ((flat_param - mean) ** 3).mean() / (std ** 3)
                                param_info["skewness"] = float(skewness.item())
                                
                                # 峰度 (kurtosis): E[(X-μ)⁴] / σ⁴ - 3 (Fisher's definition)
                                kurtosis = ((flat_param - mean) ** 4).mean() / (std ** 4) - 3.0
                                param_info["kurtosis"] = float(kurtosis.item())
                        
                        # 添加分布形态统计量：histogram 和 entropy
                        # 用于 DP参数分布形态一致性检查
                        if n > 10:  # 需要足够的元素才能计算有意义的直方图
                            # 计算直方图 (使用 64 个 bin，归一化后计算 checksum)
                            hist_bins = 64
                            hist_min = flat_param.min()
                            hist_max = flat_param.max()
                            if hist_max > hist_min:
                                hist = torch.histc(flat_param, bins=hist_bins, min=float(hist_min), max=float(hist_max))
                                hist_normalized = hist / hist.sum()  # 归一化为概率分布
                                
                                # histogram_cksum: 直方图的 checksum（用于检测分布形态差异）
                                hist_bytes = hist_normalized.cpu().numpy().tobytes()
                                param_info["histogram_cksum"] = hash(hist_bytes) & 0xFFFFFFFF  # 32-bit hash
                                
                                # 计算熵 (entropy): -sum(p * log(p))
                                # 过滤掉零概率以避免 log(0)
                                hist_nonzero = hist_normalized[hist_normalized > 1e-10]
                                if len(hist_nonzero) > 0:
                                    entropy = -torch.sum(hist_nonzero * torch.log(hist_nonzero))
                                    param_info["entropy"] = float(entropy.item())
                except Exception:
                    pass  # 统计量计算失败不影响主流程
                
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
        for group_idx, group in enumerate(pytorch_optimizer.param_groups):
            # 获取该 param_group 的 lr（用于 optimizer_state_dict.lr 一致性检查）
            group_lr = group.get('lr', None)
            
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
                            "lr": group_lr,  # 添加 lr 字段用于一致性检查
                            "param_group_idx": group_idx,  # 添加 group 索引
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
