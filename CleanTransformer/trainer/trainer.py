# -*- coding: utf-8 -*-
# @Time    : 2024/9/8 15:22
# @Author  : Patrick
# @Email   : firechecking@gmail.com
# @File    : trainer.py
# @Software: CleanTransformer
# @Description: trainer

import sys, math, inspect
from packaging import version
from contextlib import contextmanager
from typing import Mapping

import torch
import numpy as np
from torch.utils.data import (
    Dataset,
    DataLoader,
    RandomSampler,
    SequentialSampler,
)

import datasets
from accelerate import Accelerator
from transformers import (
    get_scheduler,
    PreTrainedModel,
    TrainingArguments,
    PreTrainedTokenizerBase,
    SequenceFeatureExtractor,
)
from accelerate.utils import (
    DistributedType,
    DataLoaderConfiguration,
    GradientAccumulationPlugin,
    DistributedDataParallelKwargs,
)
from transformers.utils import (
    find_labels,
    can_return_loss,
    is_datasets_available,
    is_accelerate_available,
)
from transformers.integrations import (
    deepspeed_init,
)
from transformers.pytorch_utils import (
    ALL_LAYERNORM_LAYERS,
)
from transformers.trainer_utils import (
    has_length,
    seed_worker,
    TrainOutput,
    EvalPrediction,
    EvalLoopOutput,
    number_of_arguments,
    RemoveColumnsCollator,
    denumpify_detensorize,
)
from transformers.training_args import (
    ParallelMode,
)
from transformers.modeling_utils import (
    unwrap_model,
)
from transformers.trainer_pt_utils import (
    nested_detach,
    find_batch_size,
    EvalLoopContainer,
    get_parameter_names,
    LengthGroupedSampler,
    IterableDatasetShard,
)
from transformers.data.data_collator import (
    default_data_collator,
    DataCollatorWithPadding,
)


@contextmanager
def CodeBlock(name):
    yield


class Trainer():
    def __init__(
            self,
            model=None,
            args: TrainingArguments = None,
            data_collator=None,
            train_dataset=None,
            eval_dataset=None,
            tokenizer=None,
            model_init=None,
            compute_metrics=None,
            optimizers=(None, None),
            preprocess_logits_for_metrics=None,
    ):
        self.args = args

        with CodeBlock("初始化随机种子"):
            pass  # todo: 初始化随机种子

        with CodeBlock("实例化accelerator"):
            self.create_accelerator_and_postprocess()
            args._setup_devices  # 设置device、n_gpu

        with CodeBlock("model、tokenizer预处理"):
            if model_init is not None:
                self.model_init = model_init
                model = self.call_model_init()

            ############### 判断模型是否被分布加载到多个gpu ###############
            self.is_model_parallel = False
            if hasattr(model, "is_parallelizable") and model.is_parallelizable and model.model_parallel:
                self.is_model_parallel = True
            if getattr(model, "hf_device_map", None) is not None:
                devices = [device for device in set(model.hf_device_map.values()) if device not in ["cpu", "disk"]]
                self.is_model_parallel = False
                if len(devices) > 1:
                    self.is_model_parallel = True
                elif len(devices) == 1:
                    self.is_model_parallel = self.args.device != torch.device(devices[0])

            ############### 将模型移动到device ###############
            self.place_model_on_device = args.place_model_on_device
            if self.is_model_parallel or self.is_deepspeed_enabled or self.is_fsdp_enabled:
                self.place_model_on_device = False
            if self.place_model_on_device:
                self._move_model_to_device(model, args.device)

            self.model = model
            self.tokenizer = tokenizer

        with CodeBlock("dataset、collator预处理"):
            self.train_dataset = train_dataset
            self.eval_dataset = eval_dataset
            default_collator = (
                DataCollatorWithPadding(tokenizer)
                if tokenizer is not None and isinstance(tokenizer, (PreTrainedTokenizerBase, SequenceFeatureExtractor))
                else default_data_collator
            )
            self.data_collator = data_collator if data_collator is not None else default_collator

        with CodeBlock('optimzier, scheduler处理'):
            self.optimizer, self.lr_scheduler = optimizers

        with CodeBlock("state、control、callbacks初始化"):
            pass  # todo: state, control, callbacks

        with CodeBlock("其他参数初始化"):
            ############### 判断model.forward()的输入参数 ###############
            self._signature_columns = None
            default_label_names = find_labels(self.model.__class__)
            self.label_names = default_label_names if self.args.label_names is None else self.args.label_names

            ############### 评测相关的参数 ###############
            self.compute_metrics = compute_metrics
            self.preprocess_logits_for_metrics = preprocess_logits_for_metrics

            self.is_in_train = False

        with CodeBlock("参数异常校验"):
            if (self.is_deepspeed_enabled or self.is_fsdp_enabled) and (self.optimizer is not None or self.lr_scheduler is not None):
                raise RuntimeError('使用deepspeed或fsdp时要求动态创建optimizer、scheduler，可以继承Trainer并修改create_optimizer、create_scheduler方法')
            if model_init is not None and (self.optimizer is not None or self.lr_scheduler is not None):
                raise RuntimeError('使用model_init()时要求动态创建optimizer、scheduler，可以继承Trainer并修改create_optimizer、create_scheduler方法')

    def __训练流程__(self):
        pass

    def train(self, ignore_keys_for_eval=None, resume_from_checkpoint=None, **kwargs):
        self.is_in_train = True

        with CodeBlock("训练循环"):
            train_output = self._inner_training_loop(
                batch_size=self.args.train_batch_size,
                args=self.args,
                ignore_keys_for_eval=ignore_keys_for_eval,
                resume_from_checkpoint=kwargs.pop('model_path', resume_from_checkpoint),
            )

        self.is_in_train = False

        return train_output

    def _inner_training_loop(self, batch_size=None, args=None,
                             ignore_keys_for_eval=None, resume_from_checkpoint=None):
        self.accelerator.free_memory()  # 清除变量引用和显存cache

        with CodeBlock("初始化dataloader"):
            self._train_batch_size = batch_size
            train_dataloader = self.get_train_dataloader()

        with CodeBlock("确定训练steps、epochs"):
            assert has_length(train_dataloader) or args.max_steps > 0, "train_dataloader长度未知情况下，必须设置max_steps，以正确终止训练"
            assert args.num_train_epochs > 0 or args.max_steps > 0, "max_steps或num_train_epochs必须至少设置一项，以正确终止训练"

            total_train_batch_size = self._train_batch_size * args.gradient_accumulation_steps * args.world_size

            if has_length(train_dataloader):
                ############### 数据集长度已知 ###############
                num_examples = self.num_examples(train_dataloader)
                num_update_steps_per_epoch = max(len(train_dataloader) // args.gradient_accumulation_steps, 1)
                if args.max_steps > 0:
                    ############### 用户设定了max_steps ###############
                    max_steps = args.max_steps
                    num_train_epochs = max_steps // num_update_steps_per_epoch + int(max_steps % num_update_steps_per_epoch > 0)
                    num_train_samples = args.max_steps * total_train_batch_size
                else:
                    ############### 用户设定num_train_epochs ###############
                    max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
                    num_train_epochs = math.ceil(args.num_train_epochs)
                    num_train_samples = self.num_examples(train_dataloader) * args.num_train_epochs
            else:
                ############### 数据集长度未知 ###############
                num_train_epochs = sys.maxsize
                num_update_steps_per_epoch = max_steps = args.max_steps
                num_examples = num_train_samples = total_train_batch_size * args.max_steps

        with CodeBlock("初始化optimizer、scheduler"):
            ############### deepspeed: 初始化DummyOptim或调用create_optimzier() ###############
            if self.is_deepspeed_enabled:
                # 注意！deepspeed_init与trainer的耦合性很强
                # 会调用trainer.model, trainer.args
                # 会调用trainer.create_optimizer(), trainer.create_scheduler(), 并设置trainer.optimizer, trainer.scheduler
                self.optimizer, self.lr_scheduler = deepspeed_init(self, num_training_steps=max_steps)

            ############### fsdp: 需要在model包装之后创建optimizer、scheduler ###############
            delay_optimizer_creation = self.is_fsdp_enabled
            if not delay_optimizer_creation:
                self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        with CodeBlock("使用accelerator包装model、optimizer、scheduler"):
            self.model_wrapped = self.model
            ############### DP、DDP处理 ###############
            # 在_wrap_model内，如果n_gpu>1，用torch.nn.DataParallel (后续也不需要使用accelerator)
            # 否则只进行accelerate的DistributedDataParallelKwargs配置，由后面accelerator.prepare()完成模型包装
            model = self._wrap_model(self.model_wrapped)
            use_accelerator_prepare = True if model is self.model else False

            if delay_optimizer_creation:
                ############### 先处理model，再创建optimizer、scheduler ###############
                if use_accelerator_prepare:
                    self.model = self.accelerator.prepare_model(self.model)
                self.create_optimizer_and_scheduler(num_training_steps=max_steps)

            if use_accelerator_prepare:
                ############### 对model、optimizer、scheduler应用accelerator.prepare ###############
                self.model.train()
                if hasattr(self.lr_scheduler, "step"):
                    model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)
                else:
                    model, self.optimizer, self.lr_scheduler = self.accelerator.prepare(self.model, self.optimizer, self.lr_scheduler)

            ############### 处理model、self.model、self.model_wrapped ###############
            if self.is_fsdp_enabled:
                self.model = self.model_wrapped = model
            if model is not self.model:
                self.model_wrapped = model
            if self.is_deepspeed_enabled:
                self.deepspeed = self.model_wrapped

        with CodeBlock('训练循环'):
            with CodeBlock('变量初始化'):
                tr_loss = torch.tensor(0.0).to(args.device)
                self._total_loss_scalar = 0.0
                self.current_flos = 0
                epochs_trained = 0
                global_step = 0
                total_batched_samples = 0

            ############### epoch循环 ###############
            for epoch in range(epochs_trained, num_train_epochs):
                epoch_iterator = train_dataloader
                if hasattr(epoch_iterator, "set_epoch"):
                    epoch_iterator.set_epoch(epoch)

                steps_in_epoch = (
                    len(epoch_iterator)
                    if has_length(epoch_iterator)
                    else args.max_steps * args.gradient_accumulation_steps
                )

                ############### step循环 ###############
                for step, inputs in enumerate(epoch_iterator):
                    ############### 单次forward、backward ###############
                    total_batched_samples += 1
                    model.zero_grad()
                    with self.accelerator.accumulate(model):
                        tr_loss_step = self.training_step(model, inputs)
                    tr_loss += tr_loss_step
                    # global_step += 1

                    ############### 模型参数更新 ###############
                    is_last_step_and_steps_less_than_grad_acc = (
                            steps_in_epoch <= args.gradient_accumulation_steps and (step + 1) == steps_in_epoch
                    )
                    if (
                            total_batched_samples % args.gradient_accumulation_steps == 0
                            or
                            # last step in epoch but step is always smaller than gradient_accumulation_steps
                            is_last_step_and_steps_less_than_grad_acc
                    ):
                        if is_last_step_and_steps_less_than_grad_acc:
                            self.accelerator.gradient_state._set_sync_gradients(True)

                        with CodeBlock("梯度裁剪"):
                            if args.max_grad_norm is not None and args.max_grad_norm > 0:  # 梯度裁剪
                                if is_accelerate_available() and self.accelerator.distributed_type == DistributedType.DEEPSPEED:
                                    grad_norm = model.get_global_grad_norm()
                                    if hasattr(grad_norm, "item"):
                                        grad_norm = grad_norm.item()
                                else:
                                    grad_norm = self.accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                        with CodeBlock("参数更新"):
                            self.optimizer.step()
                            if not self.accelerator.optimizer_step_was_skipped:
                                if not isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                                    self.lr_scheduler.step()

                        global_step += 1

        train_loss = tr_loss.item() / max(global_step, 0.001)
        return TrainOutput(global_step, train_loss, None)

    def training_step(self, model, inputs):
        model.train()

        ############### forward并计算loss ###############
        inputs = self._prepare_inputs(inputs)  # 预处理inputs，如处理tensor的device、dtype等
        loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()

        ############### backward得到梯度 ###############
        # loss.backward()
        self.accelerator.backward(loss)
        return loss.detach()

    def compute_loss(self, model, inputs, return_outputs=False):
        # 这部分主要是针对hf-transformers的模型，如果是自定义模型，可以覆盖compute_loss()函数
        outputs = model(**inputs)
        if isinstance(outputs, dict) and "loss" not in outputs:
            raise ValueError('模型未返回loss')
        # 默认outputs[0]为loss
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        return (loss, outputs) if return_outputs else loss

    def __评测流程__(self):
        pass

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        ############### 评测数据准备 ###############
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        if isinstance(eval_dataset, dict):
            metrics = {}
            for eval_dataset_name, _eval_dataset in eval_dataset.items():
                dataset_metrics = self.evaluate(
                    eval_dataset=_eval_dataset,
                    ignore_keys=ignore_keys,
                    metric_key_prefix=f"{metric_key_prefix}_{eval_dataset_name}",
                )
                metrics.update(dataset_metrics)
            return metrics

        eval_dataloader = self.get_eval_dataloader(eval_dataset)

        ############### 调用评测循环 ###############
        output = self.evaluation_loop(
            eval_dataloader,
            description="Evaluation",
            prediction_loss_only=True if self.compute_metrics is None else None,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

        return output.metrics

    def evaluation_loop(self, dataloader, description, prediction_loss_only=None, ignore_keys=None, metric_key_prefix='eval'):
        args = self.args
        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else args.prediction_loss_only

        with CodeBlock("处理model"):
            # 处理逻辑和train()差不多，如果是在训练过程中进入evaluate()，model不会被重复处理
            if self.is_deepspeed_enabled and self.deepspeed is None:
                _, _ = deepspeed_init(self, num_training_steps=0, inference=True)

            model = self._wrap_model(self.model, training=False)
            if len(self.accelerator._models) == 0 and model is self.model:
                model = (
                    self.accelerator.prepare(model)
                    if self.is_deepspeed_enabled
                    else self.accelerator.prepare_model(model, evaluation_mode=True)
                )
                if self.is_fsdp_enabled:
                    self.model = model
                if model is not self.model:
                    self.model_wrapped = model
                if self.is_deepspeed_enabled:
                    self.deepspeed = self.model_wrapped

            if not self.is_in_train:
                if args.fp16_full_eval:
                    model = model.to(dtype=torch.float16, device=args.device)
                elif args.bf16_full_eval:
                    model = model.to(dtype=torch.bfloat16, device=args.device)
            model.eval()

        with CodeBlock("预测循环"):
            all_losses = EvalLoopContainer(self.args.eval_do_concat_batches, padding_index=-100)
            all_preds = EvalLoopContainer(self.args.eval_do_concat_batches, padding_index=-100)
            all_labels = EvalLoopContainer(self.args.eval_do_concat_batches, padding_index=-100)
            all_inputs = EvalLoopContainer(self.args.eval_do_concat_batches, padding_index=-100)
            batch_size = self.args.eval_batch_size
            observed_num_examples = 0

            for step, inputs in enumerate(dataloader):
                ############### 单次预测 ###############
                loss, logits, labels = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)

                ############### 记录样本数量 ###############
                observed_batch_size = find_batch_size(inputs)
                if observed_batch_size is not None:
                    observed_num_examples += observed_batch_size

                with CodeBlock("记录单次预测结果"):
                    main_input_name = getattr(self.model, "main_input_name", "input_ids")

                    inputs_decode = self._prepare_input(inputs[main_input_name]) if args.include_inputs_for_metrics else None
                    if inputs_decode is not None:
                        inputs_decode = self.accelerator.pad_across_processes(inputs_decode, dim=1, pad_index=-100)
                        inputs_decode = self.gather_function((inputs_decode))
                        all_inputs.add(inputs_decode)

                    if labels is not None:
                        labels = self.accelerator.pad_across_processes(labels, dim=1, pad_index=-100)
                        labels = self.gather_function((labels))
                        all_labels.add(labels)

                    if loss is not None:
                        losses = self.gather_function((loss.repeat(batch_size)))
                        all_losses.add(losses)

                    if logits is not None:
                        logits = self.accelerator.pad_across_processes(logits, dim=1, pad_index=-100)
                        if self.preprocess_logits_for_metrics is not None:
                            logits = self.preprocess_logits_for_metrics(logits, labels)
                        logits = self.gather_function((logits))
                        all_preds.add(logits)

        with CodeBlock("计算metrics"):
            ############### 计算metrics（依赖传入的compute_metrics函数） ###############
            all_losses = all_losses.get_arrays()
            all_preds = all_preds.get_arrays()
            all_labels = all_labels.get_arrays()
            all_inputs = all_inputs.get_arrays()
            if self.compute_metrics is not None and all_preds is not None and all_labels is not None:
                if args.include_inputs_for_metrics:
                    metrics = self.compute_metrics(
                        EvalPrediction(predictions=all_preds, label_ids=all_labels, inputs=all_inputs)
                    )
                else:
                    metrics = self.compute_metrics(EvalPrediction(predictions=all_preds, label_ids=all_labels))
            else:
                metrics = {}
            metrics = denumpify_detensorize(metrics)

            if isinstance(all_losses, list) and all_losses:
                metrics[f"{metric_key_prefix}_loss"] = np.concatenate(all_losses).mean().item()
            elif isinstance(all_losses, np.ndarray):
                metrics[f"{metric_key_prefix}_loss"] = all_losses.mean().item()
            if hasattr(self, "jit_compilation_time"):
                metrics[f"{metric_key_prefix}_jit_compilation_time"] = self.jit_compilation_time

            for key in list(metrics.keys()):
                if not key.startswith(f"{metric_key_prefix}_"):
                    metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        with CodeBlock("计算样本总数"):
            eval_dataset = getattr(dataloader, "dataset", None)
            if has_length(eval_dataset):
                num_samples = len(eval_dataset)
            elif isinstance(eval_dataset, IterableDatasetShard) and getattr(eval_dataset, "num_examples", 0) > 0:
                num_samples = eval_dataset.num_examples
            else:
                if has_length(dataloader):
                    num_samples = self.num_examples(dataloader)
                else:
                    num_samples = observed_num_examples
            if num_samples == 0 and observed_num_examples > 0:
                num_samples = observed_num_examples

        return EvalLoopOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics, num_samples=num_samples)

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        ############### 判断模型是否返回loss ###############
        return_loss = inputs.get("return_loss", None)
        if return_loss is None:
            return_loss = can_return_loss(self.model.__class__)

        has_labels = False if len(self.label_names) == 0 else all(inputs.get(k) is not None for k in self.label_names)
        loss_without_labels = True if len(self.label_names) == 0 and return_loss else False
        labels = None
        if has_labels or loss_without_labels:
            labels = nested_detach(tuple(inputs.get(name) for name in self.label_names))
            if len(labels) == 1:
                labels = labels[0]

        ############### 调用模型 ###############
        with torch.no_grad():
            if has_labels or loss_without_labels:
                loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
                loss = loss.mean().detach()
                if isinstance(outputs, dict):
                    logits = tuple(v for k, v in outputs.items() if k not in ignore_keys + ["loss"])
                else:
                    logits = outputs[1:]
            else:
                loss = None
                outputs = model(**inputs)
                if isinstance(outputs, dict):
                    logits = tuple(v for k, v in outputs.items() if k not in ignore_keys)
                else:
                    logits = outputs

        ############### 返回结果 ###############
        if prediction_loss_only:
            return (loss, None, None)

        logits = nested_detach(logits)
        if len(logits) == 1:
            logits = logits[0]

        return (loss, logits, labels)

    def __训练模型处理__(self):
        pass

    def call_model_init(self, trial=None):
        model_init_argcount = number_of_arguments(self.model_init)
        if model_init_argcount == 0:
            model = self.model_init()
        elif model_init_argcount == 1:
            model = self.model_init(trial)
        else:
            raise RuntimeError("model_init should have 0 or 1 argument.")

        if model is None:
            raise RuntimeError("model_init should not return None.")

        return model

    def _move_model_to_device(self, model, device):
        model = model.to(device)
        ############### 调用模型model.tie_weights()，让output_embeddings和input_embeddings共享权重 ###############
        if hasattr(model, "tie_weights"):
            model.tie_weights()

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        optimizer = self.create_optimizer()
        self.create_scheduler(num_training_steps=num_training_steps, optimizer=optimizer)

    def create_optimizer(self):
        ############### trainer默认的optimizer初始化，有定制化需求可以覆盖本函数 ###############
        opt_model = self.model
        if self.optimizer is not None: return self.optimizer

        decay_parameters = self.get_decay_parameter_names(opt_model)
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)
                ],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [
                    p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)
                ],
                "weight_decay": 0.0,
            },
        ]

        ############### 各种optimizer和支持的参数类型太多，偷懒直接复用hf_trainer代码实例化optimizer ################
        from transformers.trainer import Trainer as HF_Trainer
        optimizer_cls, optimizer_kwargs = HF_Trainer.get_optimizer_cls_and_kwargs(self.args, opt_model)
        if "params" in optimizer_kwargs:
            optimizer_grouped_parameters = optimizer_kwargs.pop("params")
        if "optimizer_dict" in optimizer_kwargs:
            optimizer_grouped_parameters = optimizer_kwargs.pop("optimizer_dict")
        self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

        return self.optimizer

    def get_decay_parameter_names(self, model):
        ############### nn.LayerNorm和Bias不参与权重衰减 ###############
        decay_parameters = get_parameter_names(model, ALL_LAYERNORM_LAYERS)
        decay_parameters = [name for name in decay_parameters if "bias" not in name]
        return decay_parameters

    def create_scheduler(self, num_training_steps, optimizer=None):
        ############### trainer默认的scheduler初始化，有定制化需求可以覆盖本函数 ###############
        if self.lr_scheduler is None:
            self.lr_scheduler = get_scheduler(
                self.args.lr_scheduler_type,
                optimizer=self.optimizer if optimizer is None else optimizer,
                num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
                num_training_steps=num_training_steps,
                scheduler_specific_kwargs=self.args.lr_scheduler_kwargs,
            )
            self._created_lr_scheduler = True
        return self.lr_scheduler

    def __数据处理__(self):
        pass

    def get_train_dataloader(self):
        train_dataset = self.train_dataset
        data_collator = self.data_collator

        with CodeBlock("去除model.forward()不支持的参数"):
            if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
                # 直接从Dataset中移除不支持的数据列
                train_dataset = self._remove_unused_columns(train_dataset, description="training")
            else:
                # 对collator包装一层处理代码，在collator前移除不支持的数据列
                data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

        with CodeBlock("创建dataloader"):
            dataloader_params = {
                "batch_size": self._train_batch_size,
                "collate_fn": data_collator,
                "num_workers": self.args.dataloader_num_workers,
                "pin_memory": self.args.dataloader_pin_memory,
                "persistent_workers": self.args.dataloader_persistent_workers,
            }

            if not isinstance(self.train_dataset, torch.utils.data.IterableDataset):
                dataloader_params["sampler"] = self._get_train_sampler()
                dataloader_params["drop_last"] = self.args.dataloader_drop_last
                dataloader_params["worker_init_fn"] = seed_worker
                dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

            train_dataloader = DataLoader(train_dataset, **dataloader_params)
        return self.accelerator.prepare(train_dataloader)

    def _get_train_sampler(self):
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        if not self.args.group_by_length:
            return RandomSampler(self.train_dataset)
        else:
            ############### 获取每个样本长度 (作为分组依据) ###############
            if is_datasets_available() and isinstance(self.train_dataset, datasets.Dataset):
                lengths = (
                    self.train_dataset[self.args.length_column_name]
                    if self.args.length_column_name in self.train_dataset.column_names
                    else None
                )
            else:
                lengths = None

            ############### LengthGroupedSampler: 先将dataset随机分成多个megabatches组，然后每个megabatch组内按长度排序 ###############
            model_input_name = self.tokenizer.model_input_names[0] if self.tokenizer is not None else None
            return LengthGroupedSampler(
                self.args.train_batch_size * self.args.gradient_accumulation_steps,
                dataset=self.train_dataset,
                lengths=lengths,
                model_input_name=model_input_name,
            )

    def _remove_unused_columns(self, dataset, description=None):
        if not self.args.remove_unused_columns:
            return dataset

        ############### 获取model.forward()接受的输入参数 ###############
        signature_columns = self._set_signature_columns_if_needed()

        ############### 获取待删除/带保留的参数 ###############
        ignored_columns = list(set(dataset.column_names) - set(signature_columns))
        columns = [k for k in signature_columns if k in dataset.column_names]

        ############### 移除不支持的参数 ###############
        if version.parse(datasets.__version__) < version.parse("1.4.0"):
            dataset.set_format(
                type=dataset.format["type"], columns=columns, format_kwargs=dataset.format["format_kwargs"]
            )
            return dataset
        else:
            return dataset.remove_columns(ignored_columns)

    def _set_signature_columns_if_needed(self):
        '''获取model.forward()输入参数'''
        if self._signature_columns is None:
            model_to_inspect = self.model
            signature = inspect.signature(model_to_inspect.forward)
            self._signature_columns = list(signature.parameters.keys())
            # 防止误删除label标签
            self._signature_columns += list(set(["label", "label_ids"] + self.label_names))
        return self._signature_columns

    def _get_collator_with_removed_columns(self, data_collator, description=None):
        if not self.args.remove_unused_columns:
            return data_collator
        ############### 获取model.forward()接受的输入参数 ###############
        signature_columns = self._set_signature_columns_if_needed()

        ############### RemoveColumnsCollator: 调用原始colaltor之前，去除输入batch中不支持的参数 ###############
        remove_columns_collator = RemoveColumnsCollator(
            data_collator=data_collator,
            signature_columns=signature_columns,
            logger=None,
            description=description,
            model_name=self.model.__class__.__name__,
        )
        return remove_columns_collator

    def num_examples(self, dataloader: DataLoader) -> int:
        try:
            dataset = dataloader.dataset
            if isinstance(dataset, IterableDatasetShard):
                return len(dataset.dataset)
            return len(dataset)
        except (NameError, AttributeError, TypeError):
            return len(dataloader) * self.args.per_device_train_batch_size

    def _prepare_inputs(self, inputs):
        return self._prepare_input(inputs)

    def _prepare_input(self, data):
        ############### 处理data的device、dtype ###############
        if isinstance(data, Mapping):
            return type(data)({k: self._prepare_input(v) for k, v in data.items()})
        elif isinstance(data, (tuple, list)):
            return type(data)(self._prepare_input(v) for v in data)
        elif isinstance(data, torch.Tensor):
            kwargs = {"device": self.args.device}
            if self.is_deepspeed_enabled and (torch.is_floating_point(data) or torch.is_complex(data)):
                kwargs.update({"dtype": self.accelerator.state.deepspeed_plugin.hf_ds_config.dtype()})
            return data.to(**kwargs)
        return data

    def get_eval_dataloader(self, eval_dataset):
        ############### eval_dataloader复用 ###############
        if hasattr(self, "_eval_dataloader") and self.args.dataloader_persistent_workers:
            return self.accelerator.prepare(self._eval_dataloader)

        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        data_collator = self.data_collator

        ############### 移除model.forward()不支持的参数 ###############
        if is_datasets_available() and isinstance(eval_dataset, datasets.Dataset):
            eval_dataset = self._remove_unused_columns(eval_dataset, description="evaluation")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="evaluation")

        ############### 构造dataloader ###############
        dataloader_params = {
            "batch_size": self.args.eval_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(eval_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_eval_sampler(eval_dataset)
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        eval_dataloader = DataLoader(eval_dataset, **dataloader_params)
        if self.args.dataloader_persistent_workers:
            self._eval_dataloader = eval_dataloader

        return self.accelerator.prepare(eval_dataloader)

    def _get_eval_sampler(self, eval_dataset):
        return SequentialSampler(eval_dataset)

    def __并行_分布式__(self):
        pass

    def create_accelerator_and_postprocess(self):
        args = {}

        with CodeBlock("GradientAccumulationPlugin"):
            # GradientAccumulationPlugin包含4个参数：num_steps、adjust_scheduler、sync_with_dataloader、sync_each_batch
            grad_acc_kwargs = {}
            if is_accelerate_available("0.28.0") and self.args.accelerator_config.gradient_accumulation_kwargs is not None:
                grad_acc_kwargs = self.args.accelerator_config.gradient_accumulation_kwargs

            ############### 梯度累积步数 ###############
            if "num_steps" not in grad_acc_kwargs:
                grad_acc_kwargs["num_steps"] = self.args.gradient_accumulation_steps
            elif self.args.gradient_accumulation_steps > 1:
                raise ValueError("`gradient_accumulation_steps` is set in `AcceleratorConfig`, "
                                 "do not set the `TrainingArguments` `gradient_accumulation_steps`.")

            ############### sync_with_dataloader: 当dataloader的最后一个batch时，同步一次梯度 ###############
            grad_acc_kwargs["sync_with_dataloader"] = False

            ############### 实例化GradientAccumulationPlugin ###############
            gradient_accumulation_plugin = GradientAccumulationPlugin(**grad_acc_kwargs)
            args['gradient_accumulation_plugin'] = gradient_accumulation_plugin

        with CodeBlock("DataLoaderConfiguration"):
            # DataLoaderConfiguration包含4个参数：split_batches、dispatch_batches、even_batches、use_seedable_sampler
            accelerator_config = self.args.accelerator_config.to_dict()
            accelerator_config.pop("gradient_accumulation_kwargs")
            if is_accelerate_available("0.28.0"):
                dataloader_config = DataLoaderConfiguration(
                    split_batches=accelerator_config.pop("split_batches"),
                    dispatch_batches=accelerator_config.pop("dispatch_batches"),
                    even_batches=accelerator_config.pop("even_batches"),
                    use_seedable_sampler=accelerator_config.pop("use_seedable_sampler"),
                )
                args["dataloader_config"] = dataloader_config
            else:
                args.update(accelerator_config)

        with CodeBlock("DeepSpeedPlugin"):
            args['deepspeed_plugin'] = self.args.deepspeed_plugin

        with CodeBlock("FullyShardedDataParallelPlugin"):
            # 在training_args中设置环境变量ACCELERATE_USE_FSDP，以及环境变量FSDP_AUTO_WRAP_POLICY、FSDP_OFFLOAD_PARAMS等参数
            # 在Accelerator.__init__()中，会根据这些环境变量来实例化FullyShardedDataParallelPlugin
            # 所以不需要手动实例化FullyShardedDataParallelPlugin
            pass

        with CodeBlock("实例化accelerator"):
            self.accelerator = Accelerator(**args)
            self.gather_function = self.accelerator.gather_for_metrics

        with CodeBlock("设置DeepSpeed参数"):
            self.is_deepspeed_enabled = getattr(self.accelerator.state, "deepspeed_plugin", None) is not None
            if self.is_deepspeed_enabled and getattr(self.args, "hf_deepspeed_config", None) is None:
                # 如果deepspeed不是通过ds_config文件初始化，需要调用HfTrainerDeepSpeedConfig.trainer_config_process()根据training_args对deepspeed参数进行设置
                self.propagate_args_to_deepspeed()

        with CodeBlock("设置FSDP参数"):
            self.is_fsdp_enabled = getattr(self.accelerator.state, "fsdp_plugin", None) is not None
            if self.is_fsdp_enabled:
                fsdp_plugin = self.accelerator.state.fsdp_plugin
                fsdp_plugin.limit_all_gathers = self.args.fsdp_config.get("limit_all_gathers", fsdp_plugin.limit_all_gathers)
                if is_accelerate_available("0.23.0"):
                    fsdp_plugin.activation_checkpointing = self.args.fsdp_config.get("activation_checkpointing",
                                                                                     fsdp_plugin.activation_checkpointing)
                    if fsdp_plugin.activation_checkpointing and self.args.gradient_checkpointing:
                        raise ValueError(
                            "Please use FSDP's activation_checkpointing when using FSDP."
                        )

        with CodeBlock("合法性校验"):
            ds_or_fsdp = self.is_deepspeed_enabled or self.is_fsdp_enabled
            if ds_or_fsdp and self.args.save_only_model and self.args.load_best_model_at_end:
                raise ValueError(f"DeepSpeed or FSDP can't be used with `save_only_model` along with `load_best_model_at_end`.")
            if ds_or_fsdp and self.args.auto_find_batch_size:
                raise ValueError(f"DeepSpeed or FSDP doesn't support `auto_find_batch_size`.")

    def propagate_args_to_deepspeed(self, auto_find_batch_size=False):
        # 利用HfTrainerDeepSpeedConfig中的相关代码，将self.args中相关参数同步到deepspeed_plugin中
        from transformers.integrations.deepspeed import HfTrainerDeepSpeedConfig

        ds_plugin = self.accelerator.state.deepspeed_plugin

        ds_plugin.hf_ds_config = HfTrainerDeepSpeedConfig(ds_plugin.hf_ds_config.config)
        # 在DeepSpeedPlugin中，deepspeed_config和hf_ds_config.config指向同一个对象
        # 但是因为HfTrainerDeepSpeedConfig.__init__()中写了config = deepcopy(config_file_or_dict)，所以config指针断开了
        # 所以这里需要对deepspeed_config重新赋值
        ds_plugin.deepspeed_config = ds_plugin.hf_ds_config.config

        # 将training_args中的参数设置到deepspeed_config中
        ds_plugin.hf_ds_config.trainer_config_process(self.args, auto_find_batch_size)

    def _wrap_model(self, model, training=True):
        if unwrap_model(model) is not model:  # 防止多次wrap
            return model

        if self.args.n_gpu > 1:  # n_gpu表示一个进程管理多个gpu，此时使用DataParallel模式
            model = torch.nn.DataParallel(model)

        if not training: return model

        if self.args.parallel_mode == ParallelMode.DISTRIBUTED:
            # ParallelMode
            # 1. DISTRIBUTED: 对应distributed_type为MULTI_XXX、DEEPSPEED、FSDP、MEGATRON_LM等
            # 2. NOT_DISTRIBUTED: 对应DP
            # 3. NOT_PARALLEL: 单机单卡
            ############### 设置accelerate相关的DDP参数###############
            kwargs = {}
            if self.args.ddp_find_unused_parameters is not None:
                kwargs["find_unused_parameters"] = self.args.ddp_find_unused_parameters
            elif isinstance(model, PreTrainedModel):
                kwargs["find_unused_parameters"] = not model.is_gradient_checkpointing
            else:
                kwargs["find_unused_parameters"] = True

            if self.args.ddp_bucket_cap_mb is not None:
                kwargs["bucket_cap_mb"] = self.args.ddp_bucket_cap_mb

            if self.args.ddp_broadcast_buffers is not None:
                kwargs["broadcast_buffers"] = self.args.ddp_broadcast_buffers

            self.accelerator.ddp_handler = DistributedDataParallelKwargs(*kwargs)

        return model


if __name__ == "__main__":
    pass
