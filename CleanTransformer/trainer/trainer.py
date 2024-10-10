# -*- coding: utf-8 -*-
# @Time    : 2024/9/8 15:22
# @Author  : Patrick
# @Email   : firechecking@gmail.com
# @File    : trainer.py
# @Software: CleanTransformer
# @Description: trainer

import os, re, sys, math, shutil, random, inspect, warnings
from pathlib import Path
from typing import Mapping
from packaging import version
from contextlib import contextmanager

import torch
import numpy as np
from torch.utils.data import (
    Dataset,
    DataLoader,
    RandomSampler,
    SequentialSampler,
)

import datasets, transformers
import safetensors.torch
from transformers import (
    get_scheduler,
    PreTrainedModel,
    PretrainedConfig,
    TrainingArguments,
    PreTrainedTokenizerBase,
    SequenceFeatureExtractor,
)
from accelerate.utils import (
    load_fsdp_model,
    DistributedType,
    save_fsdp_model,
    save_fsdp_optimizer,
    load_fsdp_optimizer,
    DataLoaderConfiguration,
    DeepSpeedSchedulerWrapper,
    GradientAccumulationPlugin,
    DistributedDataParallelKwargs,
)
from transformers.utils import (
    logging,
    find_labels,
    can_return_loss,
    is_datasets_available,
    is_accelerate_available,
)
from transformers.trainer import (
    CONFIG_NAME,
    WEIGHTS_NAME,
    SCHEDULER_NAME,
    OPTIMIZER_NAME,
    FSDP_MODEL_NAME,
    SAFE_WEIGHTS_NAME,
    TRAINING_ARGS_NAME,
    TRAINER_STATE_NAME,
    WEIGHTS_INDEX_NAME,
    OPTIMIZER_NAME_BIN,
    SAFE_WEIGHTS_INDEX_NAME,
)
from transformers.integrations import (
    deepspeed_init,
    deepspeed_load_checkpoint,
    get_reporting_integration_callbacks,
)
from transformers.pytorch_utils import (
    ALL_LAYERNORM_LAYERS,
    is_torch_greater_or_equal_than_1_13,
)
from transformers.trainer_utils import (
    has_length,
    seed_worker,
    TrainOutput,
    EvalPrediction,
    EvalLoopOutput,
    number_of_arguments,
    get_last_checkpoint,
    RemoveColumnsCollator,
    denumpify_detensorize,
    PREFIX_CHECKPOINT_DIR,
)
from transformers.training_args import (
    ParallelMode,
)
from transformers.modeling_utils import (
    unwrap_model,
    load_sharded_checkpoint,
)
from transformers.trainer_pt_utils import (
    nested_detach,
    find_batch_size,
    EvalLoopContainer,
    distributed_concat,
    reissue_pt_warnings,
    get_parameter_names,
    LengthGroupedSampler,
    IterableDatasetShard,
    get_dataloader_sampler,
    distributed_broadcast_scalars,
)
from transformers.trainer_callback import (
    TrainerState,
    TrainerControl,
    PrinterCallback,
    CallbackHandler,
    ProgressCallback,
    DefaultFlowCallback,
)
from transformers.data.data_collator import (
    default_data_collator,
    DataCollatorWithPadding,
)
from accelerate import Accelerator, skip_first_batches
from accelerate import __version__ as accelerate_version

if is_accelerate_available() and version.parse(accelerate_version) > version.parse('0.23.0'):
    from accelerate.data_loader import SeedableRandomSampler
logger = logging.get_logger(__name__)


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
            callbacks=None,
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

        with CodeBlock("state、control、callbacks初始化"):
            ############### TrainerState, TrainerControl初始化 ###############
            self.state = TrainerState(
                is_local_process_zero=self.is_local_process_zero(),
                is_world_process_zero=self.is_world_process_zero(),
            )
            self.control = TrainerControl()

            ############### callbacks初始化 ###############
            # DefaultFlowCallback:
            # 1. 根据设置的logging_strategy, evaluation_strategy, save_strategy等触发should_log, should_evaluate, should_save
            # 2. 根据训练steps触发control.should_training_stop
            default_callbacks = [DefaultFlowCallback] + get_reporting_integration_callbacks(self.args.report_to)
            callbacks = default_callbacks if callbacks is None else default_callbacks + callbacks
            self.callback_handler = CallbackHandler(callbacks, self.model, self.tokenizer, self.optimizer, self.lr_scheduler)
            self.callback_handler.add_callback(PrinterCallback if self.args.disable_tqdm else ProgressCallback)

            ############### 触发on_init_end回调 ###############
            self.control = self.callback_handler.on_init_end(self.args, self.state, self.control)

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

        with CodeBlock("加载ckpt"):
            if isinstance(resume_from_checkpoint, bool) and resume_from_checkpoint:
                resume_from_checkpoint = get_last_checkpoint(args.output_dir)
                if resume_from_checkpoint is None:
                    raise ValueError(f"No valid checkpoint found in output directory ({args.output_dir})")
            if resume_from_checkpoint is not None:
                ############### 加载模型参数 ###############
                if self.is_deepspeed_enabled or self.is_fsdp_enabled:
                    self._load_from_checkpoint(resume_from_checkpoint, model=self.model_wrapped)
                else:
                    self._load_from_checkpoint(resume_from_checkpoint)

                ############### 加载optimizer、scheduler参数 ###############
                self._load_optimizer_and_scheduler(resume_from_checkpoint)

        with CodeBlock("恢复训练进度"):
            if resume_from_checkpoint is not None and os.path.isfile(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)):
                self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
                assert self.state.train_batch_size == self._train_batch_size # 确保batch_size没有改变
                # TODO: compare_trainer_and_checkpoint_args

                ############### 恢复epoch ###############
                epochs_trained = self.state.global_step // num_update_steps_per_epoch

                ############### 恢复step ###############
                if args.ignore_data_skip:
                    steps_trained_in_current_epoch = 0
                else:
                    steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
                    steps_trained_in_current_epoch *= args.gradient_accumulation_steps

                logger.info("  Continuing training from checkpoint, will skip to saved global_step")
                logger.info(f"  Continuing training from epoch {epochs_trained}")
                logger.info(f"  Continuing training from global step {self.state.global_step}")
                if not args.ignore_data_skip:
                    ############### 模拟前epochs_trained次数据采样，以恢复sampler的rng_state ###############
                    logger.info(f"  Will skip the first {epochs_trained} epochs")
                    for epoch in range(epochs_trained):
                        sampler = get_dataloader_sampler(train_dataloader)
                        sampler_kinds = [RandomSampler]
                        if version.parse(accelerate_version) > version.parse("0.23.0"):
                            sampler_kinds.append(SeedableRandomSampler)
                        is_random_sampler = isinstance(sampler, tuple(sampler_kinds))
                        if not is_random_sampler:
                            for _ in train_dataloader:  # TODO: 没明白非随机sampler为什么需要模拟数据采样
                                break
                        else:
                            sampler = sampler if sampler is not None else []
                            _ = list(sampler)

        with CodeBlock('更新state, control, callbacks'):
            def maybe_abs_or_ratio(abs_or_ratio, base_value, default):
                if abs_or_ratio is None: return default
                return base_value * abs_or_ratio if abs_or_ratio < 1 else abs_or_ratio

            self.state.logging_steps = maybe_abs_or_ratio(args.logging_steps, max_steps, default=self.state.logging_steps)
            self.state.eval_steps = maybe_abs_or_ratio(args.eval_steps, max_steps, default=self.state.eval_steps)
            self.state.save_steps = maybe_abs_or_ratio(args.save_steps, max_steps, default=self.state.save_steps)
            self.state.epoch = 0
            self.state.max_steps = max_steps
            self.state.num_train_epochs = num_train_epochs
            self.state.train_batch_size = self._train_batch_size
            self.state.is_local_process_zero = self.is_local_process_zero()
            self.state.is_world_process_zero = self.is_world_process_zero()

            self.callback_handler.model = self.model
            self.callback_handler.optimizer = self.optimizer
            self.callback_handler.lr_scheduler = self.lr_scheduler
            self.callback_handler.train_dataloader = train_dataloader

            # 训练开始
            self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        with CodeBlock('训练循环'):
            with CodeBlock('变量初始化'):
                tr_loss = torch.tensor(0.0).to(args.device)
                self._total_loss_scalar = 0.0
                self.current_flos = 0
                # epochs_trained = 0
                total_batched_samples = 0
                self._globalstep_last_logged = self.state.global_step

            ############### epoch循环 ###############
            for epoch in range(epochs_trained, num_train_epochs):
                # 开始epoch训练
                self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

                epoch_iterator = train_dataloader
                if hasattr(epoch_iterator, "set_epoch"):
                    epoch_iterator.set_epoch(epoch)

                steps_in_epoch = (
                    len(epoch_iterator)
                    if has_length(epoch_iterator)
                    else args.max_steps * args.gradient_accumulation_steps
                )
                ############### 恢复训练进度: 跳过steps_trained_in_current_epoch ###############
                steps_skipped = 0
                if epoch == epochs_trained and resume_from_checkpoint is not None:
                    if steps_trained_in_current_epoch > 0:
                        epoch_iterator = skip_first_batches(epoch_iterator, steps_trained_in_current_epoch)
                        steps_skipped = steps_trained_in_current_epoch

                    self._load_rng_state(resume_from_checkpoint)

                ############### step循环 ###############
                for step, inputs in enumerate(epoch_iterator):
                    ############### 单次forward、backward ###############
                    total_batched_samples += 1

                    ############### 记录tokens数 ###############
                    if self.args.include_num_input_tokens_seen:
                        main_input_name = getattr(self.model, "main_input_name", "input_ids")
                        if main_input_name in inputs:
                            input_device = inputs[main_input_name].device
                            self.state.num_input_tokens_seen += torch.sum(
                                torch.tensor(inputs[main_input_name].numel(), device=input_device, dtype=torch.int64)
                            ).item()
                    if step % args.gradient_accumulation_steps == 0:
                        # 开始step训练
                        self.control = self.callback_handler.on_step_begin(args, self.state, self.control)

                    model.zero_grad()
                    with self.accelerator.accumulate(model):
                        tr_loss_step = self.training_step(model, inputs)
                    tr_loss += tr_loss_step
                    self.current_flos += float(self.floating_point_ops(inputs))

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

                        ############### state, control, callbacks ###############
                        self.state.global_step += 1
                        self.state.epoch = epoch + (step + 1 + steps_skipped) / steps_in_epoch
                        # 1个step训练结束
                        self.control = self.callback_handler.on_step_end(args, self.state, self.control)

                        ############### evaluate ###############
                        self._maybe_log_save_evaluate(tr_loss, grad_norm, model, epoch, ignore_keys_for_eval)
                    else:
                        # 梯度累计的1个substep训练结束
                        self.control = self.callback_handler.on_substep_end(args, self.state, self.control)

                    # 跳出step循环
                    if self.control.should_epoch_stop or self.control.should_training_stop:
                        break

                ############### state, control, callbacks ###############
                if step < 0:
                    self.control.should_training_stop = True
                # 1个epoch训练结束
                self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)

                ############### evaluate ###############
                self._maybe_log_save_evaluate(tr_loss, grad_norm, model, epoch, ignore_keys_for_eval)

                # 跳出epoch循环
                if self.control.should_training_stop:
                    break

        self.control = self.callback_handler.on_train_end(args, self.state, self.control)
        train_loss = tr_loss.item() / max(self.state.global_step, 0.001)

        if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            self._load_best_model()

        return TrainOutput(self.state.global_step, train_loss, None)

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

        # 完成evaluate
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, output.metrics)
        self.log(output.metrics)
        return output.metrics

    def evaluation_loop(self, dataloader, description, prediction_loss_only=None, ignore_keys=None, metric_key_prefix='eval'):
        args = self.args
        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else args.prediction_loss_only
        self.callback_handler.eval_dataloader = dataloader

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

                # 单步预测结束
                self.control = self.callback_handler.on_prediction_step(args, self.state, self.control)

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

    def _nested_gather(self, tensors):
        if tensors is None:
            return
        if (self.args.distributed_state is not None and self.args.distributed_state.distributed_type != "NO") or (
                self.args.distributed_state is None and self.args.local_rank != -1
        ):
            tensors = distributed_concat(tensors)
        return tensors

    def __状态_日志__(self):
        pass

    def _maybe_log_save_evaluate(self, tr_loss, grad_norm, model, epoch, ignore_keys_for_eval):
        ############### LOG ###############
        if self.control.should_log and self.state.global_step > self._globalstep_last_logged:
            logs = {}

            ############### 记录loss、lr、grad_norm ###############
            # all_gather + mean() to get average loss over all processes
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()
            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            logs["learning_rate"] = self._get_learning_rate()
            if grad_norm is not None:
                logs["grad_norm"] = grad_norm.detach().item() if isinstance(grad_norm, torch.Tensor) else grad_norm

            ############### 更新记录值 ###############
            tr_loss -= tr_loss
            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            ############### 触发log相关Callbacks ###############
            self.log(logs)

        ############### EVALUATE ###############
        metrics = None
        if self.control.should_evaluate:
            metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                metric_to_check = self.args.metric_for_best_model
                if not metric_to_check.startswith("eval_"):
                    metric_to_check = f"eval_{metric_to_check}"
                self.lr_scheduler.step(metrics[metric_to_check])

        ############### SAVE ###############
        if self.control.should_save:
            self._save_checkpoint(model, metrics=metrics)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)

    def floating_point_ops(self, inputs):
        # 如果是PreTrainedModel, 可以调用model.floating_point_ops()获取forward+backward的浮点计算次数
        # 如果不是PreTrainedModel，需要重写该方法
        if hasattr(self.model, "floating_point_ops"):
            return self.model.floating_point_ops(inputs)
        else:
            return 0

    def store_flos(self):
        if self.args.parallel_mode == ParallelMode.DISTRIBUTED:
            self.state.total_flos += (
                distributed_broadcast_scalars([self.current_flos], device=self.args.device).sum().item()
            )
            self.current_flos = 0
        else:
            self.state.total_flos += self.current_flos
            self.current_flos = 0

    def _get_learning_rate(self):
        if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            last_lr = self.optimizer.param_groups[0]["lr"]
        else:
            last_lr = self.lr_scheduler.get_last_lr()[0]
        if torch.is_tensor(last_lr):
            last_lr = last_lr.item()
        return last_lr

    def log(self, logs):
        ############### 设置epoch, step, tokens等 ###############
        if self.state.epoch is not None:
            logs['epoch'] = self.state.epoch
        if self.args.include_num_input_tokens_seen:
            logs["num_input_tokens_seen"] = self.state.num_input_tokens_seen
        output = {**logs, **{"step": self.state.global_step}}

        self.state.log_history.append(output)

        ############### 触发log相关Callbacks ###############
        self.control = self.callback_handler.on_log(self.args, self.state, self.control, logs)

    def __保存__(self):
        pass

    def _save_checkpoint(self, model, metrics=None):
        ############### 获取保存路径 ###############
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
        run_dir = self._get_output_dir()
        output_dir = os.path.join(run_dir, checkpoint_folder)

        ############### 保存model ###############
        self.save_model(output_dir)

        ############### 保存optimizer、scheduler ###############
        if not self.args.save_only_model:
            self._save_optimizer_and_scheduler(output_dir)

        ############### 保存trainer_state, rng_state ###############
        if self.args.should_save:
            self.store_flos()
            self.state.save_to_json(os.path.join(output_dir, TRAINER_STATE_NAME))

            if not self.args.save_only_model:
                self._save_rng_state(output_dir)

        ############### 判断best_model ###############
        if metrics is not None and self.args.metric_for_best_model is not None:
            metric_to_check = self.args.metric_for_best_model
            if not metric_to_check.startswith("eval_"):
                metric_to_check = f"eval_{metric_to_check}"
            metric_value = metrics[metric_to_check]

            operator = np.greater if self.args.greater_is_better else np.less
            if (
                    self.state.best_metric is None
                    or self.state.best_model_checkpoint is None
                    or operator(metric_value, self.state.best_metric)
            ):
                self.state.best_metric = metric_value
                self.state.best_model_checkpoint = output_dir

        ############### 删除历史checkpoints（如果设置了save_total_limit） ###############
        if self.args.should_save:
            self._rotate_checkpoints(use_mtime=False, output_dir=run_dir)

    def _get_output_dir(self):
        return self.args.output_dir

    def save_model(self, output_dir):
        output_dir = output_dir or self.args.output_dir
        state_dict = {}

        if self.is_fsdp_enabled:
            ############### 获取fsdp state_dict ###############
            if ("FULL_STATE_DICT" in str(self.accelerator.state.fsdp_plugin.state_dict_type)) and (
                    version.parse(accelerate_version) > version.parse("0.24.1")
            ):
                from torch.distributed.fsdp import FullStateDictConfig, StateDictType
                from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

                full_state_dict_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
                with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT, full_state_dict_config):
                    state_dict = self.model.state_dict()
        elif self.is_deepspeed_enabled:
            ############### 获取deepspeed state_dict ###############
            try:
                if self.accelerator.deepspeed_config["zero_optimization"]["stage"] == 3:
                    if self.deepspeed.zero_gather_16bit_weights_on_model_save():
                        state_dict = self.deepspeed._zero3_consolidated_16bit_state_dict()
                    else:
                        raise ValueError(
                            "Cannot get 16bit model weights because `stage3_gather_16bit_weights_on_model_save` in DeepSpeed config is False. "
                            "To save the model weights in 16bit, set `stage3_gather_16bit_weights_on_model_save` to True in DeepSpeed config file or "
                            "set `zero3_save_16bit_model` to True when using `accelerate config`. "
                            "To save the full checkpoint, run `model.save_checkpoint(save_dir)` and use `zero_to_fp32.py` to recover weights."
                        )
                else:
                    from deepspeed.checkpoint.utils import clone_tensors_for_torch_save

                    state_dict = clone_tensors_for_torch_save(self.accelerator.unwrap_model(self.deepspeed).state_dict())
            except ValueError:
                ############### 直接保存成deepspeed格式(需要用zero_to_fp32.py进行格式转换) ###############
                self.deepspeed.save_checkpoint(output_dir)

        if self.args.should_save:
            ############### 保存local checkpoint ###############
            self._save(output_dir, state_dict=state_dict)

    def _save(self, output_dir, state_dict=None):
        output_dir = output_dir or self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        ############### 保存state_dict ###############
        supported_classes = (PreTrainedModel,)
        if isinstance(self.model, supported_classes):
            self.model.save_pretrained(output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors)
        elif isinstance(unwrap_model(self.model), supported_classes):
            state_dict = state_dict or self.model.state_dict()
            unwrap_model(self.model).save_pretrained(output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors)
        else:
            state_dict = state_dict or self.model.state_dict()
            if self.args.save_safetensors:
                safetensors.torch.save_file(state_dict, os.path.join(output_dir, SAFE_WEIGHTS_NAME), metadata={"format": "pt"})
            else:
                torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))

        ############### 保存tokenizer (主要针对tokenizer) ###############
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        ############### 保存training_args (dict) ###############
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))

    def _save_optimizer_and_scheduler(self, output_dir):
        if self.is_deepspeed_enabled:
            ############### deepspeed ###############
            accept_exclude_frozen_parameters = "exclude_frozen_parameters" in set(
                inspect.signature(self.model_wrapped.save_checkpoint).parameters.keys()
            )
            # 保存ds_model参数及optimizer
            if accept_exclude_frozen_parameters:
                self.model_wrapped.save_checkpoint(output_dir, exclude_frozen_parameters=True)
            else:
                self.model_wrapped.save_checkpoint(output_dir)
        elif self.is_fsdp_enabled:
            ############### fsdp ###############
            # 保存fsdp_model参数
            save_fsdp_model(
                self.accelerator.state.fsdp_plugin, self.accelerator, self.model, output_dir
            )
            # 保存fsdp optimizer
            save_fsdp_optimizer(
                self.accelerator.state.fsdp_plugin, self.accelerator, self.optimizer, self.model, output_dir
            )
        elif self.args.should_save:
            ############### local ###############
            torch.save(self.optimizer.state_dict(), os.path.join(output_dir, OPTIMIZER_NAME))

        is_deepspeed_custom_scheduler = self.is_deepspeed_enabled and not isinstance(
            self.lr_scheduler, DeepSpeedSchedulerWrapper
        )
        if (
                self.args.should_save
                and (not self.is_deepspeed_enabled or is_deepspeed_custom_scheduler)
        ):
            ############### scheduler ###############
            torch.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, SCHEDULER_NAME))

    def _save_rng_state(self, output_dir):
        rng_states = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "cpu": torch.random.get_rng_state(),
        }
        if torch.cuda.is_available():
            if self.args.parallel_mode == ParallelMode.DISTRIBUTED:
                rng_states["cuda"] = torch.cuda.random.get_rng_state_all()
            else:
                rng_states["cuda"] = torch.cuda.random.get_rng_state()

        os.makedirs(output_dir, exist_ok=True)
        if self.args.world_size <= 1:
            torch.save(rng_states, os.path.join(output_dir, "rng_state.pth"))
        else:
            torch.save(rng_states, os.path.join(output_dir, f"rng_state_{self.args.process_index}.pth"))

    def _rotate_checkpoints(self, use_mtime, output_dir):
        if self.args.save_total_limit is None or self.args.save_total_limit <= 0:
            return

        checkpoints_sorted = self._sorted_checkpoints(use_mtime=use_mtime, output_dir=output_dir)
        if len(checkpoints_sorted) <= self.args.save_total_limit:
            return

        # If save_total_limit=1 with load_best_model_at_end=True, we could end up deleting the last checkpoint, which
        # we don't do to allow resuming.
        save_total_limit = self.args.save_total_limit
        if (
                self.state.best_model_checkpoint is not None
                and self.args.save_total_limit == 1
                and checkpoints_sorted[-1] != self.state.best_model_checkpoint
        ):
            save_total_limit = 2

        number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - save_total_limit)
        checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
        for checkpoint in checkpoints_to_be_deleted:
            shutil.rmtree(checkpoint, ignore_errors=True)

    def _sorted_checkpoints(self, output_dir, checkpoint_prefix=PREFIX_CHECKPOINT_DIR, use_mtime=False):
        ordering_and_checkpoint_path = []

        glob_checkpoints = [str(x) for x in Path(output_dir).glob(f"{checkpoint_prefix}-*") if os.path.isdir(x)]

        for path in glob_checkpoints:
            if use_mtime:
                ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
            else:
                regex_match = re.match(f".*{checkpoint_prefix}-([0-9]+)", path)
                if regex_match is not None and regex_match.groups() is not None:
                    ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

        checkpoints_sorted = sorted(ordering_and_checkpoint_path)
        checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
        # Make sure we don't delete the best model.
        if (
                self.state.best_model_checkpoint is not None
                and str(Path(self.state.best_model_checkpoint)) in checkpoints_sorted
        ):
            best_model_index = checkpoints_sorted.index(str(Path(self.state.best_model_checkpoint)))
            for i in range(best_model_index, len(checkpoints_sorted) - 2):
                checkpoints_sorted[i], checkpoints_sorted[i + 1] = checkpoints_sorted[i + 1], checkpoints_sorted[i]
        return checkpoints_sorted

    def __加载__(self):
        pass

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        model = model or self.model

        ############### load config ###############
        config_file = os.path.join(resume_from_checkpoint, CONFIG_NAME)
        if os.path.isfile(config_file):
            config = PretrainedConfig.from_json_file(config_file)
            checkpoint_version = config.transformers_version
            if checkpoint_version is not None and checkpoint_version != transformers.__version__:
                logger.warning(
                    f"checkpoint version: {checkpoint_version} 与当前transformers version: {transformers.__version__} 不一致，"
                    "可能导致不兼容问题"
                )

        ############### load DeepSpeed checkpoint ###############
        if self.is_deepspeed_enabled:
            deepspeed_load_checkpoint(model, resume_from_checkpoint)
            return

        ############### load FSDP checkpoint ###############
        is_fsdp_ckpt = os.path.isdir(resume_from_checkpoint) and (
            # this checks the FSDP state dict when `SHARDED_STATE_DICT` is used
                any(
                    FSDP_MODEL_NAME in folder_name
                    for folder_name in os.listdir(resume_from_checkpoint)
                    if os.path.isdir(os.path.join(resume_from_checkpoint, folder_name))
                )
                # this checks the FSDP state dict when `FULL_STATE_DICT` is used
                or os.path.isfile(os.path.join(resume_from_checkpoint, f"{FSDP_MODEL_NAME}.bin"))
        )
        if is_fsdp_ckpt:
            assert self.is_fsdp_enabled, "FSDP checkpoint found but FSDP is not enabled"
            load_fsdp_model(  # 会对FSDP的FULL_STATE_DICT、LOCAL_STATE_DICT、SHARDED_STATE_DICT三种模式分别处理
                self.accelerator.state.fsdp_plugin,
                self.accelerator,
                model,
                resume_from_checkpoint,
            )
            return

        ############### load sharded checkpoint ###############
        weights_index_file = os.path.join(resume_from_checkpoint, WEIGHTS_INDEX_NAME)
        safe_weights_index_file = os.path.join(resume_from_checkpoint, SAFE_WEIGHTS_INDEX_NAME)
        if os.path.exists(weights_index_file) or os.path.exists(safe_weights_index_file):
            load_result = load_sharded_checkpoint(
                model, resume_from_checkpoint, strict=False, prefer_safe=self.args.save_safetensors
            )
            self._issue_warnings_after_load(load_result)
            return

        ############### load local checkpoint ###############
        weights_file = os.path.join(resume_from_checkpoint, WEIGHTS_NAME)
        safe_weights_file = os.path.join(resume_from_checkpoint, SAFE_WEIGHTS_NAME)
        if os.path.isfile(weights_file) or os.path.isfile(safe_weights_file):
            weights_only_kwarg = {"weights_only": True} if is_torch_greater_or_equal_than_1_13 else {}
            if self.args.save_safetensors and os.path.isfile(safe_weights_file):
                state_dict = safetensors.torch.load_file(safe_weights_file, device="cpu")
            else:
                state_dict = torch.load(
                    weights_file,
                    map_location="cpu",
                    **weights_only_kwarg,
                )
            load_result = model.load_state_dict(state_dict, False)
            del state_dict

            self._issue_warnings_after_load(load_result)
            return

    def _issue_warnings_after_load(self, load_result):
        if len(load_result.missing_keys) != 0:
            if self.model._keys_to_ignore_on_save is not None and set(load_result.missing_keys) == set(self.model._keys_to_ignore_on_save):
                # 有丢失的weight，但属于_keys_to_ignore_on_save，默认当做共享weight，所以调用PreTrainedModel.tie_weights()对共享weight进行绑定
                self.model.tie_weights()
            else:
                logger.warning(f"There were missing keys in the checkpoint model loaded: {load_result.missing_keys}.")
        if len(load_result.unexpected_keys) != 0:
            logger.warning(
                f"There were unexpected keys in the checkpoint model loaded: {load_result.unexpected_keys}."
            )

    def _load_best_model(self):
        logger.info(f"Loading best model from {self.state.best_model_checkpoint} (score: {self.state.best_metric}).")

        if self.is_deepspeed_enabled or self.is_fsdp_enabled:
            self._load_from_checkpoint(self.state.best_model_checkpoint, model=self.model_wrapped)
        else:
            self._load_from_checkpoint(self.state.best_model_checkpoint)

    def _load_optimizer_and_scheduler(self, checkpoint):
        if checkpoint is None: return

        ############### deepspeed ###############
        if self.is_deepspeed_enabled and isinstance(self.lr_scheduler, DeepSpeedSchedulerWrapper):
            # deepspeed loads optimizer/lr_scheduler together with the model in deepspeed_init
            return

        ############### fsdp ###############
        if self.is_fsdp_enabled:
            load_fsdp_optimizer(
                self.accelerator.state.fsdp_plugin,
                self.accelerator,
                self.optimizer,
                self.model,
                checkpoint,
            )
            return

        ############### 本地optimizer、scheduler ###############
        checkpoint_file_exists = (
                os.path.isfile(os.path.join(checkpoint, OPTIMIZER_NAME)) or os.path.isfile(os.path.join(checkpoint, OPTIMIZER_NAME_BIN))
                or (
                        os.path.isdir(checkpoint)
                        and any(OPTIMIZER_NAME_BIN.split(".")[0] in folder_name
                                for folder_name in os.listdir(checkpoint)
                                if os.path.isdir(os.path.join(checkpoint, folder_name))
                                )
                )
        )
        if checkpoint_file_exists and os.path.isfile(os.path.join(checkpoint, SCHEDULER_NAME)):
            map_location = self.args.device if self.args.world_size > 1 else "cpu"
            self.optimizer.load_state_dict(torch.load(os.path.join(checkpoint, OPTIMIZER_NAME), map_location=map_location))
            with warnings.catch_warnings(record=True) as caught_warnings:
                self.lr_scheduler.load_state_dict(torch.load(os.path.join(checkpoint, SCHEDULER_NAME)))
            reissue_pt_warnings(caught_warnings)

    def _load_rng_state(self, checkpoint):
        if checkpoint is None:
            return

        rng_file = os.path.join(checkpoint, "rng_state.pth")
        if not os.path.isfile(rng_file):
            return

        checkpoint_rng_state = torch.load(rng_file)
        random.setstate(checkpoint_rng_state["python"])
        np.random.set_state(checkpoint_rng_state["numpy"])
        torch.random.set_rng_state(checkpoint_rng_state["cpu"])

        if torch.cuda.is_available():
            torch.cuda.random.set_rng_state_all(checkpoint_rng_state["cuda"])

    def __其他方法__(self):
        pass

    def is_local_process_zero(self):
        return self.args.local_process_index == 0

    def is_world_process_zero(self):
        return self.args.process_index == 0


if __name__ == "__main__":
    pass
