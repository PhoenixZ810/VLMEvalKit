import torch
from PIL import Image
from abc import abstractproperty
import sys
import os.path as osp
from .base import BaseModel
from ..smp import *
from ..dataset import DATASET_TYPE, DATASET_MODALITY
import copy
import requests


class Wings(BaseModel):

    INSTALL_REQ = True
    INTERLEAVE = True

    def __init__(self, model_path="liuhaotian/llava_v1.5_7b", **kwargs):
        try:
            from wings.model.base_architecture import WingsMetaForCausalLM
            from wings.arguments import ModelArguments, DataArguments, TrainingArguments
            from wings.utils import load_from_safetensors
        except Exception as err:
            logging.critical("Please install llava from https://github.com/haotian-liu/LLaVA")
            raise err

        assert osp.exists(model_path) or splitlen(model_path) == 2
        self.system_prompt = (
            "A chat between a curious human and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the human's questions. "
        )
        self.stop_str = "</s>"

        config = wings_config

        if model_path is not None:
            config['model_args']['model_safetensors_load_path'] = model_path

        local_model_args = ModelArguments(**config['model_args'])
        data_args = DataArguments(**config['data_args'])
        training_args = TrainingArguments(**config['training_args'])

        self.model, self.tokenizer, self.conversation_formatter = WingsMetaForCausalLM.build(
            model_name=local_model_args.model_name,
            model_path=local_model_args.model_path,
            conversation_formatter_kwargs={
                'system_slot': local_model_args.system_slot,
                'user_slot': local_model_args.user_slot,
                'gpt_slot': local_model_args.gpt_slot,
                'eot': local_model_args.eot,
            },
            model_max_length=local_model_args.model_max_length,
        )

        self.model.get_model().initialize_vision_modules(model_args=local_model_args, fsdp=training_args.fsdp)

        if hasattr(self.model, 'initialize_modules'):
            self.model.initialize_modules(
                model_args=local_model_args,
                data_args=data_args,
                training_args=training_args,
            )
        self.model.config.tokenizer_padding_side = self.tokenizer.padding_side
        self.model.config.tokenizer_max_length = self.tokenizer.model_max_length

        if local_model_args.model_safetensors_load_path is not None:
            self.model.load_state_dict(
                load_from_safetensors(self.model, local_model_args.model_safetensors_load_path)
            )

        vision_tower = self.model.get_vision_tower()
        vision_tower.to(
            dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device
        )
        self.model.to(torch.bfloat16)

        self.model = self.model.cuda()

        # self.conv_mode = "llava_v1"

        kwargs_default = dict(
            do_sample=False,
            temperature=0,
            max_new_tokens=512,
            top_p=None,
            num_beams=1,
            use_cache=True,
        )  # noqa E501
        kwargs_default.update(kwargs)
        self.kwargs = kwargs_default
        warnings.warn(f"Following kwargs received: {self.kwargs}, will use as generation config. ")

    def use_custom_prompt(self, dataset):
        assert dataset is not None
        if DATASET_TYPE(dataset) == "MCQ":
            return True
        return False

    def build_prompt(self, line, dataset=None):
        assert self.use_custom_prompt(dataset)
        assert dataset is None or isinstance(dataset, str)
        tgt_path = self.dump_image(line, dataset)

        question = line["question"]
        hint = line["hint"] if ("hint" in line and not pd.isna(line["hint"])) else None
        if hint is not None:
            question = hint + "\n" + question

        options = {
            cand: line[cand] for cand in string.ascii_uppercase if cand in line and not pd.isna(line[cand])
        }
        for key, item in options.items():
            question += f"\n{key}. {item}"
        prompt = question

        if len(options):
            prompt += (
                "\n请直接回答选项字母。"
                if cn_string(prompt)
                else "\nAnswer with the option's letter from the given choices directly."
            )
        else:
            prompt += "\n请直接回答问题。" if cn_string(prompt) else "\nAnswer the question directly."

        message = [dict(type="image", value=s) for s in tgt_path]
        message.append(dict(type="text", value=prompt))
        return message

    def concat_tilist(self, message):
        text, images = "", []
        for item in message:
            if item["type"] == "text":
                text += item["value"]
            elif item["type"] == "image":
                text += " <image> "
                images.append(item["value"])
        return text, images

    def chat_inner(self, message, dataset=None):
        from llava.mm_utils import (
            process_images,
            tokenizer_image_token,
            KeywordsStoppingCriteria,
        )
        from llava.constants import IMAGE_TOKEN_INDEX

        prompt = self.system_prompt
        images = []
        for utter in message:
            prompt += "USER: " if utter["role"] == "user" else "ASSISTANT: "
            content, images_sub = self.concat_tilist(utter["content"])
            prompt += content
            images.extend(images_sub)
            prompt += " " if utter["role"] == "user" else self.stop_str
        assert message[-1]["role"] == "user", message
        prompt += "ASSISTANT: "

        images = [Image.open(s).convert("RGB") for s in images]
        args = abstractproperty()
        args.image_aspect_ratio = "pad"
        image_tensor = process_images(images, self.image_processor, args).to("cuda", dtype=torch.float16)

        input_ids = (
            tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            .unsqueeze(0)
            .cuda()
        )
        keywords = [self.stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor,
                stopping_criteria=[stopping_criteria],
                **self.kwargs,
            )
        output = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        return output

    def generate_inner(self, message, dataset=None):

        # Support interleave text and image
        content, images = self.concat_tilist(message)

        images = [Image.open(s).convert("RGB") for s in images]
        # args = abstractproperty()
        # args.image_aspect_ratio = "pad"
        # if images:
        #     image_tensor = process_images(images, self.image_processor, args).to("cuda", dtype=torch.float16)
        # else:
        #     image_tensor = None

        image_processor = getattr(self.model.get_vision_tower(), 'image_processor', None)
        if images is not None:
            image_tensor = process_images(images, image_processor, self.model.config).cuda()
        else:
            image_tensor = None

        prompt, input_ids = self.conversation_formatter.format_query(content)
        do_sample = False
        input_ids = input_ids.unsqueeze(0).cuda()

        # prompt = self.system_prompt + "USER: " + content + " ASSISTANT: "

        # input_ids = (
        #     tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        #     .unsqueeze(0)
        #     .cuda()
        # )
        keywords = [self.stop_str]
        # stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor,
                do_sample=False,
                num_beams=1,
                max_new_tokens=512,
                repetition_penalty=None,
                use_cache=True,
            )

        # output = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        input_token_len = input_ids.shape[1]
        output = self.tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[
            0
        ].strip()
        return output


def retain_only_first_sub_str(s, sub_s):
    first_index = s.find(sub_s)

    if first_index != -1:
        s = s[: first_index + len(sub_s)] + s[first_index + len(sub_s) :].replace(sub_s, '')
    return s


def remove_image_token(instruction, tokens):
    for src in tokens:
        if src in instruction:
            instruction = instruction.replace(src, '')
    return instruction


def replace_image_token(instruction, source_default_tokens, target_tokens, leaved_token_num=1):
    if isinstance(target_tokens, str):
        target_tokens = [target_tokens] * len(source_default_tokens)
    target_id = 0
    for src in source_default_tokens:
        if src in instruction:
            instruction = instruction.replace(src, target_tokens[target_id])
            instruction = retain_only_first_sub_str(instruction, target_tokens[target_id])
            target_id += 1
        if target_id >= leaved_token_num:
            break
    if target_id == 0 and len(target_tokens) > 0:
        instruction = target_tokens[0] + '\n' + instruction
    else:
        instruction = remove_image_token(instruction, source_default_tokens)
    return instruction


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


def process_images(images, image_processor, model_cfg):
    image_aspect_ratio = getattr(model_cfg, "image_aspect_ratio", None)
    new_images = []
    if image_aspect_ratio == 'pad':
        for image in images:
            image = expand2square(image, tuple(int(x * 255) for x in image_processor.image_mean))
            image = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            new_images.append(image)
    else:
        return image_processor(images, return_tensors='pt')['pixel_values']
    if all(x.shape == new_images[0].shape for x in new_images):
        new_images = torch.stack(new_images, dim=0)
    return new_images


wings_config = {
    "model_args": {
        "model_name": "wings_qwen2",
        "model_path": "Qwen/Qwen1.5-7B-Chat",
        "vision_tower": "google/siglip-so400m-patch14-384",
        "mm_vision_select_layer": -2,
        "mm_projector_type": "mlp2x_gelu",
        "mm_patch_merge_type": "flat",
        "mm_vision_select_feature": "patch",
        "system_prompt_length": 14,
        "model_max_length": 2048,
        "model_safetensors_load_path": "your_trained_safetensors_file_path",
        "moe_tune_mm_projector": True,
        "v_enable": False,
        "dmoe_enable": False,
        "dmoe_tune_mm_projector": True,
        "dmoe_params_init_mode": "copy",
        "dmoe_mode": "",
        "damoe_enable": False,
        "dlmoe_enable": False,
        "dlatt_enable": False,
        "dlora_enable": False,
        "damoe_ep_size": 1,
        "damoe_top_k_experts": 2,
        "damoe_capacity_factor": 1.0,
        "damoe_eval_capacity_factor": 2.0,
        "damoe_min_capacity": 0,
        "damoe_use_residual": False,
        "damoe_router_aux_loss_coef": 0.01,
        "peft_enable": False,
        "peft_mode": "",
        "peft_kwargs_k": [],
        "peft_kwargs_v": [],
        "lora_enable": False,
        "lora_r": False,
        "lora_dim": 64,
        "attn_layers_idx": [
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            29,
            30,
            31,
        ],
        "wings_router_type": "linear",
        "moe_enable": False,
        "moe_mode": "second_half",
        "ep_size": 1,
        "top_k_experts": 2,
        "capacity_factor": 1.0,
        "eval_capacity_factor": 2.0,
        "min_capacity": 0,
        "use_residual": False,
        "router_aux_loss_coef": 0.01,
    },
    "data_args": {
        "data_name": "",
        "data_dir": "",
        "is_multimodal": True,
        "image_aspect_ratio": "pad",
        "only_image_data": False,
        "image_text_data_ratio": 0.0,
        "image_token_length": 729,
        "data": "",
        "model": "",
        "work_dir": ".",
        "mode": "all",
        "nproc": 4,
        "ignore": False,
        "verbose": False,
        "prefetch": False,
        "time_str": "",
    },
    "training_args": {
        "output_dir": "temp",
        "overwrite_output_dir": False,
        "do_train": False,
        "do_eval": False,
        "do_predict": False,
        "evaluation_strategy": "no",
        "prediction_loss_only": False,
        "per_device_train_batch_size": 4,
        "per_device_eval_batch_size": 8,
        "gradient_accumulation_steps": 2,
        "eval_delay": 0,
        "learning_rate": 2e-06,
        "weight_decay": 1e-08,
        "adam_beta1": 0.9,
        "adam_beta2": 0.999,
        "adam_epsilon": 1e-08,
        "max_grad_norm": 1.0,
        "num_train_epochs": 1.0,
        "max_steps": -1,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.03,
        "warmup_steps": 0,
        "log_level": "passive",
        "log_level_replica": "warning",
        "log_on_each_node": True,
        "logging_dir": "",
        "logging_strategy": "steps",
        "logging_first_step": False,
        "logging_steps": 1.0,
        "logging_nan_inf_filter": True,
        "save_strategy": "steps",
        "save_steps": 2000,
        "save_total_limit": 1,
        "save_safetensors": True,
        "save_on_each_node": False,
        "save_only_model": False,
        "no_cuda": False,
        "use_cpu": False,
        "use_mps_device": False,
        "seed": 42,
        "jit_mode_eval": False,
        "use_ipex": False,
        "bf16": True,
        "fp16": False,
        "fp16_opt_level": "O1",
        "half_precision_backend": "auto",
        "bf16_full_eval": False,
        "fp16_full_eval": False,
        "tf32": True,
        "local_rank": 0,
        "tpu_metrics_debug": False,
        "debug": [],
        "dataloader_drop_last": False,
        "dataloader_num_workers": 4,
        "past_index": -1,
        "run_name": "",
        "disable_tqdm": False,
        "remove_unused_columns": False,
        "load_best_model_at_end": False,
        "ignore_data_skip": False,
        "fsdp": [],
        "fsdp_min_num_params": 0,
        "label_smoothing_factor": 0.0,
        "optim": "adamw_torch",
        "adafactor": False,
        "group_by_length": False,
        "length_column_name": "length",
        "report_to": [],
        "dataloader_pin_memory": True,
        "dataloader_persistent_workers": False,
        "skip_memory_metrics": True,
        "use_legacy_prediction_loop": False,
        "push_to_hub": False,
        "hub_strategy": "every_save",
        "hub_private_repo": False,
        "hub_always_push": False,
        "gradient_checkpointing": True,
        "include_inputs_for_metrics": False,
        "fp16_backend": "auto",
        "mp_parameters": "",
        "auto_find_batch_size": False,
        "full_determinism": False,
        "ray_scope": "last",
        "ddp_timeout": 1800,
        "torch_compile": False,
        "include_tokens_per_second": False,
        "include_num_input_tokens_seen": False,
        "mm_projector_lr": 1e-05,
        "vision_tower_lr_follow_mm_projector": True,
        "lr_projector_follow_tuned_keys": ["mm_projector"],
        "group_by_modality_length": False,
        "use_cache": False,
        "tuned_keys": [".attn_pool.", ".attn_t_pool.", ".reweight_module."],
        "tune_mm_projector": True,
        "tune_llm": True,
        "tune_vision_tower": False,
        "tune_only_mm_mlp_adapter": False,
    },
}
