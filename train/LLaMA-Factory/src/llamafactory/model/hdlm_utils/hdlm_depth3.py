from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaPreTrainedModel, LlamaConfig, DynamicCache, StaticCache, Cache, \
    LlamaModel
from transformers import AutoTokenizer, LlamaForCausalLM
import inspect
import math
import copy
import os
import time
import json
import warnings
from collections import defaultdict
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask, \
    _prepare_4d_causal_attention_mask_for_sdpa
from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutputWithPast
from transformers.models.llama import modeling_llama
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRMSNorm
from transformers.utils import (
    logging,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# add system path
current_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_path)


from .hdlm_prompts import (
    esc_subtask1_prompt,
    esc_subtask2_prompt,
    dbp_subtask_prompt,
    id2strategy,
    Label1,
    Label2,
    Label3,
)


class Depth3_HdLMModel(LlamaPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config, tokenizer_path,first_layer_index=None, second_layer_index=None, 
                 final_loss_weight=1, first_loss_weight=1, second_loss_weight=1,
                 record_losses=False):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        if first_layer_index is not None:
            self.first_layer_index = first_layer_index
            self.first_lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        else:
            self.first_layer_index = None

        if second_layer_index is not None:
            self.second_layer_index = second_layer_index
            self.second_lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        else:
            self.second_layer_index = None
        self.final_loss_weight = final_loss_weight
        self.first_loss_weight = first_loss_weight
        self.second_loss_weight = second_loss_weight
        self.record_losses = record_losses
        
        if self.record_losses:
            self.final_losses = []
            self.first_losses = []
            self.second_losses = []
        
        self.first_thought_token_ids = self.tokenizer.encode("<|start_header_id|>first_thought<|end_header_id|>\n\n", add_special_tokens=False)
        self.second_thought_token_ids = self.tokenizer.encode("<|start_header_id|>second_thought<|end_header_id|>\n\n", add_special_tokens=False)
        # print(thought_token_ids)
        self.assistant_token_ids = self.tokenizer.encode("<|start_header_id|>assistant<|end_header_id|>\n\n", add_special_tokens=False)
        self.subtask_1_token_ids = self.tokenizer.encode("<|start_header_id|>subtask_1<|end_header_id|>\n\n", add_special_tokens=False)
        self.subtask_2_token_ids = self.tokenizer.encode("<|start_header_id|>subtask_2<|end_header_id|>\n\n", add_special_tokens=False)
        self.end_token_ids = self.tokenizer.encode("<|eot_id|>", add_special_tokens=False)
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings
    
    def clone_lm_head_to_new_lm_head(self):
        self.first_lm_head.weight.data = self.lm_head.weight.data.clone()
        self.second_lm_head.weight.data = self.lm_head.weight.data.clone()


    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        cache_position=None,
        use_first_head=False,
        use_second_head=False,
         **kwargs,
    ):

        use_first_head = kwargs.get("use_first_head", use_first_head)
        use_second_head = kwargs.get("use_second_head", use_second_head)

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids, 
            attention_mask=attention_mask, 
            position_ids=position_ids, 
            past_key_values=past_key_values, 
            inputs_embeds=inputs_embeds, 
            use_cache=use_cache, 
            output_attentions=output_attentions, 
            output_hidden_states=True,
            return_dict=return_dict, 
        )


        final_hidden_states = outputs[0]
        final_logits = self.lm_head(final_hidden_states)
        final_logits = final_logits.float()
        
        first_logits = None
        second_logits = None

        loss = None
        final_loss = None
        first_loss = None
        second_loss = None

        if labels is not None:
            def find_subsequence_indices(sequence, subsequence):
                subsequence_length = len(subsequence)
                index_list = []
                for i in range(len(sequence) - subsequence_length + 1):
                    if torch.equal(sequence[i:i + subsequence_length], torch.tensor(subsequence, device=sequence.device)):
                        index_list.append(i)
                return index_list

            batch_size = labels.size(0)
            sequence_length = labels.size(1)
            first_thought_loss_mask = torch.zeros_like(labels, dtype=torch.float32)
            second_thought_loss_mask = torch.zeros_like(labels, dtype=torch.float32)
            assistant_loss_mask = torch.zeros_like(labels, dtype=torch.float32)

            for batch_idx in range(batch_size):
                first_thought_start_list = find_subsequence_indices(labels[batch_idx], self.first_thought_token_ids)
                second_thought_start_list = find_subsequence_indices(labels[batch_idx], self.second_thought_token_ids)
                subtask_1_start_list = find_subsequence_indices(labels[batch_idx], self.subtask_1_token_ids)
                subtask_2_start_list = find_subsequence_indices(labels[batch_idx], self.subtask_2_token_ids)
                assistant_start_list = find_subsequence_indices(labels[batch_idx], self.assistant_token_ids)
                
                if len(first_thought_start_list) > 0 and len(assistant_start_list) > 0 and len(first_thought_start_list)==len(assistant_start_list):
                    for i in range(0, len(first_thought_start_list)):
                        first_thought_loss_mask[batch_idx, first_thought_start_list[i] + len(self.first_thought_token_ids) : subtask_1_start_list[i]] = 1
                        second_thought_loss_mask[batch_idx, second_thought_start_list[i] + len(self.second_thought_token_ids) : subtask_2_start_list[i]] = 1
                        
                        next_assistant_end = torch.where(labels[batch_idx][assistant_start_list[i] + len(self.assistant_token_ids):]==128009)[0][0].item()
                        # assistant_loss_mask[batch_idx, assistant_start_list[i] + len(self.assistant_token_ids) : assistant_start_list[i] + len(self.assistant_token_ids) + next_assistant_end + 1] = 1
                        assistant_loss_mask[batch_idx, assistant_start_list[i] + len(self.assistant_token_ids):] = 1
            loss_fct = nn.CrossEntropyLoss()

            if self.first_layer_index is not None:
                first_hidden_states = outputs.hidden_states[self.first_layer_index]
                first_logits = self.first_lm_head(first_hidden_states)
                shift_first_logits = first_logits[..., :-1, :].contiguous()
                first_labels = torch.where(first_thought_loss_mask > 0, labels, torch.tensor(-100).to(labels.device))
                shift_first_labels = first_labels[...,1:].contiguous()
                first_loss = loss_fct(shift_first_logits.view(-1, self.config.vocab_size),
                                      shift_first_labels.view(-1).to(first_logits.device))
                print(f">>> First_loss: {first_loss}")
            
            if self.second_layer_index is not None:
                second_hidden_states = outputs.hidden_states[self.second_layer_index]
                second_logits = self.second_lm_head(second_hidden_states)
                shift_second_logits = second_logits[..., :-1, :].contiguous()
                second_labels = torch.where(second_thought_loss_mask > 0, labels, torch.tensor(-100).to(labels.device))
                shift_second_labels = second_labels[...,1:].contiguous()
                second_loss = loss_fct(shift_second_logits.view(-1, self.config.vocab_size),
                                      shift_second_labels.view(-1).to(second_logits.device))
                print(f">>> Second_loss: {second_loss}")

            shift_logits = final_logits[..., :-1, :].contiguous()
            assistant_labels = torch.where(assistant_loss_mask > 0, labels, torch.tensor(-100).to(labels.device))
            shift_assistant_labels = assistant_labels[..., 1:].contiguous()
            final_loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), 
                                      shift_assistant_labels.view(-1).to(final_logits.device))
            print(f">>> Final_loss: {final_loss}")
            # compute total loss
            if first_loss is not None and second_loss is not None:
                loss = final_loss * self.final_loss_weight + first_loss * self.first_loss_weight + second_loss * self.second_loss_weight
                if self.record_losses:
                    self.final_losses.append(final_loss.item())
                    self.first_losses.append(first_loss.item())
                    self.second_losses.append(second_loss.item())
            else:
                loss = final_loss
                if self.record_losses:
                    self.final_losses.append(final_loss.item())
            print(f"Loss: {loss}")

        if self.first_layer_index is not None and self.second_layer_index is not None:
            first_logits = self.first_lm_head(outputs.hidden_states[self.first_layer_index])
            second_logits = self.second_lm_head(outputs.hidden_states[self.second_layer_index])
        
        if use_first_head and self.first_layer_index is not None:
            logits = first_logits.float()
        elif use_second_head and self.second_layer_index is not None:
            logits = second_logits.float()
        else:
            logits = final_logits

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def _logits_to_text(self, logits):
        predicted_ids = torch.argmax(logits, dim=-1)
        return self.tokenizer.decode(predicted_ids[0].tolist(), skip_special_tokens=False)

    def prepare_inputs_for_generation(
            self,
            input_ids,
            past_key_values=None,
            attention_mask=None,
            inputs_embeds=None,
            cache_position=None,
            position_ids=None,
            use_cache=True,
            **kwargs,
        ):
            # If we have cache: let's slice `input_ids` through `cache_position`, to keep only the unprocessed tokens
            # Exception 1: when passing input_embeds, input_ids may be missing entries
            # Exception 2: some generation methods do special slicing of input_ids, so we don't need to do it here

            use_first_head = kwargs.get("use_first_head", False)
            use_second_head = kwargs.get("use_second_head", False)

            if past_key_values is not None:
                if inputs_embeds is not None:  # Exception 1
                    input_ids = input_ids[:, -cache_position.shape[0] :]
                elif input_ids.shape[1] != cache_position.shape[0]:  # Default case (the "else", a no op, is Exception 2)
                    input_ids = input_ids[:, cache_position]

            if attention_mask is not None and position_ids is None:
                # create position_ids on the fly for batch generation
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                if past_key_values:
                    position_ids = position_ids[:, -input_ids.shape[1] :]

                    # This `clone` call is needed to avoid recapturing cuda graphs with `torch.compile`'s  `mode="reduce-overhead`, as otherwise the input `position_ids` would have various stride during the decoding. Here, simply using `.contiguous()` is not sufficient as in the batch size = 1 case, `position_ids` is already contiguous but with varying stride which retriggers a capture.
                    position_ids = position_ids.clone(memory_format=torch.contiguous_format)

            # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
            if inputs_embeds is not None and cache_position[0] == 0:
                model_inputs = {"inputs_embeds": inputs_embeds, "input_ids": None}
            else:
                # The clone here is for the same reason as for `position_ids`.
                model_inputs = {"input_ids": input_ids.clone(memory_format=torch.contiguous_format), "inputs_embeds": None}

            if isinstance(past_key_values, StaticCache) and attention_mask.ndim == 2:
                if model_inputs["inputs_embeds"] is not None:
                    batch_size, sequence_length, _ = model_inputs["inputs_embeds"].shape
                    device = model_inputs["inputs_embeds"].device
                else:
                    batch_size, sequence_length = model_inputs["input_ids"].shape
                    device = model_inputs["input_ids"].device

                dtype = self.lm_head.weight.dtype
                min_dtype = torch.finfo(dtype).min

                attention_mask = _prepare_4d_causal_attention_mask_with_cache_position(
                    attention_mask,
                    sequence_length=sequence_length,
                    target_length=past_key_values.get_max_length(),
                    dtype=dtype,
                    device=device,
                    min_dtype=min_dtype,
                    cache_position=cache_position,
                    batch_size=batch_size,
                )

            model_inputs.update(
                {
                    "position_ids": position_ids,
                    "cache_position": cache_position,
                    "past_key_values": past_key_values,
                    "use_cache": use_cache,
                    "attention_mask": attention_mask,
                    "use_first_head": use_first_head,
                    "use_second_head": use_second_head,  # 传递到 forward 中
                }
            )
            return model_inputs



    def plot_losses(self, fig_save_dir):
        if self.record_losses:
            with open(f"{fig_save_dir}/three_losses.jsonl", "w") as jsonl_file:
                for step, final_loss, second_loss, first_loss in zip(range(len(self.final_losses)), self.final_losses, self.second_losses, self.first_losses):
                    jsonl_file.write(json.dumps({"step": step, "first_losses": first_loss, "second_losses": second_loss, "final_losses": final_loss}) + "\n")
            plt.figure(figsize=(10, 5))

            plt.plot(self.final_losses, label="Final Loss", alpha=0.5, color='#FF0000')
            plt.plot(self.second_losses, label="Second Loss", alpha=0.5, color='#00FF00')
            plt.plot(self.first_losses, label="First Loss", alpha=0.5, color='#0000FF')
            
            final_losses_smooth = savgol_filter(self.final_losses, window_length=5, polyorder=2)
            second_losses_smooth = savgol_filter(self.second_losses, window_length=5, polyorder=2)
            first_losses_smooth = savgol_filter(self.first_losses, window_length=5, polyorder=2)
            plt.plot(self.final_losses, label="Final Loss (Smooth)", alpha=0.3, color='#FF0000')
            plt.plot(self.second_losses, label="Second Loss (Smooth)", alpha=0.3, color='#00FF00')
            plt.plot(self.first_losses, label="First Loss (Smooth)", alpha=0.3, color='#0000FF')
            
            plt.xlabel("Training Step")
            plt.ylabel("Loss")
            plt.title("Loss Curves")
            plt.legend()
            plt.savefig(f"{fig_save_dir}/intermediate_and_final_losses.png") 
            plt.show()

            
    ## test chat
    def chat(self, 
          tokenizer,
          query: str, 
          system_prompt: str = None,
          history: List[Tuple[str, str]] = None, 
          max_length: int = None, 
          max_new_tokens: int = 256, 
          num_beams: int = 1, 
          do_sample: bool = False, 
          top_k: float = None,
          top_p: float = 1, 
          temperature: float = 1, 
          think: bool = False, 
          logits_processor = None, 
          dataset: str = None,
          **kwargs) -> str:
        input_text = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n" \
                     + system_prompt \
                     + "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n" \
                     + query + "<|eot_id|>"
        if think:
            input_text += "<|start_header_id|>first_thought<|end_header_id|>\n\n"
            input_ids = tokenizer.encode(input_text, return_tensors='pt', add_special_tokens=False).to(self.device)
            # First Think
            first_thought_output = self.generate(
                use_first_head=True,
                input_ids=input_ids,
                max_length=max_length,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                do_sample=do_sample,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                logits_processor=logits_processor,
                eos_token_id=[128001,128009],
                **kwargs
            )
            label1 = tokenizer.decode(first_thought_output[0][len(input_ids[0]):], skip_special_tokens=False).split("<|eot_id|>")[0]

            if dataset == "ESC":
                subtask1_text = esc_subtask1_prompt.format(strategy_list=id2strategy)
                input_text += f"{label1}<|eot_id|><|start_header_id|>subtask_1<|end_header_id|>\n\n"
                input_text += f"{subtask1_text}<|eot_id|><|start_header_id|>second_thought<|end_header_id|>\n\n"
            else:  ## DBP cls
                try:
                    label_1 = int(label1)
                except ValueError:
                    label_1 = -2
                    return -2,-1,-1
                first_label = Label1.get(label_1,None)
                subtask1_text = dbp_subtask_prompt.format(label_list=Label2[first_label])
                input_text += f"{label1}<|eot_id|><|start_header_id|>subtask_1<|end_header_id|>\n\n"
                input_text += f"{subtask1_text}<|eot_id|><|start_header_id|>second_thought<|end_header_id|>\n\n"
            input_ids = tokenizer.encode(input_text, return_tensors='pt',add_special_tokens=False).to(self.device)
        
            # Second Think
            second_thought_output = self.generate(
                use_second_head=True,
                input_ids=input_ids,
                max_length=max_length,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                do_sample=do_sample,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                logits_processor=logits_processor,
                eos_token_id=[128001,128009],
                **kwargs
            )
            label2 = self.tokenizer.decode(second_thought_output[0][len(input_ids[0]):], skip_special_tokens=False).split("<|eot_id|>")[0]
            # print("label2:\n", label2)
            if dataset == "ESC":
                subtask2_text = esc_subtask2_prompt
                input_text += f"{label2}<|eot_id|><|start_header_id|>subtask_2<|end_header_id|>\n\n"
                input_text += f"{subtask2_text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            else:
                try:
                    label_2 = int(label2)
                except ValueError:
                    label_2 = -1
                second_label = Label2[first_label].get(label_2, None)
                subtask2_text = dbp_subtask_prompt.format(label_list=Label3[second_label])
                input_text += f"{label2}<|eot_id|><|start_header_id|>subtask_2<|end_header_id|>\n\n"
                input_text += f"{subtask2_text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

            input_ids = tokenizer.encode(input_text, return_tensors='pt',add_special_tokens=False).to(self.device)

            # Answer
            response_output = self.generate(
                input_ids=input_ids,
                max_length=max_length,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                do_sample=do_sample,
                top_p=top_p,
                temperature=temperature,
                logits_processor=logits_processor,
                eos_token_id=[128001,128009],
                **kwargs
            )
            label3 = self.tokenizer.decode(response_output[0][len(input_ids[0]):], skip_special_tokens=True)
            return label1, label2, label3

        else:
            response_text = "HdLM must think!"
            return response_text