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
import sys

# add system path
current_path = os.path.abspath(__file__)
print(current_path)
base_dir = current_path.split("src/llamafactory")[0]
# sys.path.append(base_dir)

from .hdlm_prompts import (
    wos_subtask_prompt,
    Area_dict,
    Domain_dict,
    esc_subtask_prompt
)


class Depth2_HdLMModel(LlamaPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config, has_subtask, tokenizer_path, think_layer_index=None, 
                 think_loss_weight=1, final_loss_weight=1,record_losses=False):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # output from think layer
        if think_layer_index is not None:
            self.think_layer_index = think_layer_index
            self.new_lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        else:
            self.think_layer_index = None
        self.think_loss_weight = think_loss_weight
        self.final_loss_weight = final_loss_weight
        self.record_losses = record_losses
        self.has_subtask = has_subtask
        if self.record_losses:
            self.final_losses = []
            self.think_losses = []
        
        # get the token sequence of thought and assistant 
        self.thought_token_ids = self.tokenizer.encode("<|start_header_id|>thought<|end_header_id|>\n\n",add_special_tokens=False)
        self.assistant_token_ids = self.tokenizer.encode("<|start_header_id|>assistant<|end_header_id|>\n\n",add_special_tokens=False)
        if self.has_subtask:
            self.subtask_token_ids = self.tokenizer.encode("<|start_header_id|>subtask<|end_header_id|>\n\n",add_special_tokens=False)
        self.end_token_ids = self.tokenizer.encode("<|eot_id|>",add_special_tokens=False)
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
        self.new_lm_head.weight.data = self.lm_head.weight.data.clone()


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
        use_intermediate_head=False, 
         **kwargs,
    ):

        use_intermediate_head = kwargs.get("use_intermediate_head", use_intermediate_head)

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
        
        think_logits = None

        loss = None
        final_loss = None
        think_loss = None

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
            thought_loss_mask = torch.zeros_like(labels, dtype=torch.float32)
            assistant_loss_mask = torch.zeros_like(labels, dtype=torch.float32)

            for batch_idx in range(batch_size):
                thought_start_list = find_subsequence_indices(labels[batch_idx], self.thought_token_ids)
                if self.has_subtask:
                    subtask_start_list = find_subsequence_indices(labels[batch_idx], self.subtask_token_ids)
                # print(thought_start_list)
                assistant_start_list = find_subsequence_indices(labels[batch_idx], self.assistant_token_ids)
                # print(assistant_start_list)
                if len(thought_start_list) > 0 and len(assistant_start_list) > 0 and len(thought_start_list)==len(assistant_start_list): ## 这里可能需要修改，在HTC中，三层分类时应该是两次thounght过程
                    for i in range(0, len(thought_start_list)):
                        # Mask for thought part (think layer)
                        if self.has_subtask:
                            thought_loss_mask[batch_idx, thought_start_list[i] + len(self.thought_token_ids) : subtask_start_list[i]] = 1
                        else:
                            thought_loss_mask[batch_idx, thought_start_list[i] + len(self.thought_token_ids) : assistant_start_list[i]] = 1
                        # Mask for assistant part (final layer)
                        # assistant_loss_mask[batch_idx, assistant_start_list[i] + len(self.assistant_token_ids) : assistant_start_list[i] + len(self.assistant_token_ids) + next_assistant_end + 1] = 1
                        assistant_loss_mask[batch_idx, assistant_start_list[i] + len(self.assistant_token_ids):] = 1

            loss_fct = nn.CrossEntropyLoss()

            if self.think_layer_index is not None:
                intermediate_hidden_states = outputs.hidden_states[self.think_layer_index]
                think_logits = self.new_lm_head(intermediate_hidden_states)
                shift_think_logits = think_logits[..., :-1, :].contiguous()
                intermediate_labels = torch.where(thought_loss_mask > 0, labels, torch.tensor(-100).to(labels.device))
                shift_intermediate_labels = intermediate_labels[..., 1:].contiguous()
                think_loss = loss_fct(shift_think_logits.view(-1, self.config.vocab_size), 
                                        shift_intermediate_labels.view(-1).to(think_logits.device))
                print(f">>> think_loss: {think_loss}")
            shift_logits = final_logits[..., :-1, :].contiguous()
            assistant_labels = torch.where(assistant_loss_mask > 0, labels, torch.tensor(-100).to(labels.device))
            shift_assistant_labels = assistant_labels[..., 1:].contiguous()
            final_loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), 
                                      shift_assistant_labels.view(-1).to(final_logits.device))
            print(f">>> Final_loss: {final_loss}")
            # compute total loss
            if think_loss is not None:
                loss = final_loss * self.final_loss_weight + think_loss * self.think_loss_weight
                if self.record_losses:
                    self.final_losses.append(final_loss.item())
                    self.think_losses.append(think_loss.item())
            else:
                loss = final_loss
                if self.record_losses:
                    self.final_losses.append(final_loss.item())
            print(f"Loss: {loss}")

        if self.think_layer_index is not None:
            think_logits = self.new_lm_head(outputs.hidden_states[self.think_layer_index])
        
        if use_intermediate_head and (self.think_layer_index is not None):
            logits = think_logits.float()
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

            use_intermediate_head = kwargs.get("use_intermediate_head", False)

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
                    "use_intermediate_head": use_intermediate_head, 
                }
            )
            return model_inputs



    def plot_losses(self, fig_save_dir):
        if self.record_losses:
            with open(f"{fig_save_dir}/two_losses.jsonl", "w") as jsonl_file:
                for step, final_loss, think_loss in zip(range(len(self.final_losses)), self.final_losses, self.think_losses):
                    jsonl_file.write(json.dumps({"step": step, "think_losses": think_loss, "final_losses": final_loss}) + "\n")
            plt.figure(figsize=(10, 5))
            
            
            plt.plot(self.final_losses, label="Final Loss", alpha=0.5, color='#92a5d1')
            plt.plot(self.think_losses, label="Think Loss", alpha=0.5, color='#d9b9d4')
            
            
            final_losses_smooth = savgol_filter(self.final_losses, window_length=5, polyorder=2)
            think_losses_smooth = savgol_filter(self.think_losses, window_length=5, polyorder=2)
            plt.plot(self.final_losses, label="Final Loss (Smooth)", alpha=0.3, color='#92a5d1')
            plt.plot(self.think_losses, label="Think Loss (Smooth)", alpha=0.3, color='#d9b9d4')
            
            plt.xlabel("Training Step")
            plt.ylabel("Loss")
            plt.title("Loss Curves")
            plt.legend()
            plt.savefig(f"{fig_save_dir}/think_and_final_losses.png") 
            plt.show()

            
    
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
          dataset: str = "WOS",
          logits_processor = None, 
          **kwargs) -> str:
        
        input_text = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n" \
                     + system_prompt \
                     + "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n" \
                     + query + "<|eot_id|>"
        if think:
            input_text += "<|start_header_id|>thought<|end_header_id|>\n\n"
            input_ids = tokenizer.encode(input_text, return_tensors='pt').to(self.device)
            thought_output = self.generate(
                use_intermediate_head=True,
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
            
            label1 = tokenizer.decode(thought_output[0][len(input_ids[0]):], skip_special_tokens=False).split("<|eot_id|>")[0]
            if dataset == "WOS":
                try:
                    subtask_text = wos_subtask_prompt.format(Area_subdict=Area_dict[Domain_dict[int(label1)]])
                except:
                    print(f"Invalid label1 value: {label1}")
                    return -1, -1
            elif dataset == "ESC":
                subtask_text = esc_subtask_prompt
            elif dataset == "DailyDialog":
                subtask_text = '''As the assistant in this conversation, based on the above information and your chosen strategy, continue to respond to the conversation.\n\nAnswer:\n'''
            elif dataset == "ESC_Emo":
                subtask_text="Based on the identified emotion, continue the conversation to provide support and empathy to the seeker.\n\nAnswer:"
            else:
                subtask_text = ""
            input_text += f"{label1}<|eot_id|><|start_header_id|>subtask<|end_header_id|>\n\n"
            input_text += f"{subtask_text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            input_ids = tokenizer.encode(input_text, return_tensors='pt').to(self.device)
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
            label2 = tokenizer.decode(response_output[0][len(input_ids[0]):], skip_special_tokens=True)
            return label1, label2

        else:

            response_text = "HdLM must think!"
            return response_text