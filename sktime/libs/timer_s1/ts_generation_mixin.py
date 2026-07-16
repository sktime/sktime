# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http:www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Generation utilities for TimerS1 time-series outputs."""

import warnings
from collections.abc import Callable
from typing import Any, Optional

from sktime.utils.dependencies import _safe_import

torch = _safe_import("torch")
GenerationMixin = _safe_import("transformers.GenerationMixin")
LogitsProcessorList = _safe_import("transformers.LogitsProcessorList")
StoppingCriteriaList = _safe_import("transformers.StoppingCriteriaList")
PreTrainedModel = _safe_import("transformers.PreTrainedModel")
BaseStreamer = _safe_import("transformers.generation.streamers.BaseStreamer")
EosTokenCriteria = _safe_import("transformers.generation.EosTokenCriteria")
validate_stopping_criteria = _safe_import(
    "transformers.generation.validate_stopping_criteria"
)
GenerateDecoderOnlyOutput = _safe_import(
    "transformers.generation.utils.GenerateDecoderOnlyOutput"
)
GenerateEncoderDecoderOutput = _safe_import(
    "transformers.generation.utils.GenerateEncoderDecoderOutput"
)
GenerateNonBeamOutput = _safe_import(
    "transformers.generation.utils.GenerateNonBeamOutput"
)
GenerateOutput = _safe_import("transformers.generation.utils.GenerateOutput")
GenerationConfig = _safe_import("transformers.generation.utils.GenerationConfig")
ModelOutput = _safe_import("transformers.utils.ModelOutput")

ALL_CACHE_NAMES = [
    "past_key_values",  # default
    "cache_params",  # mamba-based models
    "state",  # rwkv
    "mems",  # xlnet
    "past_buckets_states",  # reformer
]


class TSGenerationMixin(GenerationMixin):
    """Generation mixin customized for time-series forecasting models."""

    @torch.no_grad()
    def generate(
        self,
        inputs: torch.Tensor | None = None,
        generation_config: GenerationConfig | None = None,
        logits_processor: LogitsProcessorList | None = None,
        stopping_criteria: StoppingCriteriaList | None = None,
        prefix_allowed_tokens_fn: Callable[[int, torch.Tensor], list[int]]
        | None = None,
        synced_gpus: bool | None = None,
        assistant_model: Optional["PreTrainedModel"] = None,
        streamer: Optional["BaseStreamer"] = None,
        negative_prompt_ids: torch.Tensor | None = None,
        negative_prompt_attention_mask: torch.Tensor | None = None,
        revin: bool | None = True,
        **kwargs,
    ) -> GenerateOutput | torch.LongTensor:
        """Generate forecast samples from normalized time-series inputs."""
        if len(inputs.shape) != 2:
            raise ValueError("Input shape must be: [batch_size, seq_len]")
        if revin:
            means = inputs.mean(dim=-1, keepdim=True)
            stdev = inputs.std(dim=-1, keepdim=True, unbiased=False) + 1e-5
            inputs = (inputs - means) / stdev
        outputs = super().generate(
            inputs=inputs,
            generation_config=generation_config,
            logits_processor=logits_processor,
            stopping_criteria=stopping_criteria,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            synced_gpus=synced_gpus,
            assistant_model=assistant_model,
            streamer=streamer,
            negative_prompt_ids=negative_prompt_ids,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            **kwargs,
        )
        if revin:
            stdev = stdev.unsqueeze(1)
            means = means.unsqueeze(1)
            outputs = (outputs * stdev) + means
        return outputs

    def _sample(
        self,
        input_ids: torch.Tensor,
        logits_processor: LogitsProcessorList | None = None,
        stopping_criteria: StoppingCriteriaList | None = None,
        max_length: int | None = None,
        pad_token_id: int | None = None,
        eos_token_id: int | list[int] | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        output_scores: bool | None = None,
        output_logits: bool | None = None,
        return_dict_in_generate: bool | None = None,
        synced_gpus: bool = False,
        streamer: Optional["BaseStreamer"] = None,
        **model_kwargs,
    ) -> GenerateNonBeamOutput | torch.Tensor:
        input_ids = input_ids.to(self.device)
        batch_size, cur_len = input_ids.shape
        # init values
        logits_processor = (
            logits_processor if logits_processor is not None else LogitsProcessorList()
        )
        stopping_criteria = (
            stopping_criteria
            if stopping_criteria is not None
            else StoppingCriteriaList()
        )
        if max_length is not None:
            warnings.warn(
                "`max_length` is deprecated in this function, use"
                " `stopping_criteria=StoppingCriteriaList(["
                "MaxLengthCriteria(max_length=max_length)])` instead.",
                UserWarning,
            )
            stopping_criteria = validate_stopping_criteria(
                stopping_criteria, max_length
            )
        pad_token_id = (
            pad_token_id
            if pad_token_id is not None
            else self.generation_config.pad_token_id
        )
        if eos_token_id is not None:
            stopping_criteria.append(EosTokenCriteria(eos_token_id=eos_token_id))
        else:
            # Add eos criteria so generation does not continue indefinitely.
            eos_token_id = [
                criteria.eos_token_id.tolist()
                for criteria in stopping_criteria
                if hasattr(criteria, "eos_token_id")
            ]
            eos_token_id = eos_token_id[0] if eos_token_id else None
            if eos_token_id is None and self.generation_config.eos_token_id is not None:
                eos_token_id = self.generation_config.eos_token_id
                stopping_criteria.append(EosTokenCriteria(eos_token_id=eos_token_id))

        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        output_scores = (
            output_scores
            if output_scores is not None
            else self.generation_config.output_scores
        )
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.generation_config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.generation_config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate
            if return_dict_in_generate is not None
            else self.generation_config.return_dict_in_generate
        )

        # init attention / hidden states / scores tuples
        raw_logits = () if (return_dict_in_generate and output_logits) else None
        scores = () if (return_dict_in_generate and output_scores) else None
        decoder_attentions = (
            () if (return_dict_in_generate and output_attentions) else None
        )
        cross_attentions = (
            () if (return_dict_in_generate and output_attentions) else None
        )
        decoder_hidden_states = (
            () if (return_dict_in_generate and output_hidden_states) else None
        )

        # Retrieve encoder attention weights and hidden states.
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = (
                model_kwargs["encoder_outputs"].get("attentions")
                if output_attentions
                else None
            )
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states")
                if output_hidden_states
                else None
            )

        # keep track of which sequences are already finished
        if "inputs_embeds" in model_kwargs:
            cur_len = model_kwargs["inputs_embeds"].shape[1]
        this_peer_finished = False
        unfinished_sequences = torch.ones(
            batch_size, dtype=torch.long, device=input_ids.device
        )
        model_kwargs["cache_position"] = torch.arange(cur_len, device=input_ids.device)
        true_seq_len = (
            cur_len + self.config.input_token_len - 1
        ) // self.config.input_token_len
        model_kwargs["attention_mask"] = model_kwargs["attention_mask"][
            :, -true_seq_len:
        ]
        max_length = stopping_criteria.max_length

        generate_results = None
        while self._has_unfinished_sequences(
            this_peer_finished, synced_gpus, device=input_ids.device
        ):
            # prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            input_length = input_ids.shape[1]

            # forward pass to get next token
            outputs = self(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                max_output_length=max_length - input_length,
            )

            if synced_gpus and this_peer_finished:
                continue  # don't waste resources running the code we don't need
            next_token_logits = outputs.logits

            # pre-process distribution
            next_tokens_scores = logits_processor(input_ids, next_token_logits)

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_tokens_scores,)
                if output_logits:
                    raw_logits += (next_token_logits,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,)
                        if self.config.is_encoder_decoder
                        else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # argmax
            # next_tokens = torch.argmax(next_tokens_scores, dim=-1)
            next_tokens = next_tokens_scores

            # finished sentences should have their next token be a padding token
            if eos_token_id is not None:
                if pad_token_id is None:
                    raise ValueError(
                        "If `eos_token_id` is defined, make sure that "
                        "`pad_token_id` is defined."
                    )
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (
                    1 - unfinished_sequences
                )

            # update generated ids, model inputs, and length for next step
            horizon_length = next_tokens.shape[-1] // self.config.input_token_len

            if generate_results is None:
                generate_results = next_tokens
            else:
                generate_results = torch.cat([generate_results, next_tokens], dim=-1)

            # Use quantile instead of median for deterministic CUDA behavior.

            selected_tokens = torch.quantile(next_tokens.float(), q=0.5, dim=1)
            input_ids = torch.cat([input_ids, selected_tokens], dim=-1)

            if streamer is not None:
                streamer.put(next_tokens.cpu())
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                horizon_length=horizon_length,
                is_encoder_decoder=self.config.is_encoder_decoder,
            )
            unfinished_sequences = unfinished_sequences & ~stopping_criteria(
                input_ids, scores
            )
            this_peer_finished = unfinished_sequences.max() == 0

        if input_ids.shape[-1] > max_length:
            input_ids = input_ids[:, :max_length]

        if streamer is not None:
            streamer.end()

        if return_dict_in_generate:
            if self.config.is_encoder_decoder:
                return GenerateEncoderDecoderOutput(
                    sequences=input_ids,
                    scores=scores,
                    logits=raw_logits,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
            else:
                return GenerateDecoderOnlyOutput(
                    sequences=input_ids,
                    scores=scores,
                    logits=raw_logits,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
        else:
            return generate_results[:, :, : (max_length - cur_len)]

    def _update_model_kwargs_for_generation(
        self,
        outputs: ModelOutput,
        model_kwargs: dict[str, Any],
        horizon_length: int = 1,
        is_encoder_decoder: bool = False,
        standardize_cache_format: bool = False,
    ) -> dict[str, Any]:
        # update past_key_values
        for possible_cache_name in ALL_CACHE_NAMES:
            if possible_cache_name in outputs:
                if possible_cache_name in ("past_buckets_states", "mems"):
                    cache_name = "past_key_values"
                else:
                    cache_name = possible_cache_name
                model_kwargs[cache_name] = getattr(outputs, possible_cache_name)
                break

        # update token_type_ids with last value
        if "token_type_ids" in model_kwargs:
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = torch.cat(
                [token_type_ids, token_type_ids[:, -1].unsqueeze(-1)], dim=-1
            )

        if not is_encoder_decoder:
            # update attention mask
            if "attention_mask" in model_kwargs:
                attention_mask = model_kwargs["attention_mask"]
                model_kwargs["attention_mask"] = torch.cat(
                    [
                        attention_mask,
                        attention_mask.new_ones(
                            (attention_mask.shape[0], horizon_length)
                        ),
                    ],
                    dim=-1,
                )
        else:
            # update decoder attention mask
            if "decoder_attention_mask" in model_kwargs:
                decoder_attention_mask = model_kwargs["decoder_attention_mask"]
                model_kwargs["decoder_attention_mask"] = torch.cat(
                    [
                        decoder_attention_mask,
                        decoder_attention_mask.new_ones(
                            (decoder_attention_mask.shape[0], horizon_length)
                        ),
                    ],
                    dim=-1,
                )

        if (
            "cache_position" in model_kwargs
            and model_kwargs["cache_position"] is not None
        ):
            model_kwargs["cache_position"] = (
                model_kwargs["cache_position"][-1:] + horizon_length
            )

        # Accumulate hidden states across generation steps for MTP layers.
        if (
            hasattr(outputs, "hidden_states_for_mtp")
            and outputs.hidden_states_for_mtp is not None
        ):
            new_hs = outputs.hidden_states_for_mtp
            if (
                "full_hidden_states" in model_kwargs
                and model_kwargs["full_hidden_states"] is not None
            ):
                existing = model_kwargs["full_hidden_states"]
                model_kwargs["full_hidden_states"] = torch.cat(
                    [existing.to(new_hs.device), new_hs], dim=1
                )
            else:
                model_kwargs["full_hidden_states"] = new_hs

        return model_kwargs
