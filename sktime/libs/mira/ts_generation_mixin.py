# ruff: noqa
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import warnings
from typing import Any, Dict, List, Optional, Union

from sktime.utils.dependencies import _safe_import

torch = _safe_import("torch")

GenerationMixin = _safe_import("transformers.GenerationMixin")
LogitsProcessorList = _safe_import(
    "transformers.generation.logits_process.LogitsProcessorList"
)
StoppingCriteriaList = _safe_import(
    "transformers.generation.stopping_criteria.StoppingCriteriaList"
)
validate_stopping_criteria = _safe_import(
    "transformers.generation.stopping_criteria.validate_stopping_criteria"
)
EosTokenCriteria = _safe_import(
    "transformers.generation.stopping_criteria.EosTokenCriteria"
)
GenerateNonBeamOutput = _safe_import(
    "transformers.generation.utils.GenerateNonBeamOutput"
)
GenerateEncoderDecoderOutput = _safe_import(
    "transformers.generation.utils.GenerateEncoderDecoderOutput"
)
GenerateDecoderOnlyOutput = _safe_import(
    "transformers.generation.utils.GenerateDecoderOnlyOutput"
)
ModelOutput = _safe_import("transformers.utils.ModelOutput")
BaseStreamer = _safe_import("transformers.generation.streaming_utils.BaseStreamer")


class MIRAGenerationMixin(GenerationMixin):
    """
    Please note that the current version does not support inference with key-value caching.
    The following code will update soon.
    """

    def _greedy_search(
        self,
        input_ids: torch.Tensor,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        output_logits: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        synced_gpus: bool = False,
        streamer: Optional["BaseStreamer"] = None,
        **model_kwargs,
    ) -> Union[GenerateNonBeamOutput, torch.Tensor]:
        input_ids_origin_device = input_ids.device
        input_ids = input_ids.to(self.device)

        if len(input_ids.shape) == 2:
            batch_size, cur_len = input_ids.shape
            # 如果是 2D，添加 input_size 维度
            input_ids = input_ids.unsqueeze(-1)
        elif len(input_ids.shape) == 3:
            batch_size, cur_len, _ = input_ids.shape
        else:
            raise ValueError(
                f"Input shape must be [batch_size, seq_len] or [batch_size, seq_len, input_size], got {input_ids.shape}"
            )

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
                " `stopping_criteria=StoppingCriteriaList([MaxLengthCriteria(max_length=max_length)])` instead.",
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

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
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

        max_length = stopping_criteria.max_length

        if "time_values" not in model_kwargs or model_kwargs["time_values"] is None:
            # 创建默认时间序列 [0, 1, 2, ..., cur_len-1]
            model_kwargs["time_values"] = (
                torch.arange(cur_len, dtype=torch.float32, device=input_ids.device)
                .unsqueeze(0)
                .expand(batch_size, -1)
            )
            warnings.warn(
                "time_values not provided, using default sequential time [0, 1, 2, ...]"
            )

        if model_kwargs["time_values"].shape[1] > 1:
            time_diffs = (
                model_kwargs["time_values"][:, 1:] - model_kwargs["time_values"][:, :-1]
            )
            self._internal_time_step = time_diffs.mean(dim=1, keepdim=True)  # [B, 1]
        else:
            self._internal_time_step = torch.ones(
                batch_size, 1, device=input_ids.device
            )

        self._last_time_values = model_kwargs["time_values"][:, -1:].clone()

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
                max_horizon_length=max_length - input_length,
            )

            if synced_gpus and this_peer_finished:
                continue  # don't waste resources running the code we don't need

            next_token_logits = outputs.logits[:, -1, :]

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

            # argmax (time-series uses the prediction directly)
            next_tokens = next_tokens_scores

            # finished sentences should have their next token be a padding token
            if eos_token_id is not None:
                if pad_token_id is None:
                    raise ValueError(
                        "If `eos_token_id` is defined, make sure that `pad_token_id` is defined."
                    )
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (
                    1 - unfinished_sequences
                )

            # update generated ids, model inputs, and length for next step
            next_tokens = next_tokens.reshape(batch_size, -1, self.config.input_size)
            horizon_length = next_tokens.shape[1]

            input_ids = torch.cat([input_ids, next_tokens], dim=-2)

            if streamer is not None:
                streamer.put(next_tokens.cpu())

            model_kwargs = self._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                horizon_length=horizon_length,
                is_encoder_decoder=self.config.is_encoder_decoder,
            )

            unfinished_sequences = unfinished_sequences & ~stopping_criteria(
                input_ids[..., 0], scores
            )
            this_peer_finished = unfinished_sequences.max() == 0

        if input_ids.shape[1] > max_length:
            input_ids = input_ids[:, :max_length]

        if streamer is not None:
            streamer.end()

        input_ids = input_ids.squeeze(dim=-1).to(input_ids_origin_device)

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
            return input_ids

    def _update_model_kwargs_for_generation(
        self,
        outputs: ModelOutput,
        model_kwargs: Dict[str, Any],
        horizon_length: int = 1,
        is_encoder_decoder: bool = False,
        standardize_cache_format: bool = False,
    ) -> Dict[str, Any]:
        model_kwargs["past_key_values"] = self._extract_past_from_model_output(
            outputs, standardize_cache_format=standardize_cache_format
        )

        if getattr(outputs, "state", None) is not None:
            model_kwargs["state"] = outputs.state

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

        if "time_values" in model_kwargs and model_kwargs["time_values"] is not None:
            current_time_values = model_kwargs["time_values"]

            if hasattr(self, "_internal_time_step"):
                time_step = self._internal_time_step
            else:
                if current_time_values.shape[1] > 1:
                    time_diffs = (
                        current_time_values[:, 1:] - current_time_values[:, :-1]
                    )
                    time_step = time_diffs.mean(dim=1, keepdim=True)
                else:
                    time_step = torch.ones(
                        current_time_values.shape[0],
                        1,
                        device=current_time_values.device,
                    )

            last_time = current_time_values[:, -1:]  # [B, 1]
            new_times = []
            for i in range(1, horizon_length + 1):
                new_times.append(last_time + time_step * i)

            if new_times:
                new_time_values = torch.cat(new_times, dim=1)  # [B, horizon_length]
                model_kwargs["time_values"] = torch.cat(
                    [current_time_values, new_time_values], dim=1
                )

            self._last_time_values = model_kwargs["time_values"][:, -1:].clone()

        if "time_values" in model_kwargs and model_kwargs["time_values"] is not None:
            last_time = model_kwargs["time_values"][:, -1:]  # [B, 1]

            if hasattr(self, "_internal_time_step"):
                time_step = self._internal_time_step
            else:
                time_step = torch.ones_like(last_time)

            model_kwargs["next_target_time_values"] = last_time + time_step

        if (
            "cache_position" in model_kwargs
            and model_kwargs["cache_position"] is not None
        ):
            model_kwargs["cache_position"] = (
                model_kwargs["cache_position"][-1:] + horizon_length
            )

        return model_kwargs
