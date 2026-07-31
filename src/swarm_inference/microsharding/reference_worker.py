"""Independent-process unsharded Qwen3 reference for Experiment 006."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch


def _cache_tensors(cache: Any, layer_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    if hasattr(cache, "layers"):
        layer = cache.layers[layer_id]
        for key_name, value_name in (("keys", "values"), ("key_cache", "value_cache")):
            if hasattr(layer, key_name) and hasattr(layer, value_name):
                return getattr(layer, key_name), getattr(layer, value_name)
    if hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
        return cache.key_cache[layer_id], cache.value_cache[layer_id]
    try:
        key, value = cache[layer_id][:2]
    except (TypeError, IndexError, AttributeError) as exc:
        raise RuntimeError("unsupported Transformers KV-cache representation") from exc
    return key, value


def run_reference(request_path: Path, output_path: Path, boundary_path: Path) -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    request = json.loads(request_path.read_text(encoding="utf-8"))
    model_path = str(request["model_path"])
    tokenizer_factory: Any = AutoTokenizer
    model_factory: Any = AutoModelForCausalLM
    tokenizer = tokenizer_factory.from_pretrained(model_path, local_files_only=True)
    model = model_factory.from_pretrained(
        model_path,
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation="eager",
    ).to("cuda")
    model.eval()
    prompt_results: list[dict[str, Any]] = []
    with torch.inference_mode():
        for prompt in request["prompts"]:
            encoded = tokenizer(str(prompt["text"]), return_tensors="pt")
            input_ids = encoded.input_ids.to("cuda")
            attention_mask = encoded.attention_mask.to("cuda")
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=int(request["max_new_tokens"]),
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
            prompt_results.append(
                {
                    "prompt_id": prompt["prompt_id"],
                    "input_token_count": int(input_ids.shape[1]),
                    "input_ids": input_ids[0].cpu().tolist(),
                    "generated_token_ids": generated[0, input_ids.shape[1] :].cpu().tolist(),
                }
            )

        selected_layers = [int(item) for item in request["selected_layers"]]
        capture: dict[int, dict[str, torch.Tensor]] = {layer: {} for layer in selected_layers}
        handles: list[Any] = []
        for layer_id in selected_layers:
            layer = model.model.layers[layer_id]

            def layer_pre_hook(
                _module: torch.nn.Module,
                args: tuple[Any, ...],
                kwargs: dict[str, Any],
                *,
                selected: int = layer_id,
            ) -> None:
                hidden = kwargs.get("hidden_states", args[0] if args else None)
                if hidden is None:
                    raise RuntimeError("reference layer hook did not receive hidden states")
                capture[selected]["layer_input"] = hidden.detach().cpu()

            def attention_hook(
                _module: torch.nn.Module,
                _args: tuple[Any, ...],
                output: Any,
                *,
                selected: int = layer_id,
            ) -> None:
                value = output[0] if isinstance(output, tuple) else output
                capture[selected]["attention_output"] = value.detach().cpu()

            def mlp_hook(
                _module: torch.nn.Module,
                _args: tuple[Any, ...],
                output: torch.Tensor,
                *,
                selected: int = layer_id,
            ) -> None:
                capture[selected]["mlp_output"] = output.detach().cpu()

            def layer_hook(
                _module: torch.nn.Module,
                _args: tuple[Any, ...],
                output: Any,
                *,
                selected: int = layer_id,
            ) -> None:
                value = output[0] if isinstance(output, tuple) else output
                capture[selected]["final_hidden"] = value.detach().cpu()

            handles.extend(
                [
                    layer.register_forward_pre_hook(layer_pre_hook, with_kwargs=True),
                    layer.self_attn.register_forward_hook(attention_hook),
                    layer.mlp.register_forward_hook(mlp_hook),
                    layer.register_forward_hook(layer_hook),
                ]
            )
        boundary_prompt = next(
            item
            for item in request["prompts"]
            if item["prompt_id"] == request["boundary_prompt_id"]
        )
        boundary_encoded = tokenizer(str(boundary_prompt["text"]), return_tensors="pt")
        boundary_ids = boundary_encoded.input_ids.to("cuda")
        boundary_attention = boundary_encoded.attention_mask.to("cuda")
        boundary_output = model(
            input_ids=boundary_ids,
            attention_mask=boundary_attention,
            use_cache=True,
            output_hidden_states=True,
        )
        for handle in handles:
            handle.remove()
        tensor_payload: dict[str, torch.Tensor] = {
            "input_ids": boundary_ids.cpu(),
            "final_normalised_hidden": boundary_output.hidden_states[-1].detach().cpu(),
            "logits": boundary_output.logits.detach().cpu(),
        }
        for layer_id in selected_layers:
            layer_capture = capture[layer_id]
            layer_capture["post_attention_hidden"] = (
                layer_capture["layer_input"] + layer_capture["attention_output"]
            )
            key, value = _cache_tensors(boundary_output.past_key_values, layer_id)
            layer_capture["key_cache"] = key.detach().cpu()
            layer_capture["value_cache"] = value.detach().cpu()
            for name, tensor in layer_capture.items():
                tensor_payload[f"layer_{layer_id}_{name}"] = tensor
        torch.save(tensor_payload, boundary_path)
    output = {
        "classification": "logical_microsharding_correctness",
        "reference_process": "independent_unsharded_transformers_process",
        "physical_process_count": 1,
        "cuda_context_count": 1,
        "model_id": request["model_id"],
        "model_revision": request["model_revision"],
        "attention_implementation": "eager",
        "dtype": "bfloat16",
        "prompts": prompt_results,
        "boundary_tensor_file": str(boundary_path),
    }
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--boundaries", type=Path, required=True)
    arguments = parser.parse_args()
    run_reference(arguments.request, arguments.output, arguments.boundaries)


if __name__ == "__main__":
    main()
