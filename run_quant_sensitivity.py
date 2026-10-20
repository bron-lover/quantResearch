"""

Qwen3 block-wise weight quantization sensitivity screening

Runs BF16 reference and one-block-at-a-time simulated symmetric quantization.
Does NOT compress weights or measure INT4 runtime speed --> simulated quantization, not real quantization
The weights still take up 16 bits, we are just putting them down to 4 bit precision

"""

import argparse
import json
import math
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def simulated_quantize(weight: torch.Tensor, bits: int) -> torch.Tensor:
    """Per-tensor symmetric round-to-nearest; returns original floating dtype."""
    w = weight.detach().float()
    qmax = 2 ** (bits - 1) - 1
    scale = w.abs().max() / qmax
    if scale.item() == 0:
        return weight.clone()
    codes = torch.round(w / scale).clamp(-qmax, qmax)
    return (codes * scale).to(weight.dtype)


def evaluate_loss(model, token_ids: torch.Tensor, block_size: int, max_blocks: int):
    """Non-overlapping context chunks; suitable for comparing variants, not standard PPL."""
    total_nll = 0.0
    total_targets = 0
    device = model.get_input_embeddings().weight.device

    with torch.inference_mode():
        for index in range(max_blocks):
            start = index * block_size
            batch = token_ids[:, start : start + block_size]
            if batch.shape[1] < 2:
                break
            batch = batch.to(device)
            result = model(input_ids=batch, labels=batch, use_cache=False)
            n_targets = batch.shape[1] - 1  # causal LM loss ignores first token
            total_nll += float(result.loss.item()) * n_targets
            total_targets += n_targets

    if not total_targets:
        raise ValueError("The dataset is too short to evaluate.")
    loss = total_nll / total_targets
    return loss, math.exp(loss), total_targets


def capture_block_outputs(model, input_ids: torch.Tensor):
    """Use hooks to capture exactly the output of each transformer block."""
    blocks = model.model.layers
    hidden = [None] * len(blocks)
    handles = []

    def make_hook(index):
        def record(_module, _inputs, output):
            states = output[0] if isinstance(output, (tuple, list)) else output
            hidden[index] = states.detach().float().cpu()

        return record

    try:
        for i, block in enumerate(blocks):
            handles.append(block.register_forward_hook(make_hook(i)))
        with torch.inference_mode():
            model(input_ids=input_ids, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()

    assert all(h is not None for h in hidden)
    return hidden


def relative_errors(reference, modified):
    errors = []
    for a, b in zip(reference, modified):
        err = torch.linalg.vector_norm(b - a)
        denom = torch.linalg.vector_norm(a).clamp_min(1e-12)
        errors.append(float((err / denom).item()))
    return errors


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B-Base")
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument(
        "--block",
        type=int,
        default=None,
        help="Specific block index. Omit to sweep all blocks.",
    )
    parser.add_argument(
        "--target",
        default="all",
        help="all, or a relative matrix name like mlp.down_proj",
    )
    parser.add_argument(
        "--baseline-only",
        action="store_true",
        help="Evaluate untouched BF16 model and exit",
    )
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--max-blocks", type=int, default=8)
    parser.add_argument("--probe-length", type=int, default=128)
    args = parser.parse_args()
    if not 2 <= args.bits <= 16:
        parser.error("--bits must be between 2 and 16")
    if not torch.cuda.is_available():
        parser.error("This example requires a CUDA GPU; run it on your cluster.")
    torch.manual_seed(42)
    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.model}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map="cuda:0",
    )
    model.eval()

    dataset = load_dataset(
        "Salesforce/wikitext", "wikitext-2-raw-v1", split="validation"
    )
    text = "\n\n".join(row for row in dataset["text"] if row.strip())
    token_ids = tokenizer(text, return_tensors="pt", add_special_tokens=False)[
        "input_ids"
    ]
    if token_ids.shape[1] < 2:
        raise ValueError("Not enough tokens to evaluate.")

    baseline_loss, baseline_ppl, n_targets = evaluate_loss(
        model, token_ids, args.block_size, args.max_blocks
    )
    print(
        f"Reference BF16 loss={baseline_loss:.5f}; PPL={baseline_ppl:.3f}; "
        f"scored tokens={n_targets}",
        flush=True,
    )
    (out_dir / "baseline.json").write_text(
        json.dumps(
            {
                "model": args.model,
                "dataset": "Salesforce/wikitext: wikitext-2-raw-v1 validation",
                "block_size": args.block_size,
                "n_scored_tokens": n_targets,
                "loss": baseline_loss,
                "perplexity": baseline_ppl,
            },
            indent=2,
        )
    )
    if args.baseline_only:
        return

    probe = token_ids[:, : args.probe_length].to(
        model.get_input_embeddings().weight.device
    )
    reference_hidden = capture_block_outputs(model, probe)

    blocks = model.model.layers
    indices = range(len(blocks)) if args.block is None else [args.block]
    for index in indices:
        if not 0 <= index < len(blocks):
            raise ValueError(
                f"Invalid block {index}; valid range is 0..{len(blocks)-1}"
            )
        block = blocks[index]
        # BF16 originals exist only for this block; 1 GPU and no second model copy.
        original = {
            name: module.weight.detach().clone()
            for name, module in block.named_modules()
            if isinstance(module, torch.nn.Linear)
            and (args.target == "all" or name == args.target)
        }
        if not original:
            raise ValueError(f"No Linear matrix named {args.target!r} in block {index}")
        try:
            with torch.no_grad():
                for name, module in block.named_modules():
                    if name in original:
                        module.weight.copy_(
                            simulated_quantize(original[name], args.bits)
                        )

            new_loss, new_ppl, _ = evaluate_loss(
                model, token_ids, args.block_size, args.max_blocks
            )
            quant_hidden = capture_block_outputs(model, probe)
            errors = relative_errors(reference_hidden, quant_hidden)
            result = {
                "model": args.model,
                "block": index,
                "target": args.target,
                "bits": args.bits,
                "quantization": "symmetric per-tensor RTN fake quantization, linear weights only",
                "dataset": "Salesforce/wikitext: wikitext-2-raw-v1 validation",
                "block_size": args.block_size,
                "n_scored_tokens": n_targets,
                "reference_loss": baseline_loss,
                "reference_ppl": baseline_ppl,
                "quant_loss": new_loss,
                "quant_ppl": new_ppl,
                "delta_loss": new_loss - baseline_loss,
                "relative_block_output_errors": errors,
            }
            safe_target = args.target.replace(".", "-")
            output_path = out_dir / f"block_{index:02d}_{safe_target}_w{args.bits}.json"
            output_path.write_text(json.dumps(result, indent=2))
            print(
                f"Block {index:02d} | delta loss={new_loss - baseline_loss:+.5f} "
                f"| immediate state error={errors[index]:.5f} "
                f"| final block error={errors[-1]:.5f}",
                flush=True,
            )
        finally:
            with torch.no_grad():
                for name, module in block.named_modules():
                    if name in original:
                        module.weight.copy_(original[name])


if __name__ == "__main__":
    run()
