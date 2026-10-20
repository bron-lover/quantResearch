import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen3-0.6B-Base"

# Wnhat are the different types of tokenizers?
tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    # Determines which device the model is loaded onto
    device_map="auto",
)

# Evaluation mode
model.eval()

# Inspect the weight matrices
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear):
        print(name, tuple(module.weight.shape))
