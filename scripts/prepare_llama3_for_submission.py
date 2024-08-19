import torch

from transformers import AutoTokenizer

from human_pref.inference.modeling_llama import LlamaForSequenceClassification


save_path = "../uploads/m3"
checkpoint_path = "../artifacts/stage3/m3/update_last.pth"
model_name_or_path = "RLHFlow/ArmoRM-Llama3-8B-v0.1"

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model = LlamaForSequenceClassification.from_pretrained(
    model_name_or_path,
    torch_dtype=torch.float16,
)
state_dict = torch.load(checkpoint_path, "cpu")["model"]
for idx, layer in enumerate(model.model.layers):
    state_dict[f"model.layers.{idx}.mlp.gate_up_proj.weight"] = torch.cat(
        [
            state_dict[f"model.layers.{idx}.mlp.gate_proj.weight"],
            state_dict[f"model.layers.{idx}.mlp.up_proj.weight"],
        ],
        dim=0,
    )
print(model.load_state_dict(state_dict, strict=False))
tokenizer.save_pretrained(save_path)
model.save_pretrained(save_path)
