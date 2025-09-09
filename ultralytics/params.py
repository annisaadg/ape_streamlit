import torch
import torch.nn as nn
from nn.modules.block import C2PSA  # Ganti dengan path sebenarnya jika perlu

# Fungsi untuk ambil atribut penting
def get_attrs(obj, attr_names):
    results = {}
    for name in attr_names:
        try:
            value = getattr(obj, name)
            results[name] = value
        except AttributeError:
            continue
    return results

# Fungsi utama untuk tampilkan parameter & argumen
def print_params_and_args(module, indent=0):
    for name, child in module.named_children():
        num_params = sum(p.numel() for p in child.parameters())
        trainable = sum(p.numel() for p in child.parameters() if p.requires_grad)
        indent_str = "  " * indent

        # Argumen penting untuk modul C2PSS dan FRM
        important_attrs = [
            "c", "cv1", "cv2", "m", "attn", "ffn", "add", "num_heads", "attn_ratio",  # C2PSS
            "local_conv", "ca_conv1", "ca_conv2", "sa_conv",                         # FRM
        ]
        attrs = get_attrs(child, important_attrs)
        attr_str = ", ".join(f"{k}={v}" for k, v in attrs.items())

        print(f"{indent_str}{name} ({child.__class__.__name__}): "
              f"total={num_params}, trainable={trainable}"
              + (f", args=({attr_str})" if attr_str else ""))

        print_params_and_args(child, indent + 1)

# ====== UJI DENGAN FRM ======
model = C2PSA(c=1024)

# Tampilkan total parameter
print(f"\nTotal params: {sum(p.numel() for p in model.parameters())}")
print(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}\n")

# Detail per layer dengan argumen
print("Parameter breakdown with args:")
print_params_and_args(model)
