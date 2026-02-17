import torch
from torchinfo import summary
import inspect

from examples.dino_clssifier_head_s2_temportal_one_block import load_backbone
from universal_models.multi_sensor_panopticon_adapter import (
    MultiSensorPanopticonClassifier,
    custom_collate_fn,
)

device = torch.device("cpu")
model = MultiSensorPanopticonClassifier(
    backbone=load_backbone(weights_path="none", device=device),
    sensors=("s2", "l89", "s5p"),
).to(device)
model.eval()

backbone=load_backbone(weights_path="none", device=device)

for mname, module in backbone.named_modules():
    # print(pname)
    # print(type(param))
    # print(param.shape)
    if mname == "patch_embed.conv3d":
        print(mname)
        module.__class__
        inspect.signature(module.__class__.__init__)
        

# raw_batch = [
#     ({"imgs": torch.randn(36, 224, 224), "chn_ids": torch.zeros(36, 1)}, 0, "s2"),
#     ({"imgs": torch.randn(36, 224, 224), "chn_ids": torch.zeros(36, 1)}, 1, "l89"),
#     ({"imgs": torch.randn(3, 224, 224), "chn_ids": torch.zeros(3, 1)}, 1, "s5p"),
# ]
# x_dict, labels, sensors = custom_collate_fn(raw_batch)
# sensor_tensor = model.encode_sensors(sensors, device=device)

# summary(
#     model,
#     input_data=(x_dict, sensor_tensor),
#     depth=4,
#     col_names=("input_size", "output_size", "num_params", "mult_adds"),
# )
