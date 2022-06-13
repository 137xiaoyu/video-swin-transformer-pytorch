import torch
import torch.nn as nn
from collections import OrderedDict
from video_swin_transformer import SwinTransformer3D
'''
initialize a SwinTransformer3D model
'''
model = SwinTransformer3D(patch_size=(2, 4, 4),
                          embed_dim=128,
                          depths=[2, 2, 18, 2],
                          num_heads=[4, 8, 16, 32],
                          window_size=(8, 7, 7),
                          mlp_ratio=4.,
                          qkv_bias=True,
                          qk_scale=None,
                          drop_rate=0.,
                          attn_drop_rate=0.,
                          drop_path_rate=0.2,
                          patch_norm=True)
print(model)

# https://github.com/SwinTransformer/storage/releases/download/v1.0.4/swin_base_patch244_window877_kinetics400_22k.pth
checkpoint = torch.load('swin_base_patch244_window877_kinetics400_22k.pth')

new_state_dict = OrderedDict()
for k, v in checkpoint['state_dict'].items():
    if 'backbone' in k:
        name = k[9:]
        new_state_dict[name] = v
    elif 'cls_head' in k:
        name = k[9:]
        new_state_dict[name] = v

model.load_state_dict(new_state_dict)

dummy_x = torch.rand(1, 3, 16, 224, 224)
feat, avg_feat, logit = model(dummy_x)
print(feat.shape)
print(avg_feat.shape)
print(logit.shape)
