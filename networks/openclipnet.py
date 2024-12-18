'''                                        
Copyright 2024 Image Processing Research Group of University Federico
II of Naples ('GRIP-UNINA'). All rights reserved.
                        
Licensed under the Apache License, Version 2.0 (the "License");       
you may not use this file except in compliance with the License. 
You may obtain a copy of the License at                    
                                           
    http://www.apache.org/licenses/LICENSE-2.0
                                                      
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,    
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.                         
See the License for the specific language governing permissions and
limitations under the License.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip
from .resnet_mod import ChannelLinear

dict_pretrain = {
    'clipL14openai'     : ('ViT-L-14', 'openai'),
    'clipL14laion400m'  : ('ViT-L-14', 'laion400m_e32'),
    'clipL14laion2B'    : ('ViT-L-14', 'laion2b_s32b_b82k'),
    'clipL14datacomp'   : ('ViT-L-14', 'laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K', 'open_clip_pytorch_model.bin'),
    'clipL14commonpool' : ('ViT-L-14', "laion/CLIP-ViT-L-14-CommonPool.XL-s13B-b90K", 'open_clip_pytorch_model.bin'),
    'clipaL14datacomp'  : ('ViT-L-14-CLIPA', 'datacomp1b'),
    'cocaL14laion2B'    : ('coca_ViT-L-14', 'laion2b_s13b_b90k'),
    'clipg14laion2B'    : ('ViT-g-14', 'laion2b_s34b_b88k'),
    'eva2L14merged2b'   : ('EVA02-L-14', 'merged2b_s4b_b131k'),
    'clipB16laion2B'    : ('ViT-B-16', 'laion2b_s34b_b88k'),
}

import torch
import torch.nn.functional as F

class DifferentiableClipTransform(nn.Module):
    def __init__(self):
        super().__init__()
        self.resize_size = 224  # Resize target size
        self.crop_size = 224    # Center crop size
        self.mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)

    def forward(self, x):
        # Resize using bilinear interpolation
        x = F.interpolate(x, size=self.resize_size, mode='bicubic', align_corners=False)
        
        # Center crop
        h, w = x.shape[2], x.shape[3]
        top = (h - self.crop_size) // 2
        left = (w - self.crop_size) // 2
        x = x[:, :, top:top + self.crop_size, left:left + self.crop_size]
        
        # Normalize
        x = (x - self.mean.to(x.device)) / self.std.to(x.device)
        return x


class OpenClipLinear(nn.Module):
    def __init__(self, num_classes=1, pretrain='clipL14commonpool', normalize=True, next_to_last=False):
        super(OpenClipLinear, self).__init__()
        
        if len(dict_pretrain[pretrain])==2:
            backbone = open_clip.create_model(dict_pretrain[pretrain][0], pretrained=dict_pretrain[pretrain][1])
        else:
            from huggingface_hub import hf_hub_download
            backbone = open_clip.create_model(dict_pretrain[pretrain][0], pretrained=hf_hub_download(*dict_pretrain[pretrain][1:]))
        
        if next_to_last:
            self.num_features = backbone.visual.proj.shape[0]
            backbone.visual.proj = None
        else:
            self.num_features = backbone.visual.output_dim

        # self.transform = DifferentiableClipTransform()
        self.bb = [backbone, ]
        self.normalize = normalize
        
        self.fc = ChannelLinear(self.num_features, num_classes)
        torch.nn.init.normal_(self.fc.weight.data, 0.0, 0.02)

    def to(self, *args, **kwargs):
        self.bb[0].to(*args, **kwargs)
        super(OpenClipLinear, self).to(*args, **kwargs)
        return self

    def forward_features(self, x):
        # with torch.no_grad():
        self.bb[0].eval()
        features = self.bb[0].encode_image(x, normalize=self.normalize)
        return features

    def forward_head(self, x):
        return self.fc(x)

    def forward(self, x):
        # x = self.transform(x)
        return self.forward_head(self.forward_features(x))
