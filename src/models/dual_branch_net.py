import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Dict, Tuple, Optional

class FeatureExtractor(nn.Module):
    
    def __init__(self, backbone_name: str = "resnet34", pretrained: bool = True):
        super(FeatureExtractor, self).__init__()
        
        if backbone_name == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            resnet = models.resnet18(weights=weights)
            self.out_channels = 512
        else:
            weights = models.ResNet34_Weights.DEFAULT if pretrained else None
            resnet = models.resnet34(weights=weights)
            self.out_channels = 512

        # Break down ResNet layers to extract intermediate spatial maps
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1  # 64 channels
        self.layer2 = resnet.layer2  # 128 channels
        self.layer3 = resnet.layer3  # 256 channels
        self.layer4 = resnet.layer4  # 512 channels

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        feat_map = self.layer4(x)  # (B, 512, H/32, W/32)

        feat_vec = self.avgpool(feat_map)
        feat_vec = torch.flatten(feat_vec, 1)  # (B, 512)
        return feat_map, feat_vec


class CrossStreamAttentionFusion(nn.Module):
    
    def __init__(self, in_channels: int = 512, out_channels: int = 512):
        super(CrossStreamAttentionFusion, self).__init__()
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(out_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, rgb_map: torch.Tensor, ela_map: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        cat = torch.cat([rgb_map, ela_map], dim=1)  
        fused_map = self.fusion_conv(cat)           

        
        gate = self.spatial_gate(fused_map)         
        attended_map = fused_map * (1.0 + gate)     

        pooled_vec = self.pool(attended_map)
        pooled_vec = torch.flatten(pooled_vec, 1)   

        return attended_map, pooled_vec


class MaskDecoder(nn.Module):
    
    def __init__(self, in_channels: int = 512):
        super(MaskDecoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels, 256, kernel_size=4, stride=2, padding=1),  # 8 -> 16
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),          # 16 -> 32
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),           # 32 -> 64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),            # 64 -> 128
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),            # 128 -> 256
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            nn.Conv2d(16, 1, kernel_size=3, padding=1)                                  # 1 channel mask logit
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(x)


class DualBranchTamperDetector(nn.Module):
    
    def __init__(
        self,
        num_classes: int = 5,
        backbone: str = "resnet34",
        pretrained: bool = True,
        dropout: float = 0.3
    ):
        super(DualBranchTamperDetector, self).__init__()
        
        
        self.rgb_stream = FeatureExtractor(backbone_name=backbone, pretrained=pretrained)
        self.ela_stream = FeatureExtractor(backbone_name=backbone, pretrained=pretrained)

        feat_dim = self.rgb_stream.out_channels
        self.fusion = CrossStreamAttentionFusion(in_channels=feat_dim, out_channels=feat_dim)

        
        combined_dim = feat_dim * 3  

       
        self.binary_head = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 1)  
        )

        
        self.type_head = nn.Sequential(
            nn.Linear(combined_dim, 384),
            nn.BatchNorm1d(384),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(384, num_classes)
        )

        
        self.mask_decoder = MaskDecoder(in_channels=feat_dim)

    def forward(
        self,
        rgb: torch.Tensor,
        ela: torch.Tensor,
        return_mask: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        Args:
          - rgb: (B, 3, H, W)
          - ela: (B, 3, H, W)
          - return_mask: whether to decode the localization mask
        Returns:
          Dict containing:
            - binary_logits: (B, 1)
            - type_logits: (B, num_classes)
            - mask_logits: (B, 1, H, W) if return_mask else None
            - fused_feature_map: (B, 512, H', W')
        """

        rgb_map, rgb_vec = self.rgb_stream(rgb)
        ela_map, ela_vec = self.ela_stream(ela)

        fused_map, fused_vec = self.fusion(rgb_map, ela_map)

        global_repr = torch.cat([rgb_vec, ela_vec, fused_vec], dim=1)
        binary_logits = self.binary_head(global_repr)
        type_logits = self.type_head(global_repr)

        mask_logits = self.mask_decoder(fused_map) if return_mask else None

        return {
            "binary_logits": binary_logits,
            "type_logits": type_logits,
            "mask_logits": mask_logits,
            "fused_map": fused_map
        }
