"""
Multimodal skin lesion classification model.

Architecture:
    clinical image  -> ImageEncoder  -\
    dermoscopic img -> ImageEncoder  --> GatedFusion -> classifier -> logits
    metadata         -> MetadataEncoder -/

Each modality is encoded separately, then combined with a learned gate
that weighs how much each modality contributes per-sample (e.g. the
model can learn to lean on the dermoscopic image when it's sharp, or
fall back on metadata when the images are ambiguous).
"""

from typing import Dict, Tuple

import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0


class ImageEncoder(nn.Module):
    """Wraps an EfficientNet-B0 backbone and projects its features to a fixed size."""

    def __init__(self, out_dim: int = 256, pretrained: bool = True):
        super().__init__()
        weights = "IMAGENET1K_V1" if pretrained else None
        backbone = efficientnet_b0(weights=weights)
        in_features = backbone.classifier[1].in_features
        backbone.classifier = nn.Identity()  # drop the original ImageNet head
        self.backbone = backbone

        self.proj = nn.Sequential(
            nn.Linear(in_features, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.SiLU(),
            nn.Dropout(0.3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.proj(features)


class MetadataEncoder(nn.Module):
    """Encodes patient metadata (age, sex, anatomical site, melanocytic flag)."""

    def __init__(self, n_sex: int, n_site: int, n_mel: int, out_dim: int = 128):
        super().__init__()
        self.sex_emb = nn.Embedding(n_sex, 8)
        self.site_emb = nn.Embedding(n_site, 16)
        self.mel_emb = nn.Embedding(n_mel, 4)

        in_dim = 1 + 8 + 16 + 4  # age (scalar) + embeddings
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
            nn.Linear(64, out_dim),
            nn.ReLU(),
        )

    def forward(self, meta: Dict[str, torch.Tensor]) -> torch.Tensor:
        x = torch.cat(
            [
                meta["age"],
                self.sex_emb(meta["sex"]),
                self.site_emb(meta["site"]),
                self.mel_emb(meta["melanocytic"]),
            ],
            dim=1,
        )
        return self.mlp(x)


class GatedFusion(nn.Module):
    """Learns a per-sample softmax gate over the 3 modalities before combining them."""

    def __init__(self, d_clinical: int = 256, d_dermoscopic: int = 256, d_meta: int = 128, out_dim: int = 256):
        super().__init__()
        total_dim = d_clinical + d_dermoscopic + d_meta

        self.gate = nn.Sequential(
            nn.Linear(total_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 3),
            nn.Softmax(dim=1),
        )
        self.proj = nn.Sequential(
            nn.Linear(total_dim, out_dim),
            nn.ReLU(),
            nn.BatchNorm1d(out_dim),
            nn.Dropout(0.3),
        )

    def forward(
        self, clinical: torch.Tensor, dermoscopic: torch.Tensor, metadata: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        combined = torch.cat([clinical, dermoscopic, metadata], dim=1)
        gate_weights = self.gate(combined)  # (batch, 3)

        gated = torch.cat(
            [
                clinical * gate_weights[:, 0:1],
                dermoscopic * gate_weights[:, 1:2],
                metadata * gate_weights[:, 2:3],
            ],
            dim=1,
        )
        return self.proj(gated), gate_weights


class SkinLesionMultimodalModel(nn.Module):
    """Full model: two image encoders + metadata encoder + gated fusion + classifier head."""

    def __init__(self, num_classes: int, n_sex: int, n_site: int, n_mel: int):
        super().__init__()
        self.clinical_encoder = ImageEncoder()
        self.dermoscopic_encoder = ImageEncoder()
        self.metadata_encoder = MetadataEncoder(n_sex, n_site, n_mel)
        self.fusion = GatedFusion()

        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes),
        )

    def forward(
        self,
        clinical_img: torch.Tensor,
        dermoscopic_img: torch.Tensor,
        metadata: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        c = self.clinical_encoder(clinical_img)
        d = self.dermoscopic_encoder(dermoscopic_img)
        m = self.metadata_encoder(metadata)

        fused, gate_weights = self.fusion(c, d, m)
        logits = self.classifier(fused)
        return logits, gate_weights

    def freeze_backbones(self) -> None:
        """Freeze both image backbones (used during warm-up training)."""
        for param in self.clinical_encoder.backbone.parameters():
            param.requires_grad = False
        for param in self.dermoscopic_encoder.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbones(self) -> None:
        """Unfreeze both image backbones (used for fine-tuning after warm-up)."""
        for param in self.clinical_encoder.backbone.parameters():
            param.requires_grad = True
        for param in self.dermoscopic_encoder.backbone.parameters():
            param.requires_grad = True