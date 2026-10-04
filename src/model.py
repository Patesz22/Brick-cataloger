import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class ArcMarginProduct(nn.Module):
    """
    Computes angular margin cosine logits (ArcFace) for hard-negative metric learning.
    """

    def __init__(self, in_features: int, out_features: int, scale: float = 30.0, margin: float = 0.35):
        """
        Initializes metric parameters and weight vectors.

        @parameters:
            @param in_features: int - Dimensionality of incoming feature embeddings.
            @param out_features: int - Number of target classes.
            @param scale: float - Logit scaling factor.
            @param margin: float - Additive angular margin penalty in radians.
        @returns:
            None
        """
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scale = scale
        self.margin = margin
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.threshold = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self, features: torch.Tensor, labels: torch.Tensor | None = None) -> torch.Tensor:
        """
        Projects normalized embeddings against normalized class centroids.

        @parameters:
            @param features: torch.Tensor - Input feature tensor of shape (B, in_features).
            @param labels: torch.Tensor | None - Ground truth class indices for margin injection.
        @returns:
            torch.Tensor - Scaled angular logits of shape (B, out_features).
        """
        cosine = F.linear(F.normalize(features), F.normalize(self.weight))

        if labels is None or not self.training:
            return cosine * self.scale

        sine = torch.sqrt(torch.clamp(1.0 - torch.pow(cosine, 2), min=1e-7, max=1.0))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.threshold, phi, cosine - self.mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1.0)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        return output * self.scale


class FocalLoss(nn.Module):
    """
    Multi-class focal loss penalizing hard negatives and down-weighting easy examples.
    """

    def __init__(self, gamma: float = 2.0, reduction: str = "mean"):
        """
        Initializes focal focusing parameters.

        @parameters:
            @param gamma: float - Focusing parameter modulating easy vs hard loss.
            @param reduction: str - Loss reduction strategy ('mean' or 'sum').
        @returns:
            None
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Evaluates the focal cross entropy on unnormalized logits.

        @parameters:
            @param logits: torch.Tensor - Model output logits of shape (B, C).
            @param targets: torch.Tensor - Ground truth indices of shape (B).
        @returns:
            torch.Tensor - Computed scalar loss.
        """
        log_prob = F.log_softmax(logits, dim=-1)
        prob = torch.exp(log_prob)
        target_log_prob = log_prob.gather(1, targets.unsqueeze(1)).squeeze(1)
        target_prob = prob.gather(1, targets.unsqueeze(1)).squeeze(1)

        focal_weight = torch.pow(1.0 - target_prob, self.gamma)
        loss = -focal_weight * target_log_prob

        if self.reduction == "mean":
            return loss.mean()
        return loss.sum()


class BrickNetDual(nn.Module):
    """
    Decoupled dual-head neural network using pretrained EfficientNet-B0 for
    metric part identification and an independent chromatic analyzer for plastic colors.
    """

    def __init__(self, num_parts: int, num_colors: int, embedding_dim: int = 256):
        """
        Constructs the geometry backbone, ArcFace metric head, and chromatic classifier.

        @parameters:
            @param num_parts: int - Number of distinct part design classes.
            @param num_colors: int - Number of plastic color classes.
            @param embedding_dim: int - Output dimension of the metric part embedding space.
        @returns:
            None
        """
        super(BrickNetDual, self).__init__()
        weights = models.EfficientNet_B0_Weights.DEFAULT
        base_model = models.efficientnet_b0(weights=weights)
        self.geo_backbone = base_model.features
        self.geo_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.geo_embed = nn.Sequential(
            nn.Linear(1280, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.PReLU()
        )
        self.part_head = ArcMarginProduct(in_features=embedding_dim, out_features=num_parts)

        self.color_net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_colors)
        )

    def freeze_backbone(self) -> None:
        """
        Freezes feature extractor layers during linear probing.

        @parameters:
            None
        @returns:
            None
        """
        for param in self.geo_backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """
        Unfreezes all backbone parameters for end-to-end fine tuning.

        @parameters:
            None
        @returns:
            None
        """
        for param in self.geo_backbone.parameters():
            param.requires_grad = True

    def forward(
            self,
            x_geo: torch.Tensor,
            x_color: torch.Tensor,
            part_labels: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Executes parallel forward inference over decoupled inputs.

        @parameters:
            @param x_geo: torch.Tensor - Grayscale/edge-invariant tensor (B, 3, 224, 224).
            @param x_color: torch.Tensor - Chromatic HSV/Lab color tensor (B, 3, 224, 224).
            @param part_labels: torch.Tensor | None - Part labels for ArcFace training.
        @returns:
            tuple[torch.Tensor, torch.Tensor] - (part_logits, color_logits).
        """
        geo_feats = self.geo_pool(self.geo_backbone(x_geo)).flatten(1)
        geo_embeddings = self.geo_embed(geo_feats)
        part_logits = self.part_head(geo_embeddings, part_labels)

        color_logits = self.color_net(x_color)
        return part_logits, color_logits
