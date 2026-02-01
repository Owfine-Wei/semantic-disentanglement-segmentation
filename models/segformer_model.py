import torch
import os
from .registry import register_models
from transformers import SegformerForSemanticSegmentation, SegformerConfig
from typing import Optional, Tuple, Union


class SegformerForSemanticSegmentationWithFeatures(SegformerForSemanticSegmentation):
    """
    Subclass of SegformerForSemanticSegmentation that optionally returns
    the decoder fused features (classifier之前的 feature map).
    Usage:
        model = SegformerForSemanticSegmentationWithFeatures.from_pretrained(...)
        logits, features = model(pixel_values, return_features=True)
    """

    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        return_features: bool = False,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], dict]:
        """
        Args:
            pixel_values: (B, C, H, W)
            return_features: 若为 True, 返回 (logits, features) 或 dict 包含 features
            output_hidden_states / return_dict: 与 transformers 风格兼容（但我们内部会强制 output_hidden_states=True）
            **kwargs: 其余传给 self.segformer(...)
        Returns:
            - 若 return_dict 为 False 且 return_features=False: 返回 logits (B, num_labels, H_out, W_out)
            - 若 return_dict 为 False 且 return_features=True: 返回 (logits, features)
            - 若 return_dict True: 返回 dict 包含键 "logits" 和（当请求时）"features"，并保留 encoder hidden_states
        """
        # 保持与 config 的兼容
        if return_dict is None:
            return_dict = self.config.use_return_dict

        # 我们需要 encoder 的 hidden states 来构造 decoder 的 features
        # 强制 output_hidden_states=True 以便拿到 encoder 各尺度特征
        outputs = self.segformer(
            pixel_values,
            output_hidden_states=True,
            return_dict=True,
            **kwargs
        )

        encoder_hidden_states = outputs.hidden_states  # tuple of (B, C_i, H_i, W_i)
        # ------------------------------------------
        # 下面复刻 decode_head 的融合逻辑，和原始 decode_head 行为保持一致
        # linear_c: ModuleList（每个 MLP），linear_fuse, batch_norm, activation, dropout, classifier
        # ------------------------------------------
        all_hidden_states = []
        batch_size = encoder_hidden_states[0].shape[0]

        # linear_c 的顺序应和 encoder_hidden_states 的顺序一一对应
        for enc_feat, mlp in zip(encoder_hidden_states, self.decode_head.linear_c):
            # enc_feat: (B, C_i, H_i, W_i)
            h, w = enc_feat.shape[2], enc_feat.shape[3]
            # mlp 通常把 (B, C_i, H, W) -> (B, H*W, embed_dim)
            x = mlp(enc_feat)
            # 还原为 (B, embed_dim, H, W)
            x = x.permute(0, 2, 1).reshape(batch_size, -1, h, w)
            # upsample 到第一级（encoder_hidden_states[0]）的分辨率（通常是 H/4, W/4）
            target_size = encoder_hidden_states[0].shape[2:]
            x = torch.nn.functional.interpolate(x, size=target_size, mode="bilinear", align_corners=False)
            all_hidden_states.append(x)

        # concat (reverse order 与 HF 实现一致)，并通过 linear_fuse + bn + act + dropout
        fused = torch.cat(all_hidden_states[::-1], dim=1)  # [B, 4*C, H/4, W/4] (具体 C 与配置有关)
        fused = self.decode_head.linear_fuse(fused)
        # 保持 decode_head 的 BN/activation/dropout 顺序
        # 注意：如果 decode_head 没有这些属性（极少情况），则会抛错，按你的 transformers 版本应存在
        fused = self.decode_head.batch_norm(fused)
        fused = self.decode_head.activation(fused)
        fused = self.decode_head.dropout(fused)

        features = fused  # classifier 之前的特征（semantic-rich）

        # classifier -> logits (保持原始 decode_head.classifier 逻辑)
        logits = self.decode_head.classifier(features)

        # 根据 return_dict / return_features 返回
        if not return_dict:
            if return_features:
                return logits, features
            else:
                return logits
        else:
            out = {
                "logits": logits,
                "hidden_states": outputs.hidden_states,  # encoder hidden states
            }
            if return_features:
                out["features"] = features
            return out


@register_models("segformer")
def get_model(num_classes, checkpoint=None):
    # 本地配置和基础权重的目录（必须包含 config.json）
    local_base_path = "../segformer-b3-weights/"

    print(f"Initializing model architecture from: {local_base_path}")

    # -----------------------------
    # 1. checkpoint 是单独的 bin / pth / pt
    # -----------------------------
    if checkpoint and checkpoint.endswith(('.bin', '.pth', '.pt')):
        print(f"Loading specific weight file: {checkpoint}")

        # 先基于本地 config 初始化“带 features 输出能力”的模型结构
        model = SegformerForSemanticSegmentationWithFeatures.from_pretrained(
            local_base_path,
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
            local_files_only=True
        )

        # 手动加载权重
        state_dict = torch.load(checkpoint, map_location='cpu')

        # 兼容 HF 官方权重里包了一层 "model" 的情况
        if 'model' in state_dict:
            state_dict = state_dict['model']

        msg = model.load_state_dict(state_dict, strict=False)
        print(f"Successfully loaded weights from {checkpoint}")
        print(f"Note: {msg}")

    # -----------------------------
    # 2. checkpoint 是 transformers 标准目录
    # -----------------------------
    elif checkpoint and os.path.isdir(checkpoint):
        model = SegformerForSemanticSegmentationWithFeatures.from_pretrained(
            checkpoint,
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
            local_files_only=True
        )

    # -----------------------------
    # 3. 无 checkpoint，用本地基础权重
    # -----------------------------
    else:
        model = SegformerForSemanticSegmentationWithFeatures.from_pretrained(
            local_base_path,
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
            local_files_only=True
        )

    # -----------------------------
    # 4. 移动到设备（DDP / 单卡均兼容）
    # -----------------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device, memory_format=torch.channels_last)

    return model