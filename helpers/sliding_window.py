import torch
import torch.nn.functional as F


class SlidingWindowInferer:
    def __init__(self, crop_size=(1024, 1024), stride=(768, 768), num_classes=19):
        """
        Args:
            crop_size (tuple): 训练时的裁剪尺寸 (H, W)
            stride (tuple): 滑动步长 (H, W)，建议小于 crop_size 以产生重叠
            num_classes (int): 类别数量
        """
        self.crop_size = crop_size
        self.stride = stride
        self.num_classes = num_classes

    @torch.no_grad()
    def __call__(self, model, image):
        """
        Args:
            model: 你的分割模型
            image (torch.Tensor): 输入图像，形状为 (1, C, H, W)
        """
        model.eval()
        n, c, h, w = image.shape
        device = image.device
        
        # 1. 填充原图以适配窗口
        pad_h = max(self.crop_size[0] - h, 0)
        pad_w = max(self.crop_size[1] - w, 0)
        image = F.pad(image, (0, pad_w, 0, pad_h), mode='constant', value=0)
        _, _, h_pad, w_pad = image.shape

        # 2. 初始化全分辨率累加器 (内部上采样的关键)
        preds = torch.zeros((n, self.num_classes, h_pad, w_pad), device=device)
        count = torch.zeros((n, 1, h_pad, w_pad), device=device)

        # 3. 生成滑动窗口的坐标
        h_starts = list(range(0, h_pad - self.crop_size[0] + 1, self.stride[0]))
        if h_starts[-1] + self.crop_size[0] < h_pad:
            h_starts.append(h_pad - self.crop_size[0])
            
        w_starts = list(range(0, w_pad - self.crop_size[1] + 1, self.stride[1]))
        if w_starts[-1] + self.crop_size[1] < w_pad:
            w_starts.append(w_pad - self.crop_size[1])

        # 3. 坐标计算... (同前)

        for h_s in h_starts:
            for w_s in w_starts:
                h_e, w_e = h_s + self.crop_size[0], w_s + self.crop_size[1]
                crop_tile = image[:, :, h_s:h_e, w_s:w_e]
                
                out = model(crop_tile, return_features = False, return_dict = False)
                
                # --- 内部上采样逻辑 ---
                # 将 1/4 尺寸的输出直接拉伸到当前窗口的原始尺寸 (如 1024x1024)
                if out.shape[-2:] != self.crop_size:
                    out = F.interpolate(out, size=self.crop_size, mode='bilinear', align_corners=False)
                
                # 概率累加 (比 Logits 累加更平滑)
                prob = F.softmax(out, dim=1)
                
                preds[:, :, h_s:h_e, w_s:w_e] += prob
                count[:, :, h_s:h_e, w_s:w_e] += 1

        final_out = (preds / count)[:, :, :h, :w]
        return final_out # 直接返回跟原图标签尺寸一致的概率图