# Warmup wrapper: linearly increase LR from `warmup_factor * base_lr` to `base_lr`
class WarmupScheduler:
    def __init__(self, optimizer, base_scheduler, warmup_iters=0, warmup_factor=1.0, is_enabled = True):
        self.optimizer = optimizer
        self.base_scheduler = base_scheduler
        self.warmup_iters = int(warmup_iters)
        self.warmup_factor = float(warmup_factor)
        self.last_step = 0
        # record base lrs from optimizer param groups
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.is_enabled = is_enabled

    def step(self):
        if self.is_enabled :
            cur = self.last_step
            if cur < self.warmup_iters and self.warmup_iters > 0:
                # linear warmup factor for step index `cur` (1..warmup_iters)
                alpha = float(cur + 1) / float(self.warmup_iters)
                factor = self.warmup_factor + (1.0 - self.warmup_factor) * alpha
                for i, group in enumerate(self.optimizer.param_groups):
                    group['lr'] = self.base_lrs[i] * factor
            else:
                # sync base scheduler to the global step index `cur`
                # pass epoch=cur so LambdaLR computes based on full progress
                try:
                    self.base_scheduler.step(cur)
                except TypeError:
                    # fallback: call without arg
                    self.base_scheduler.step()
            self.last_step += 1
        else:
            self.base_scheduler.step()