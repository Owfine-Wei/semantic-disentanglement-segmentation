"""
Warmup scheduler wrapper.

Linearly increases optimizer learning rates from
`warmup_factor * base_lr` up to `base_lr` over a number of iterations,
then delegates to a provided base scheduler.

Usage: call `step()` each training iteration.
"""

class WarmupScheduler:
    """
    Wraps a PyTorch LR scheduler with a linear warmup phase.

    Args:
        optimizer: optimizer with `param_groups` (expects 'lr').
        base_scheduler: scheduler to call after warmup (e.g., LambdaLR).
        warmup_iters (int): number of warmup iterations (0 disables warmup).
        warmup_factor (float): starting LR factor (multiplied with base lr).
        is_enabled (bool): if False, warmup is skipped entirely.
    """

    def __init__(self, optimizer, base_scheduler, warmup_iters=0, warmup_factor=1.0, is_enabled=True):
        self.optimizer = optimizer
        self.base_scheduler = base_scheduler
        self.warmup_iters = int(warmup_iters)
        self.warmup_factor = float(warmup_factor)
        self.last_step = 0
        # store the original learning rates for each param group
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.is_enabled = is_enabled

    def step(self):
        """
        Advance one iteration: apply warmup or advance base scheduler.

        While in warmup, scale each param group's lr linearly between
        `warmup_factor * base_lr` and `base_lr`. After warmup_iters, forward
        the step (using the global step index) to the wrapped scheduler.
        """
        if self.is_enabled:
            cur = self.last_step
            # Warmup phase: linearly increase factor from warmup_factor -> 1.0
            if cur < self.warmup_iters and self.warmup_iters > 0:
                alpha = float(cur + 1) / float(self.warmup_iters)
                factor = self.warmup_factor + (1.0 - self.warmup_factor) * alpha
                for i, group in enumerate(self.optimizer.param_groups):
                    group['lr'] = self.base_lrs[i] * factor
            else:
                # After warmup: synchronize base scheduler with global step
                # Pass `cur` so schedulers like LambdaLR compute based on full progress
                try:
                    self.base_scheduler.step(cur)
                except TypeError:
                    # Some schedulers expect no arg; fall back to that API
                    self.base_scheduler.step()
            self.last_step += 1
        else:
            # Warmup disabled: simply step the base scheduler
            self.base_scheduler.step()