"""GRPO reward function using Manim verifier as deterministic judge."""

import re

from .manim_verifier import verify_code, VerifyResult


def count_play_calls(code: str) -> int:
    """Count self.play() calls in code."""
    return len(re.findall(r"self\.play\(", code))


def compute_reward(
    code: str,
    render_success_weight: float = 1.0,
    animation_bonus_per_play: float = 0.1,
    animation_bonus_cap: float = 0.5,
    timeout: int = 120,
) -> tuple[float, VerifyResult]:
    """
    Compute reward for a single code output.

    Reward components:
    - render_success: 1.0 if code renders successfully, 0.0 if not
    - animation_bonus: 0.1 per self.play() call, capped at 0.5

    Total reward range: 0.0 to 1.5

    The Manim renderer is a deterministic verifier - code either renders
    or it doesn't. This replaces the need for a reward model, exactly
    like DeepSeek-R1 uses math answer checkers.
    """
    result = verify_code(code, timeout=timeout)

    reward = 0.0

    # Binary render success
    if result.success:
        reward += render_success_weight

        # Animation bonus only for successful renders
        play_count = count_play_calls(code)
        animation_bonus = min(
            play_count * animation_bonus_per_play,
            animation_bonus_cap,
        )
        reward += animation_bonus

    return reward, result


def batch_compute_rewards(
    codes: list[str],
    render_success_weight: float = 1.0,
    animation_bonus_per_play: float = 0.1,
    animation_bonus_cap: float = 0.5,
    max_workers: int = 4,
    timeout: int = 120,
) -> list[tuple[float, VerifyResult]]:
    """Compute rewards for a batch of codes in parallel."""
    from concurrent.futures import ProcessPoolExecutor, as_completed

    results = [None] * len(codes)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(
                compute_reward,
                code,
                render_success_weight,
                animation_bonus_per_play,
                animation_bonus_cap,
                timeout,
            ): i
            for i, code in enumerate(codes)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                from .manim_verifier import ErrorType
                results[idx] = (
                    0.0,
                    VerifyResult(
                        success=False,
                        error_type=ErrorType.UNKNOWN,
                        error_message=str(e),
                        code=codes[idx],
                    ),
                )

    return results
