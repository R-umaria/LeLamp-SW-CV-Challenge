"""Temporal smoothing for frame-level engagement predictions."""

from __future__ import annotations

from collections import Counter, deque

from backend.perception.engagement_detector import EngagementResult
from backend.utils.config import SmoothingConfig


class EngagementSmoother:
    """Convert noisy raw predictions into a stable short-window estimate."""

    def __init__(self, config: SmoothingConfig) -> None:
        if config.window_size < 1:
            raise ValueError("Smoothing window_size must be >= 1")
        self.config = config
        self._window: deque[EngagementResult] = deque(maxlen=config.window_size)

    def update(self, raw: EngagementResult) -> EngagementResult:
        self._window.append(raw)
        samples = list(self._window)
        total = len(samples)
        counts = Counter(sample.status for sample in samples)

        engaged_votes = counts["engaged"]
        disengaged_votes = counts["disengaged"]
        absent_votes = counts["absent"]
        not_engaged_votes = disengaged_votes + absent_votes

        latest_with_face = next((sample for sample in reversed(samples) if sample.face_bbox is not None), raw)

        if engaged_votes / total >= self.config.engaged_vote_ratio:
            status = "engaged"
            reason = self._top_reason(samples, status) or "smoothed_engaged"
            confidence = self._average_confidence(samples, status)
        elif absent_votes / total >= self.config.absent_vote_ratio:
            status = "absent"
            reason = self._top_reason(samples, status) or "smoothed_absent"
            confidence = self._average_confidence(samples, status)
        elif not_engaged_votes / total >= self.config.disengaged_vote_ratio:
            status = "disengaged"
            reason = self._top_reason(samples, "disengaged") or self._top_reason(samples, "absent") or "smoothed_disengaged"
            confidence = self._average_confidence(samples, "disengaged", fallback_status="absent")
        else:
            # Ambiguous windows should not create a new state by themselves. Use the
            # latest prediction but lower confidence to make the FSM conservative.
            status = raw.status
            reason = f"unstable_window_{raw.reason}"
            confidence = min(raw.confidence, 0.55)

        if status == "absent":
            latest_with_face = raw

        return EngagementResult(
            status=status,
            confidence=max(0.0, min(0.99, confidence)),
            reason=reason,
            face_bbox=latest_with_face.face_bbox,
            face_center_norm=latest_with_face.face_center_norm,
            face_area_ratio=latest_with_face.face_area_ratio,
            raw_face_count=raw.raw_face_count,
            candidate_count=raw.candidate_count,
            selected_face_score=latest_with_face.selected_face_score,
        )

    def _top_reason(self, samples: list[EngagementResult], status: str) -> str:
        reasons = Counter(sample.reason for sample in samples if sample.status == status)
        if not reasons:
            return ""
        return reasons.most_common(1)[0][0]

    def _average_confidence(
        self,
        samples: list[EngagementResult],
        status: str,
        fallback_status: str | None = None,
    ) -> float:
        selected = [sample.confidence for sample in samples if sample.status == status]
        if not selected and fallback_status is not None:
            selected = [sample.confidence for sample in samples if sample.status == fallback_status]
        if not selected:
            return 0.0
        return sum(selected) / len(selected)
