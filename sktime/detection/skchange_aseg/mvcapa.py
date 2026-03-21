"""Compatibility wrapper for MVCAPA."""

import numpy as np

from sktime.detection._skchange.anomaly_detectors import CAPA


class MVCAPA(CAPA):
    """Multivariate CAPA compatibility wrapper with affected-component inference."""

    def __init__(
        self,
        collective_saving=None,
        point_saving=None,
        collective_penalty="combined",
        collective_penalty_scale: float = 2.0,
        point_penalty="sparse",
        point_penalty_scale: float = 2.0,
        min_segment_length: int = 2,
        max_segment_length: int = 1000,
        ignore_point_anomalies: bool = False,
        segment_saving=None,
        segment_penalty=None,
        find_affected_components: bool = True,
    ):
        self.collective_saving = collective_saving
        self.point_saving = point_saving
        self.collective_penalty = collective_penalty
        self.collective_penalty_scale = collective_penalty_scale
        self.point_penalty = point_penalty
        self.point_penalty_scale = point_penalty_scale
        self.min_segment_length = min_segment_length
        self.max_segment_length = max_segment_length
        self.ignore_point_anomalies = ignore_point_anomalies
        self.segment_saving = segment_saving
        self.segment_penalty = segment_penalty
        self.find_affected_components = find_affected_components

        segment_saving = collective_saving if segment_saving is None else segment_saving
        segment_penalty = (
            self._resolve_penalty(
                collective_penalty,
                collective_penalty_scale,
                default_name="combined",
            )
            if segment_penalty is None
            else segment_penalty
        )
        point_penalty_resolved = self._resolve_penalty(
            point_penalty,
            point_penalty_scale,
            default_name="sparse",
        )

        super().__init__(
            segment_saving=segment_saving,
            segment_penalty=segment_penalty,
            point_saving=point_saving,
            point_penalty=point_penalty_resolved,
            min_segment_length=min_segment_length,
            max_segment_length=max_segment_length,
            ignore_point_anomalies=ignore_point_anomalies,
            find_affected_components=find_affected_components,
        )

    @staticmethod
    def _resolve_penalty(penalty, scale: float, default_name: str):
        if isinstance(penalty, str):
            if penalty == default_name:
                return None
            raise ValueError(
                "Only default string penalties are supported by this compatibility "
                f"wrapper: expected '{default_name}', got '{penalty}'."
            )

        if penalty is None:
            return None

        if np.isscalar(penalty):
            return penalty * scale

        return np.asarray(penalty) * scale
