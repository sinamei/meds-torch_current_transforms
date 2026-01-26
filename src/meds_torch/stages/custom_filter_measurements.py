"""Custom filter measurements stage for MEDS-transforms pipeline.

This stage filters measurements by code frequency with support for additional codes
that should always be retained (e.g., MEDS_DEATH, HOSPITAL_ADMISSION, etc.).
"""

from collections.abc import Callable

import polars as pl
from MEDS_transforms.stages import Stage
from omegaconf import DictConfig

from meds_torch.utils.custom_filter_measurements import filter_measurements_fntr


@Stage.register
def custom_filter_measurements(
    stage_cfg: DictConfig, code_metadata: pl.LazyFrame, code_modifiers: list[str] | None = None
) -> Callable[[pl.LazyFrame], pl.LazyFrame]:
    """Filters measurements by code frequency with support for additional required codes.

    This is a wrapper around the custom_filter_measurements utility that integrates with MEDS-transforms.

    Args:
        stage_cfg: Configuration for the custom_filter_measurements stage.
        code_metadata: Metadata about codes including occurrence counts.
        code_modifiers: Optional list of code modifier columns.

    Returns:
        A function that filters a MEDS dataframe based on code frequency.
    """
    return filter_measurements_fntr(stage_cfg, code_metadata, code_modifiers)
