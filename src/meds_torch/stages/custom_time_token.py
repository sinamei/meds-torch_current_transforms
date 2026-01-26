"""Custom time token stage for MEDS-transforms pipeline.

This stage adds time-derived measurements (TIME//START//TOKEN and TIME//DELTA//TOKEN)
to a MEDS cohort as separate observations at each unique time.
"""

from collections.abc import Callable

import polars as pl
from MEDS_transforms.stages import Stage
from omegaconf import DictConfig

from meds_torch.utils.custom_time_token import add_time_derived_measurements_fntr


@Stage.register
def custom_time_token(stage_cfg: DictConfig) -> Callable[[pl.LazyFrame], pl.LazyFrame]:
    """Adds time-derived measurements to a MEDS cohort as separate observations at each unique time.

    This is a wrapper around the custom_time_token utility that integrates with MEDS-transforms.

    Args:
        stage_cfg: Configuration for the custom_time_token stage.

    Returns:
        A function that transforms a MEDS dataframe by adding time tokens.
    """
    return add_time_derived_measurements_fntr(stage_cfg)
