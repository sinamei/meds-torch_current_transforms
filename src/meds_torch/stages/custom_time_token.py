"""Custom time token stage for MEDS-transforms pipeline.

This stage adds time-derived measurements (TIME//START//TOKEN and TIME//DELTA//TOKEN)
to a MEDS cohort as separate observations at each unique time.
"""

import logging

from MEDS_transforms.mapreduce.mapper import map_over
from MEDS_transforms.stages import Stage
from omegaconf import DictConfig

from meds_torch.utils.custom_time_token import add_time_derived_measurements_fntr

logger = logging.getLogger(__name__)


def main(cfg: DictConfig):
    """Adds time-derived measurements to a MEDS cohort as separate observations at each unique time.

    This is a wrapper around the custom_time_token utility that integrates with MEDS-transforms.
    """
    map_over(cfg, compute_fn=add_time_derived_measurements_fntr)


# Register the stage with MEDS-transforms
# This is a data stage (processes data shards, not metadata)
stage = Stage.register(main_fn=main, is_metadata=False)
