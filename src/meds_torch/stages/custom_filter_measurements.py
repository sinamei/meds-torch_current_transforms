"""Custom filter measurements stage for MEDS-transforms pipeline.

This stage filters measurements by code frequency with support for additional codes
that should always be retained (e.g., MEDS_DEATH, HOSPITAL_ADMISSION, etc.).
"""

import logging

from MEDS_transforms.mapreduce.mapper import map_over
from MEDS_transforms.stages import Stage
from omegaconf import DictConfig

from meds_torch.utils.custom_filter_measurements import filter_measurements_fntr

logger = logging.getLogger(__name__)


def main(cfg: DictConfig):
    """Filters measurements by code frequency with support for additional required codes.

    This is a wrapper around the custom_filter_measurements utility that integrates with MEDS-transforms.
    """
    map_over(cfg, compute_fn=filter_measurements_fntr)


# Register the stage with MEDS-transforms
# This is a data stage (processes data shards, not metadata)
stage = Stage.register(main_fn=main, is_metadata=False)
