"""Quantile binning stage for MEDS-transforms pipeline.

This stage bins numeric values into quantiles and updates both the data shards and metadata.
It collapses the bin number into the code name (e.g., lab//A becomes lab//A//_Q_1).

WARNING: DO NOT RUN THIS WITH PARALLELISM as it will recursively perform quantile binning N workers times.
"""

import logging
from pathlib import Path

import polars as pl
from MEDS_transforms.mapreduce.mapper import map_over
from MEDS_transforms.stages import Stage
from omegaconf import DictConfig

from meds_torch.utils.quantile_binning import (
    convert_metadata_codes_to_discrete_quantiles,
    quantile_normalize,
)

logger = logging.getLogger(__name__)


def main(cfg: DictConfig):
    """Bins the numeric values and collapses the bin number into the code name.

    This stage performs two operations:
    1. Transforms data shards by binning numeric values into quantiles
    2. Updates the code metadata to reflect the new quantile-based codes

    DO NOT RUN THIS WITH PARALLELISM as it will recursively perform quantile binning N workers times.
    """

    def normalize(df, code_metadata, code_modifiers=None):
        return quantile_normalize(
            df,
            code_metadata,
            code_modifiers=code_modifiers,
            custom_quantiles=cfg.stage_cfg.get("custom_quantiles", {}),
        )

    map_over(cfg, compute_fn=normalize)

    custom_quantiles = cfg.stage_cfg.get("custom_quantiles", {})

    metadata_input_dir = Path(cfg.stage_cfg.metadata_input_dir)
    code_metadata = pl.read_parquet(metadata_input_dir / "codes.parquet", use_pyarrow=True)
    quantile_code_metadata = convert_metadata_codes_to_discrete_quantiles(code_metadata, custom_quantiles)

    output_fp = metadata_input_dir / "codes.parquet"
    output_fp.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Indices assigned. Writing to {output_fp}")
    quantile_code_metadata.write_parquet(output_fp, use_pyarrow=True)


# Register the stage with MEDS-transforms
# This is a data stage that also modifies metadata
stage = Stage.register(main_fn=main, is_metadata=False)
