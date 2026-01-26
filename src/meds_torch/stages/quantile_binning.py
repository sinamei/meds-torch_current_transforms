"""Quantile binning stage for MEDS-transforms pipeline.

This stage bins numeric values into quantiles and updates both the data shards and metadata.
It collapses the bin number into the code name (e.g., lab//A becomes lab//A//_Q_1).

WARNING: DO NOT RUN THIS WITH PARALLELISM as it will recursively perform quantile binning N workers times.
"""

import logging
from collections.abc import Callable
from pathlib import Path

import polars as pl
from MEDS_transforms.stages import Stage
from omegaconf import DictConfig

from meds_torch.utils.quantile_binning import (
    convert_metadata_codes_to_discrete_quantiles,
    quantile_normalize,
)

logger = logging.getLogger(__name__)


@Stage.register
def quantile_binning(
    stage_cfg: DictConfig, code_metadata: pl.LazyFrame, code_modifiers: list[str] | None = None
) -> Callable[[pl.LazyFrame], pl.LazyFrame]:
    """Bins the numeric values and collapses the bin number into the code name.

    This function returns a transformation that bins numeric values into quantiles based on
    metadata and custom quantile specifications.

    WARNING: DO NOT RUN THIS WITH PARALLELISM as it will recursively perform quantile binning N workers times.

    Args:
        stage_cfg: Configuration for the quantile_binning stage, including custom_quantiles.
        code_metadata: Metadata about codes including quantile information.
        code_modifiers: Optional list of code modifier columns.

    Returns:
        A function that transforms a MEDS dataframe by binning numeric values.
    """
    custom_quantiles = stage_cfg.get("custom_quantiles", {})

    def transform_fn(df: pl.LazyFrame) -> pl.LazyFrame:
        return quantile_normalize(
            df,
            code_metadata,
            code_modifiers=code_modifiers,
            custom_quantiles=custom_quantiles,
        )

    return transform_fn


def quantile_binning_metadata_main(cfg: DictConfig):
    """Updates code metadata to reflect quantile-based codes.

    This is a metadata-only stage that converts the code metadata after quantile binning.
    """
    custom_quantiles = cfg.stage_cfg.get("custom_quantiles", {})

    metadata_input_dir = Path(cfg.stage_cfg.metadata_input_dir)
    code_metadata = pl.read_parquet(metadata_input_dir / "codes.parquet", use_pyarrow=True)
    quantile_code_metadata = convert_metadata_codes_to_discrete_quantiles(code_metadata, custom_quantiles)

    output_fp = metadata_input_dir / "codes.parquet"
    output_fp.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Quantile metadata updated. Writing to {output_fp}")
    quantile_code_metadata.write_parquet(output_fp, use_pyarrow=True)


# Register as a metadata stage
quantile_binning_metadata = Stage.register(main_fn=quantile_binning_metadata_main, is_metadata=True)
