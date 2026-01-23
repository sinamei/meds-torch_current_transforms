"""Tokenization stage for MEDS-transforms pipeline.

This stage converts normalized MEDS data into tokenized sequences suitable for deep learning models.
It splits data into static (time=null) and dynamic (time!=null) components, then creates:
- schemas: Static data + unique times per subject
- event_seqs: Dynamic sequences grouped by subject and time with time deltas
"""

import logging
from pathlib import Path

import polars as pl
from MEDS_transforms.mapreduce.rwlock import rwlock_wrap
from MEDS_transforms.mapreduce.shard_iteration import shard_iterator
from MEDS_transforms.stages import Stage
from omegaconf import DictConfig, OmegaConf

# Import the processing functions from the existing module
from meds_torch.utils.custom_tokenization import extract_seq_of_subject_events, extract_statics_and_schema

logger = logging.getLogger(__name__)


def write_lazyframe(df: pl.LazyFrame, out_fp: Path) -> None:
    """Write a LazyFrame to a parquet file."""
    if isinstance(df, pl.LazyFrame):
        df = df.collect()
    df.write_parquet(out_fp, use_pyarrow=True)


def main(cfg: DictConfig):
    """Run tokenization on the MEDS data.

    This function processes MEDS data and creates:
    - schemas/: Static data and unique times per subject
    - event_seqs/: Dynamic event sequences with time deltas

    Args:
        cfg: Hydra configuration containing stage_cfg with output_dir
    """
    logger.info(
        f"Running with config:\n{OmegaConf.to_yaml(cfg)}\n"
        f"Stage: {cfg.stage}\n\n"
        f"Stage config:\n{OmegaConf.to_yaml(cfg.stage_cfg)}"
    )

    output_dir = Path(cfg.stage_cfg.output_dir)
    if train_only := cfg.stage_cfg.get("train_only", False):
        raise ValueError(f"train_only={train_only} is not supported for this stage.")
    shards_single_output, include_only_train = shard_iterator(cfg)

    for in_fp, out_fp in shards_single_output:
        sharded_path = out_fp.relative_to(output_dir)

        schema_out_fp = output_dir / "schemas" / sharded_path
        event_seq_out_fp = output_dir / "event_seqs" / sharded_path

        logger.info(f"Tokenizing {str(in_fp.resolve())} into schemas at {str(schema_out_fp.resolve())}")

        rwlock_wrap(
            in_fp,
            schema_out_fp,
            pl.scan_parquet,
            write_lazyframe,
            extract_statics_and_schema,
            do_overwrite=cfg.do_overwrite,
        )

        logger.info(f"Tokenizing {str(in_fp.resolve())} into event_seqs at {str(event_seq_out_fp.resolve())}")

        rwlock_wrap(
            in_fp,
            event_seq_out_fp,
            pl.scan_parquet,
            write_lazyframe,
            extract_seq_of_subject_events,
            do_overwrite=cfg.do_overwrite,
        )

    logger.info(f"Done with {cfg.stage}")


# Register the stage with MEDS-transforms
stage = Stage.register(main_fn=main)
