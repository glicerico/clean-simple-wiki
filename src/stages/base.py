"""Base utilities for processing stages."""

import os
from typing import Optional


def stage_path(args, stage_name: str) -> str:
    """Get the output path for a processing stage.
    
    Args:
        args: Argument namespace with output configuration
        stage_name: Name of the processing stage
        
    Returns:
        Full path to the stage output file
    """
    filename = f"{args.out_prefix}_{stage_name}.parquet"
    return os.path.join(args.output_dir, filename)
