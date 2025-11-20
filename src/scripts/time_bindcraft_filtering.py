import random
import json
from pathlib import Path
from datetime import datetime

import polars as pl
from pydantic import BaseModel

from src.bindcraft_utils.filters import BindCraftFiltersForTarget
from src.bindcraft_utils.random_sequence import random_aa_sequence
from src.schemas.bindcraft.settings import (
    AdvancedSettings,
    FilterSettings,
    TargetSettings,
)
from src.general_utils.timer import Timer, time_for_filename


# This object will store the results of our test
class FilterTimingObject(BaseModel):
    sequence: str = ""
    length: int = 0
    load_time: float = 0
    filter_time: float = 0
    af2_time: float = 0
    pyrosetta_time: float = 0
    pass_filters: bool = False


if __name__ == "__main__":
    out_path = Path("../", "data/processed/")
    _filename_time = time_for_filename()

    _min_len = 40
    _max_len = 100
    _n_repeats = 10

    random.seed(69)
    _target_json_path = Path("../", "bindcraft", "settings_target", "PDL1.json")
    with open(_target_json_path, "r") as f:
        target_json = json.load(f)
    target_settings = TargetSettings(**target_json)
    advanced_settings = AdvancedSettings()
    filters = FilterSettings()
    filters_pipeline = BindCraftFiltersForTarget(
        target_settings, advanced_settings, filters
    )
    # List of ints from _min_len to _max_len at interval 10
    lengths = range(_min_len, _max_len + 1, 10)
    # List of results to be turned into a table
    results = []
    for length in lengths:
        timer = Timer()
        # Initialize the model for the given sequence lenth
        start_load = timer.elapsed()
        model = filters_pipeline.prep_model(length)
        end_load = timer.elapsed()
        # Make n predictions for n random sequences of the same length
        for i in range(1, _n_repeats + 1):
            result = FilterTimingObject()
            sequence = random_aa_sequence(length)
            filter_start = timer.elapsed()
            pass_filters = filters_pipeline.run(model, sequence)
            filter_end = timer.elapsed()
            # Write our result object
            result.sequence = sequence
            result.length = length
            result.load_time = end_load - start_load
            result.filter_time = filter_end - filter_start
            result.pass_filters = pass_filters
            # TODO: add timing inside of the function to get more broken-down measurements
            # also, implement all of the filtering steps (pyrosetta + other)

            # Create a dictionary of the object so we can write it to a df
            result_dict = result.model_dump()
            results.append(result_dict)

    results_df = pl.DataFrame(results)
    try:
        results_df.write_csv(out_path)
    except Exception as e:
        print(f"Couldn't write to path: {e}")
        results_df.write_csv()
