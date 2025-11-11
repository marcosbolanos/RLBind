import sys
import os

from bindcraft.functions.colabdesign_utils import predict_binder_complex
from bindcraft.functions.generic_utils import (
    load_json_settings,
    generate_directories,
    load_af2_models,
)
from colabdesign import mk_afdesign_model

_bindcraft_dir = "../../bindcraft"

sys.path.append(_bindcraft_dir)

### load settings from JSON
# We're setting up the default settings here as recommended on the bindcraft readme
settings_path = os.path.join(_bindcraft_dir, "settings_target", "PDL1.json")
filters_path = os.path.join(_bindcraft_dir, "settings_filters", "default_filters.json")
advanced_path = os.path.join(
    _bindcraft_dir, "settings_advanced", "default_4stage_multimer.json"
)
target_settings, advanced_settings, filters = load_json_settings(
    settings_path, filters_path, advanced_path
)

# We laod all of the required parameters from the settings, as done in bindcraft.py
multimer_validation = advanced_settings["use_multimer_design"]
mpnn_design_name = target_settings["binder_name"]
design_paths = generate_directories(target_settings["design_path"])
failure_csv = os.path.join(target_settings["design_path"], "failure_csv.csv")

# Dummy variables that the function doesn't need (Ruff signaled it)
# TODO: clean up the BindCraft function by removing them
length = "_"  # length is unused by the function, will not be kept in my implementation
trajectory_pdb = "_"  # This will also not be used

# This sets up the config for which AF2 models to use for which prediction
# BindCraft uses multiple models concurrently to increase accuracy
design_model, prediction_models, multimer_validation = load_af2_models(
    multimer_validation
)


def apply_bindcraft_filters(sequence: str) -> bool:
    complex_prediction_model = mk_afdesign_model(
        protocol="binder",
        num_recycles=advanced_settings["num_recycles_validation"],
        data_dir=advanced_settings["af_params_dir"],
        use_multimer=multimer_validation,
        use_initial_guess=advanced_settings["predict_initial_guess"],
        use_initial_atom_pos=advanced_settings["predict_bigbang"],
    )

    _, pass_af2_filters = predict_binder_complex(
        complex_prediction_model,
        sequence,
        mpnn_design_name,
        target_settings["starting_pdb"],
        target_settings["chains"],
        length,
        trajectory_pdb,
        prediction_models,
        advanced_settings,
        filters,
        design_paths,
        failure_csv,
    )
    if not pass_af2_filters:
        return False

    return True


if __name__ == "__main__":
    sequence = "MADEVRLRQLKELGKVGVVEAATGQYDLIRRLLKETGYTLVPTKDKVVEAAEAGLKVYGRLVTN"
    pass_filters = apply_bindcraft_filters(sequence)
    print(pass_filters)
