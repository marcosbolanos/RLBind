import sys
import os

# Get the absolute path to bindcraft directory
# __file__ is the current script location
_current_dir = os.path.dirname(os.path.abspath(__file__))
_bindcraft_dir = os.path.abspath(os.path.join(_current_dir, "../../bindcraft"))

# Add bindcraft to path BEFORE any imports from it
if _bindcraft_dir not in sys.path:
    sys.path.insert(0, _bindcraft_dir)

# Import directly from the function modules to avoid triggering bindcraft.py execution
from functions.colabdesign_utils import predict_binder_complex
from functions.generic_utils import (
    load_json_settings,
    generate_directories,
    load_af2_models,
)
from colabdesign import mk_afdesign_model

### load settings from JSON
# We're setting up the default settings here as recommended on the bindcraft readme
settings_path = os.path.join(_bindcraft_dir, "settings_target/PDL1.json")
filters_path = os.path.join(_bindcraft_dir, "settings_filters/default_filters.json")
advanced_path = os.path.join(
    _bindcraft_dir, "settings_advanced/default_4stage_multimer.json"
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


def apply_bindcraft_filters(sequence: str, trajectory_pdb_path: str = None) -> bool:
    """
    Apply BindCraft filters to a given sequence.

    Args:
        sequence: The amino acid sequence to validate
        trajectory_pdb_path: Optional path to trajectory PDB for initial guess/bigbang mode

    Returns:
        True if sequence passes all filters, False otherwise
    """
    # Calculate sequence length
    binder_length = len(sequence)

    # Create the complex prediction model
    complex_prediction_model = mk_afdesign_model(
        protocol="binder",
        num_recycles=advanced_settings["num_recycles_validation"],
        data_dir=advanced_settings["af_params_dir"],
        use_multimer=multimer_validation,
        use_initial_guess=advanced_settings["predict_initial_guess"],
        use_initial_atom_pos=advanced_settings["predict_bigbang"],
    )

    # Prepare model inputs based on settings
    if (
        advanced_settings["predict_initial_guess"]
        or advanced_settings["predict_bigbang"]
    ):
        if trajectory_pdb_path is None:
            raise ValueError(
                "trajectory_pdb_path is required when using initial_guess or bigbang mode"
            )
        complex_prediction_model.prep_inputs(
            pdb_filename=trajectory_pdb_path,
            chain="A",
            binder_chain="B",
            binder_len=binder_length,
            use_binder_template=True,
            rm_target_seq=advanced_settings["rm_template_seq_predict"],
            rm_target_sc=advanced_settings["rm_template_sc_predict"],
            rm_template_ic=True,
        )
    else:
        complex_prediction_model.prep_inputs(
            pdb_filename=target_settings["starting_pdb"],
            chain=target_settings["chains"],
            binder_len=binder_length,
            rm_target_seq=advanced_settings["rm_template_seq_predict"],
            rm_target_sc=advanced_settings["rm_template_sc_predict"],
        )

    # Predict and check filters
    _, pass_af2_filters = predict_binder_complex(
        complex_prediction_model,
        sequence,
        mpnn_design_name,
        target_settings["starting_pdb"],
        target_settings["chains"],
        length,  # This is the dummy variable, but kept for compatibility
        trajectory_pdb,  # This is also dummy
        prediction_models,
        advanced_settings,
        filters,
        design_paths,
        failure_csv,
    )

    return pass_af2_filters


if __name__ == "__main__":
    sequence = "MADEVRLRQLKELGKVGVVEAATGQYDLIRRLLKETGYTLVPTKDKVVEAAEAGLKVYGRLVTN"
    pass_filters = apply_bindcraft_filters(sequence)
    print(pass_filters)
