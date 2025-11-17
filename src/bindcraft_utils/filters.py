import sys
import os

# Get the absolute path to bindcraft directory
# __file__ is the current script location
_current_dir = os.path.dirname(os.path.abspath(__file__))
_bindcraft_dir = os.path.abspath(os.path.join(_current_dir, "../../bindcraft"))

# Add bindcraft to path BEFORE any imports from it
if _bindcraft_dir not in sys.path:
    sys.path.insert(0, _bindcraft_dir)

import pyrosetta as pr

# Import directly from the function modules to avoid triggering bindcraft.py execution
from functions.colabdesign_utils import predict_binder_complex
from functions.generic_utils import (
    load_json_settings,
    generate_directories,
    load_af2_models,
)
from colabdesign import mk_afdesign_model


class BindCraftFiltersForTarget:
    def __init__(self, target_settings, advanced_settings, filters) -> None:
        self.target_settings = target_settings
        self.advanced_settings = advanced_settings
        self.filters = filters

        self.multimer_validation = advanced_settings["use_multimer_design"]
        self.mpnn_design_name = target_settings["binder_name"]
        self.design_paths = generate_directories(target_settings["design_path"])
        self.failure_csv = os.path.join(
            target_settings["design_path"], "failure_csv.csv"
        )
        self.length = "_"
        self.trajectory_pdb = "_"
        self.design_models, self.prediction_models, self.multimer_validation = (
            load_af2_models(self.multimer_validation)
        )

    def prep_model(
        self, sequence_length: int, trajectory_pdb_path: str | None = None
    ) -> bool:
        """
        Apply BindCraft filters to a given sequence.

        Args:
            sequence: The amino acid sequence to validate
            trajectory_pdb_path: Optional path to trajectory PDB for initial guess/bigbang mode

        Returns:
            True if sequence passes all filters, False otherwise
        """
        # Create the complex prediction model
        complex_prediction_model = mk_afdesign_model(
            protocol="binder",
            num_recycles=self.advanced_settings["num_recycles_validation"],
            data_dir=self.advanced_settings["af_params_dir"],
            use_multimer=self.multimer_validation,
            use_initial_guess=self.advanced_settings["predict_initial_guess"],
            use_initial_atom_pos=self.advanced_settings["predict_bigbang"],
        )

        # Prepare model inputs based on settings
        if (
            self.advanced_settings["predict_initial_guess"]
            or self.advanced_settings["predict_bigbang"]
        ):
            if trajectory_pdb_path is None:
                raise ValueError(
                    "trajectory_pdb_path is required when using initial_guess or bigbang mode"
                )
            complex_prediction_model.prep_inputs(
                pdb_filename=trajectory_pdb_path,
                chain="A",
                binder_chain="B",
                binder_len=sequence_length,
                use_binder_template=True,
                rm_target_seq=self.advanced_settings["rm_template_seq_predict"],
                rm_target_sc=self.advanced_settings["rm_template_sc_predict"],
                rm_template_ic=True,
            )
        else:
            complex_prediction_model.prep_inputs(
                pdb_filename=self.target_settings["starting_pdb"],
                chain=self.target_settings["chains"],
                binder_len=sequence_length,
                rm_target_seq=self.advanced_settings["rm_template_seq_predict"],
                rm_target_sc=self.advanced_settings["rm_template_sc_predict"],
            )

        return complex_prediction_model

    def run(self, complex_prediction_model, sequence):
        # Predict and check filters
        _, pass_af2_filters = predict_binder_complex(
            complex_prediction_model,
            sequence,
            self.mpnn_design_name,
            self.target_settings["starting_pdb"],
            self.target_settings["chains"],
            self.length,  # This is the dummy variable, but kept for compatibility
            self.trajectory_pdb,  # This is also dummy
            self.prediction_models,
            self.advanced_settings,
            self.filters,
            self.design_paths,
            self.failure_csv,
        )

        return pass_af2_filters


if __name__ == "__main__":
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

    pr.init(
        f"-ignore_unrecognized_res -ignore_zero_occupancy -mute all -holes:dalphaball {advanced_settings['dalphaball_path']} -corrections::beta_nov16 true -relax:default_repeats 1"
    )
    sequence = "MADEVRLRQLKELGKVGVVEAATGQYDLIRRLLKETGYTLVPTKDKVVEAAEAGLKVYGRLVTN"
    sequence_length = len(sequence)

    Filters = BindCraftFiltersForTarget(target_settings, advanced_settings, filters)
    model = Filters.prep_model(sequence_length)
    pass_filters = Filters.run(model, sequence)
    print(pass_filters)
