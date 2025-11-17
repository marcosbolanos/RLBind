import sys
import os

import pyrosetta as pr

# Import directly from the function modules to avoid triggering bindcraft.py execution
from bindcraft.functions.colabdesign_utils import predict_binder_complex
from bindcraft.functions.generic_utils import (
    generate_directories,
    load_af2_models,
)
from colabdesign import mk_afdesign_model

# Import schemas
from schemas.bindcraft.settings import TargetSettings, AdvancedSettings, FilterSettings


class BindCraftFiltersForTarget:
    def __init__(
        self,
        target_settings: TargetSettings,
        advanced_settings: AdvancedSettings,
        filters: FilterSettings,
    ) -> None:
        self.target_settings = target_settings
        self.advanced_settings = advanced_settings
        self.filters = filters

        self.multimer_validation = advanced_settings.use_multimer_design
        self.mpnn_design_name = target_settings.binder_name
        self.design_paths = generate_directories(target_settings.design_path)
        self.failure_csv = os.path.join(target_settings.design_path, "failure_csv.csv")
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

        Arg:
            sequence_length: The length of the amino acid sequence to validate
            trajectory_pdb_path: Optional path to trajectory PDB for initial guess/bigbang mode

        Returns:
            The prepared complex prediction model
        """
        # Create the complex prediction model
        complex_prediction_model = mk_afdesign_model(
            protocol="binder",
            num_recycles=self.advanced_settings.num_recycles_validation,
            data_dir=self.advanced_settings.af_params_dir,
            use_multimer=self.multimer_validation,
            use_initial_guess=self.advanced_settings.predict_initial_guess,
            use_initial_atom_pos=self.advanced_settings.predict_bigbang,
        )

        # Prepare model inputs based on settings
        if (
            self.advanced_settings.predict_initial_guess
            or self.advanced_settings.predict_bigbang
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
                rm_target_seq=self.advanced_settings.rm_template_seq_predict,
                rm_target_sc=self.advanced_settings.rm_template_sc_predict,
                rm_template_ic=True,
            )
        else:
            complex_prediction_model.prep_inputs(
                pdb_filename=self.target_settings.starting_pdb,
                chain=self.target_settings.chains,
                binder_len=sequence_length,
                rm_target_seq=self.advanced_settings.rm_template_seq_predict,
                rm_target_sc=self.advanced_settings.rm_template_sc_predict,
            )

        return complex_prediction_model

    def run(self, complex_prediction_model, sequence):
        """
        Run the prediction and check if the sequence passes all filters.

        Args:
            complex_prediction_model: The prepared AF2 model
            sequence: The amino acid sequence to validate

        Returns:
            True if sequence passes all filters, False otherwise
        """
        # Convert BaseModels to dictionaries for compatibility with bindcraft functions
        advanced_settings_dict = self.advanced_settings.model_dump()
        filters_dict = self.filters.model_dump(by_alias=True)

        # Predict and check filters
        _, pass_af2_filters = predict_binder_complex(
            complex_prediction_model,
            sequence,
            self.mpnn_design_name,
            self.target_settings.starting_pdb,
            self.target_settings.chains,
            self.length,  # This is the dummy variable, but kept for compatibility
            self.trajectory_pdb,  # This is also dummy
            self.prediction_models,
            advanced_settings_dict,
            filters_dict,
            self.design_paths,
            self.failure_csv,
        )

        return pass_af2_filters


if __name__ == "__main__":
    import json

    ### load settings from JSON using BaseModels
    # We're setting up the default settings here as recommended on the bindcraft readme
    settings_path = os.path.join(_bindcraft_dir, "settings_target/PDL1.json")

    # Load JSON and create BaseModel instances
    with open(settings_path) as f:
        target_settings = TargetSettings(**json.load(f))

    filters = FilterSettings()

    advanced_settings = AdvancedSettings()

    pr.init(
        f"-ignore_unrecognized_res -ignore_zero_occupancy -mute all -holes:dalphaball {advanced_settings.dalphaball_path} -corrections::beta_nov16 true -relax:default_repeats 1"
    )
    sequence = "MADEVRLRQLKELGKVGVVEAATGQYDLIRRLLKETGYTLVPTKDKVVEAAEAGLKVYGRLVTN"
    sequence_length = len(sequence)

    Filters = BindCraftFiltersForTarget(target_settings, advanced_settings, filters)
    model = Filters.prep_model(sequence_length)
    pass_filters = Filters.run(model, sequence)
    print(pass_filters)
