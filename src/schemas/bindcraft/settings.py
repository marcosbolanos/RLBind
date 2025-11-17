from typing import Optional
from pydantic import BaseModel, Field


class TargetSettings(BaseModel):
    design_path: str
    binder_name: str
    starting_pdb: str
    chains: str
    target_hotspot_residues: str
    lengths: list[int]
    number_of_final_designs: int


class ThresholdConfig(BaseModel):
    threshold: Optional[float] = None
    higher: bool


class AminoAcidThresholds(BaseModel):
    A: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=None, higher=False))
    C: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=None, higher=False))
    D: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=None, higher=False))
    E: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=None, higher=False))
    F: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=None, higher=False))
    G: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=None, higher=False))
    H: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=None, higher=False))
    I: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=None, higher=False))
    K: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=None, higher=False))
    L: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=None, higher=False))
    M: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=None, higher=False))
    N: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=None, higher=False))
    P: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=None, higher=False))
    Q: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=None, higher=False))
    R: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=None, higher=False))
    S: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=None, higher=False))
    T: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=None, higher=False))
    V: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=None, higher=False))
    W: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=None, higher=False))
    Y: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=None, higher=False))


class FilterSettings(BaseModel):
    MPNN_score: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=None, higher=False))
    MPNN_seq_recovery: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=None, higher=False))
    Average_pLDDT: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=0.8, higher=True))
    field_1_pLDDT: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=0.8, higher=True), alias="1_pLDDT")
    field_2_pLDDT: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=0.8, higher=True), alias="2_pLDDT")
    field_3_pLDDT: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=None, higher=True), alias="3_pLDDT")
    field_4_pLDDT: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=None, higher=True), alias="4_pLDDT")
    field_5_pLDDT: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=None, higher=True), alias="5_pLDDT")
    Average_pTM: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=0.55, higher=True))
    field_1_pTM: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=0.55, higher=True), alias="1_pTM")
    field_2_pTM: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=0.55, higher=True), alias="2_pTM")
    field_3_pTM: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=None, higher=True), alias="3_pTM")
    field_4_pTM: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=None, higher=True), alias="4_pTM")
    field_5_pTM: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=None, higher=True), alias="5_pTM")
    Average_i_pTM: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=0.5, higher=True))
    field_1_i_pTM: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=0.5, higher=True), alias="1_i_pTM")
    field_2_i_pTM: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=0.5, higher=True), alias="2_i_pTM")
    field_3_i_pTM: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=None, higher=True), alias="3_i_pTM")
    field_4_i_pTM: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=None, higher=True), alias="4_i_pTM")
    field_5_i_pTM: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=None, higher=True), alias="5_i_pTM")
    Average_pAE: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=None, higher=False))
    field_1_pAE: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=None, higher=False), alias="1_pAE")
    field_2_pAE: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=None, higher=False), alias="2_pAE")
    field_3_pAE: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=None, higher=False), alias="3_pAE")
    field_4_pAE: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=None, higher=False), alias="4_pAE")
    field_5_pAE: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=None, higher=False), alias="5_pAE")
    Average_i_pAE: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=0.35, higher=False))
    field_1_i_pAE: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=0.35, higher=False), alias="1_i_pAE")
    field_2_i_pAE: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=0.35, higher=False), alias="2_i_pAE")
    field_3_i_pAE: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=None, higher=False), alias="3_i_pAE")
    field_4_i_pAE: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=None, higher=False), alias="4_i_pAE")
    field_5_i_pAE: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=None, higher=False), alias="5_i_pAE")
    Average_i_pLDDT: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=None, higher=True))
    field_1_i_pLDDT: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=None, higher=True), alias="1_i_pLDDT")
    field_2_i_pLDDT: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=None, higher=True), alias="2_i_pLDDT")
    field_3_i_pLDDT: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=None, higher=True), alias="3_i_pLDDT")
    field_4_i_pLDDT: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=None, higher=True), alias="4_i_pLDDT")
    field_5_i_pLDDT: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=None, higher=True), alias="5_i_pLDDT")
    Average_ss_pLDDT: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=None, higher=True))
    field_1_ss_pLDDT: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=None, higher=True), alias="1_ss_pLDDT")
    field_2_ss_pLDDT: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=None, higher=True), alias="2_ss_pLDDT")
    field_3_ss_pLDDT: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=None, higher=True), alias="3_ss_pLDDT")
    field_4_ss_pLDDT: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=None, higher=True), alias="4_ss_pLDDT")
    field_5_ss_pLDDT: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=None, higher=True), alias="5_ss_pLDDT")
    Average_Unrelaxed_Clashes: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=None, higher=False))
    field_1_Unrelaxed_Clashes: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=None, higher=False), alias="1_Unrelaxed_Clashes")
    field_2_Unrelaxed_Clashes: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=None, higher=False), alias="2_Unrelaxed_Clashes")
    field_3_Unrelaxed_Clashes: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=None, higher=False), alias="3_Unrelaxed_Clashes")
    field_4_Unrelaxed_Clashes: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=None, higher=False), alias="4_Unrelaxed_Clashes")
    field_5_Unrelaxed_Clashes: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=None, higher=False), alias="5_Unrelaxed_Clashes")
    Average_Relaxed_Clashes: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=None, higher=False))
    field_1_Relaxed_Clashes: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=None, higher=False), alias="1_Relaxed_Clashes")
    field_2_Relaxed_Clashes: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=None, higher=False), alias="2_Relaxed_Clashes")
    field_3_Relaxed_Clashes: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=None, higher=False), alias="3_Relaxed_Clashes")
    field_4_Relaxed_Clashes: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=None, higher=False), alias="4_Relaxed_Clashes")
    field_5_Relaxed_Clashes: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=None, higher=False), alias="5_Relaxed_Clashes")
    Average_Binder_Energy_Score: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=0, higher=False))
    field_1_Binder_Energy_Score: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=0, higher=False), alias="1_Binder_Energy_Score")
    field_2_Binder_Energy_Score: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=0, higher=False), alias="2_Binder_Energy_Score")
    field_3_Binder_Energy_Score: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=None, higher=False), alias="3_Binder_Energy_Score")
    field_4_Binder_Energy_Score: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=None, higher=False), alias="4_Binder_Energy_Score")
    field_5_Binder_Energy_Score: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=None, higher=False), alias="5_Binder_Energy_Score")
    Average_Surface_Hydrophobicity: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=0.35, higher=False))
    field_1_Surface_Hydrophobicity: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=0.35, higher=False), alias="1_Surface_Hydrophobicity")
    field_2_Surface_Hydrophobicity: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=0.35, higher=False), alias="2_Surface_Hydrophobicity")
    field_3_Surface_Hydrophobicity: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=None, higher=False), alias="3_Surface_Hydrophobicity")
    field_4_Surface_Hydrophobicity: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=None, higher=False), alias="4_Surface_Hydrophobicity")
    field_5_Surface_Hydrophobicity: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=None, higher=False), alias="5_Surface_Hydrophobicity")
    Average_ShapeComplementarity: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=0.6, higher=True))
    field_1_ShapeComplementarity: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=0.55, higher=True), alias="1_ShapeComplementarity")
    field_2_ShapeComplementarity: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=0.55, higher=True), alias="2_ShapeComplementarity")
    field_3_ShapeComplementarity: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=None, higher=True), alias="3_ShapeComplementarity")
    field_4_ShapeComplementarity: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=None, higher=True), alias="4_ShapeComplementarity")
    field_5_ShapeComplementarity: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=None, higher=True), alias="5_ShapeComplementarity")
    Average_PackStat: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=None, higher=True))
    field_1_PackStat: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=None, higher=True), alias="1_PackStat")
    field_2_PackStat: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=None, higher=True), alias="2_PackStat")
    field_3_PackStat: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=None, higher=True), alias="3_PackStat")
    field_4_PackStat: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=None, higher=True), alias="4_PackStat")
    field_5_PackStat: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=None, higher=True), alias="5_PackStat")
    Average_dG: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=0, higher=False))
    field_1_dG: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=0, higher=False), alias="1_dG")
    field_2_dG: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=0, higher=False), alias="2_dG")
    field_3_dG: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=None, higher=False), alias="3_dG")
    field_4_dG: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=None, higher=False), alias="4_dG")
    field_5_dG: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=None, higher=False), alias="5_dG")
    Average_dSASA: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=1, higher=True))
    field_1_dSASA: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=1, higher=True), alias="1_dSASA")
    field_2_dSASA: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=1, higher=True), alias="2_dSASA")
    field_3_dSASA: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=None, higher=True), alias="3_dSASA")
    field_4_dSASA: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=None, higher=True), alias="4_dSASA")
    field_5_dSASA: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=None, higher=True), alias="5_dSASA")
    field_Average_dG_dSASA: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=None, higher=False), alias="Average_dG/dSASA")
    field_1_dG_dSASA: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=None, higher=False), alias="1_dG/dSASA")
    field_2_dG_dSASA: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=None, higher=False), alias="2_dG/dSASA")
    field_3_dG_dSASA: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=None, higher=False), alias="3_dG/dSASA")
    field_4_dG_dSASA: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=None, higher=False), alias="4_dG/dSASA")
    field_5_dG_dSASA: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=None, higher=False), alias="5_dG/dSASA")
    field_Average_Interface_SASA_percent: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=None, higher=True), alias="Average_Interface_SASA_%")
    field_1_Interface_SASA_percent: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=None, higher=True), alias="1_Interface_SASA_%")
    field_2_Interface_SASA_percent: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=None, higher=True), alias="2_Interface_SASA_%")
    field_3_Interface_SASA_percent: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=None, higher=True), alias="3_Interface_SASA_%")
    field_4_Interface_SASA_percent: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=None, higher=True), alias="4_Interface_SASA_%")
    field_5_Interface_SASA_percent: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=None, higher=True), alias="5_Interface_SASA_%")
    Average_Interface_Hydrophobicity: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=None, higher=True))
    field_1_Interface_Hydrophobicity: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=None, higher=True), alias="1_Interface_Hydrophobicity")
    field_2_Interface_Hydrophobicity: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=None, higher=True), alias="2_Interface_Hydrophobicity")
    field_3_Interface_Hydrophobicity: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=None, higher=True), alias="3_Interface_Hydrophobicity")
    field_4_Interface_Hydrophobicity: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=None, higher=True), alias="4_Interface_Hydrophobicity")
    field_5_Interface_Hydrophobicity: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=None, higher=True), alias="5_Interface_Hydrophobicity")
    Average_n_InterfaceResidues: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=7, higher=True))
    field_1_n_InterfaceResidues: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=7, higher=True), alias="1_n_InterfaceResidues")
    field_2_n_InterfaceResidues: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=7, higher=True), alias="2_n_InterfaceResidues")
    field_3_n_InterfaceResidues: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=None, higher=True), alias="3_n_InterfaceResidues")
    field_4_n_InterfaceResidues: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=None, higher=True), alias="4_n_InterfaceResidues")
    field_5_n_InterfaceResidues: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=None, higher=True), alias="5_n_InterfaceResidues")
    Average_n_InterfaceHbonds: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=3, higher=True))
    field_1_n_InterfaceHbonds: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=3, higher=True), alias="1_n_InterfaceHbonds")
    field_2_n_InterfaceHbonds: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=3, higher=True), alias="2_n_InterfaceHbonds")
    field_3_n_InterfaceHbonds: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=None, higher=True), alias="3_n_InterfaceHbonds")
    field_4_n_InterfaceHbonds: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=None, higher=True), alias="4_n_InterfaceHbonds")
    field_5_n_InterfaceHbonds: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=None, higher=True), alias="5_n_InterfaceHbonds")
    Average_InterfaceHbondsPercentage: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=None, higher=True))
    field_1_InterfaceHbondsPercentage: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=None, higher=True), alias="1_InterfaceHbondsPercentage")
    field_2_InterfaceHbondsPercentage: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=None, higher=True), alias="2_InterfaceHbondsPercentage")
    field_3_InterfaceHbondsPercentage: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=None, higher=True), alias="3_InterfaceHbondsPercentage")
    field_4_InterfaceHbondsPercentage: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=None, higher=True), alias="4_InterfaceHbondsPercentage")
    field_5_InterfaceHbondsPercentage: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=None, higher=True), alias="5_InterfaceHbondsPercentage")
    Average_n_InterfaceUnsatHbonds: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=4, higher=False))
    field_1_n_InterfaceUnsatHbonds: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=4, higher=False), alias="1_n_InterfaceUnsatHbonds")
    field_2_n_InterfaceUnsatHbonds: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=4, higher=False), alias="2_n_InterfaceUnsatHbonds")
    field_3_n_InterfaceUnsatHbonds: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=None, higher=False), alias="3_n_InterfaceUnsatHbonds")
    field_4_n_InterfaceUnsatHbonds: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=None, higher=False), alias="4_n_InterfaceUnsatHbonds")
    field_5_n_InterfaceUnsatHbonds: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=None, higher=False), alias="5_n_InterfaceUnsatHbonds")
    Average_InterfaceUnsatHbondsPercentage: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=None, higher=False))
    field_1_InterfaceUnsatHbondsPercentage: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=None, higher=False), alias="1_InterfaceUnsatHbondsPercentage")
    field_2_InterfaceUnsatHbondsPercentage: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=None, higher=False), alias="2_InterfaceUnsatHbondsPercentage")
    field_3_InterfaceUnsatHbondsPercentage: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=None, higher=False), alias="3_InterfaceUnsatHbondsPercentage")
    field_4_InterfaceUnsatHbondsPercentage: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=None, higher=False), alias="4_InterfaceUnsatHbondsPercentage")
    field_5_InterfaceUnsatHbondsPercentage: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=None, higher=True), alias="5_InterfaceUnsatHbondsPercentage")
    field_Average_Interface_Helix_percent: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=None, higher=True), alias="Average_Interface_Helix%")
    field_1_Interface_Helix_percent: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=None, higher=True), alias="1_Interface_Helix%")
    field_2_Interface_Helix_percent: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=None, higher=True), alias="2_Interface_Helix%")
    field_3_Interface_Helix_percent: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=None, higher=True), alias="3_Interface_Helix%")
    field_4_Interface_Helix_percent: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=None, higher=True), alias="4_Interface_Helix%")
    field_5_Interface_Helix_percent: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=None, higher=True), alias="5_Interface_Helix%")
    field_Average_Interface_BetaSheet_percent: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=None, higher=True), alias="Average_Interface_BetaSheet%")
    field_1_Interface_BetaSheet_percent: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=None, higher=True), alias="1_Interface_BetaSheet%")
    field_2_Interface_BetaSheet_percent: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=None, higher=True), alias="2_Interface_BetaSheet%")
    field_3_Interface_BetaSheet_percent: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=None, higher=True), alias="3_Interface_BetaSheet%")
    field_4_Interface_BetaSheet_percent: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=None, higher=True), alias="4_Interface_BetaSheet%")
    field_5_Interface_BetaSheet_percent: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=None, higher=True), alias="5_Interface_BetaSheet%")
    field_Average_Interface_Loop_percent: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=None, higher=False), alias="Average_Interface_Loop%")
    field_1_Interface_Loop_percent: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=None, higher=False), alias="1_Interface_Loop%")
    field_2_Interface_Loop_percent: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=None, higher=False), alias="2_Interface_Loop%")
    field_3_Interface_Loop_percent: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=None, higher=False), alias="3_Interface_Loop%")
    field_4_Interface_Loop_percent: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=None, higher=False), alias="4_Interface_Loop%")
    field_5_Interface_Loop_percent: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=None, higher=False), alias="5_Interface_Loop%")
    field_Average_Binder_Helix_percent: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=None, higher=True), alias="Average_Binder_Helix%")
    field_1_Binder_Helix_percent: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=None, higher=True), alias="1_Binder_Helix%")
    field_2_Binder_Helix_percent: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=None, higher=True), alias="2_Binder_Helix%")
    field_3_Binder_Helix_percent: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=None, higher=True), alias="3_Binder_Helix%")
    field_4_Binder_Helix_percent: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=None, higher=True), alias="4_Binder_Helix%")
    field_5_Binder_Helix_percent: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=None, higher=True), alias="5_Binder_Helix%")
    field_Average_Binder_BetaSheet_percent: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=None, higher=True), alias="Average_Binder_BetaSheet%")
    field_1_Binder_BetaSheet_percent: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=None, higher=True), alias="1_Binder_BetaSheet%")
    field_2_Binder_BetaSheet_percent: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=None, higher=True), alias="2_Binder_BetaSheet%")
    field_3_Binder_BetaSheet_percent: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=None, higher=True), alias="3_Binder_BetaSheet%")
    field_4_Binder_BetaSheet_percent: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=None, higher=True), alias="4_Binder_BetaSheet%")
    field_5_Binder_BetaSheet_percent: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=None, higher=True), alias="5_Binder_BetaSheet%")
    field_Average_Binder_Loop_percent: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=90, higher=False), alias="Average_Binder_Loop%")
    field_1_Binder_Loop_percent: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=90, higher=False), alias="1_Binder_Loop%")
    field_2_Binder_Loop_percent: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=90, higher=False), alias="2_Binder_Loop%")
    field_3_Binder_Loop_percent: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=None, higher=False), alias="3_Binder_Loop%")
    field_4_Binder_Loop_percent: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=None, higher=False), alias="4_Binder_Loop%")
    field_5_Binder_Loop_percent: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=None, higher=False), alias="5_Binder_Loop%")
    Average_InterfaceAAs: AminoAcidThresholds = Field(default_factory=AminoAcidThresholds)
    field_1_InterfaceAAs: AminoAcidThresholds = Field(default_factory=AminoAcidThresholds, alias="1_InterfaceAAs")
    field_2_InterfaceAAs: AminoAcidThresholds = Field(default_factory=AminoAcidThresholds, alias="2_InterfaceAAs")
    field_3_InterfaceAAs: AminoAcidThresholds = Field(default_factory=AminoAcidThresholds, alias="3_InterfaceAAs")
    field_4_InterfaceAAs: AminoAcidThresholds = Field(default_factory=AminoAcidThresholds, alias="4_InterfaceAAs")
    field_5_InterfaceAAs: AminoAcidThresholds = Field(default_factory=AminoAcidThresholds, alias="5_InterfaceAAs")
    Average_Hotspot_RMSD: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=6, higher=False))
    field_1_Hotspot_RMSD: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=6, higher=False), alias="1_Hotspot_RMSD")
    field_2_Hotspot_RMSD: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=6, higher=False), alias="2_Hotspot_RMSD")
    field_3_Hotspot_RMSD: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=None, higher=False), alias="3_Hotspot_RMSD")
    field_4_Hotspot_RMSD: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=None, higher=False), alias="4_Hotspot_RMSD")
    field_5_Hotspot_RMSD: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=None, higher=False), alias="5_Hotspot_RMSD")
    Average_Target_RMSD: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=None, higher=False))
    field_1_Target_RMSD: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=None, higher=False), alias="1_Target_RMSD")
    field_2_Target_RMSD: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=None, higher=False), alias="2_Target_RMSD")
    field_3_Target_RMSD: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=None, higher=False), alias="3_Target_RMSD")
    field_4_Target_RMSD: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=None, higher=False), alias="4_Target_RMSD")
    field_5_Target_RMSD: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=None, higher=False), alias="5_Target_RMSD")
    Average_Binder_pLDDT: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=0.8, higher=True))
    field_1_Binder_pLDDT: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=0.8, higher=True), alias="1_Binder_pLDDT")
    field_2_Binder_pLDDT: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=0.8, higher=True), alias="2_Binder_pLDDT")
    field_3_Binder_pLDDT: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=0.8, higher=True), alias="3_Binder_pLDDT")
    field_4_Binder_pLDDT: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=0.8, higher=True), alias="4_Binder_pLDDT")
    field_5_Binder_pLDDT: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=0.8, higher=True), alias="5_Binder_pLDDT")
    Average_Binder_pTM: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=None, higher=True))
    field_1_Binder_pTM: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=None, higher=True), alias="1_Binder_pTM")
    field_2_Binder_pTM: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=None, higher=True), alias="2_Binder_pTM")
    field_3_Binder_pTM: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=None, higher=True), alias="3_Binder_pTM")
    field_4_Binder_pTM: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=None, higher=True), alias="4_Binder_pTM")
    field_5_Binder_pTM: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=None, higher=True), alias="5_Binder_pTM")
    Average_Binder_pAE: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=None, higher=False))
    field_1_Binder_pAE: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=None, higher=False), alias="1_Binder_pAE")
    field_2_Binder_pAE: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=None, higher=False), alias="2_Binder_pAE")
    field_3_Binder_pAE: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=None, higher=False), alias="3_Binder_pAE")
    field_4_Binder_pAE: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=None, higher=False), alias="4_Binder_pAE")
    field_5_Binder_pAE: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=None, higher=False), alias="5_Binder_pAE")
    Average_Binder_RMSD: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=3.5, higher=False))
    field_1_Binder_RMSD: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=3.5, higher=False), alias="1_Binder_RMSD")
    field_2_Binder_RMSD: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=3.5, higher=False), alias="2_Binder_RMSD")
    field_3_Binder_RMSD: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=3.5, higher=False), alias="3_Binder_RMSD")
    field_4_Binder_RMSD: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=3.5, higher=False), alias="4_Binder_RMSD")
    field_5_Binder_RMSD: ThresholdConfig = Field(default_factory=lambda: ThresholdConfig(threshold=3.5, higher=False), alias="5_Binder_RMSD")

    class Config:
        populate_by_name = True


class AdvancedSettings(BaseModel):
    omit_AAs: str = "C"
    force_reject_AA: bool = False
    use_multimer_design: bool = True
    design_algorithm: str = "4stage"
    sample_models: bool = True
    rm_template_seq_design: bool = False
    rm_template_seq_predict: bool = False
    rm_template_sc_design: bool = False
    rm_template_sc_predict: bool = False
    predict_initial_guess: bool = False
    predict_bigbang: bool = False
    soft_iterations: int = 75
    temporary_iterations: int = 45
    hard_iterations: int = 5
    greedy_iterations: int = 15
    greedy_percentage: int = 1
    save_design_animations: bool = True
    save_design_trajectory_plots: bool = True
    weights_plddt: float = 0.1
    weights_pae_intra: float = 0.4
    weights_pae_inter: float = 0.1
    weights_con_intra: float = 1.0
    weights_con_inter: float = 1.0
    intra_contact_distance: float = 14.0
    inter_contact_distance: float = 20.0
    intra_contact_number: int = 2
    inter_contact_number: int = 2
    weights_helicity: float = -0.3
    random_helicity: bool = False
    use_i_ptm_loss: bool = True
    weights_iptm: float = 0.05
    use_rg_loss: bool = True
    weights_rg: float = 0.3
    use_termini_distance_loss: bool = False
    weights_termini_loss: float = 0.1
    enable_mpnn: bool = True
    mpnn_fix_interface: bool = True
    num_seqs: int = 20
    max_mpnn_sequences: int = 2
    sampling_temp: float = 0.1
    backbone_noise: float = 0.00
    model_path: str = "v_48_020"
    mpnn_weights: str = "soluble"
    save_mpnn_fasta: bool = False
    num_recycles_design: int = 1
    num_recycles_validation: int = 3
    optimise_beta: bool = True
    optimise_beta_extra_soft: int = 0
    optimise_beta_extra_temp: int = 0
    optimise_beta_recycles_design: int = 3
    optimise_beta_recycles_valid: int = 3
    remove_unrelaxed_trajectory: bool = True
    remove_unrelaxed_complex: bool = True
    remove_binder_monomer: bool = True
    zip_animations: bool = True
    zip_plots: bool = True
    save_trajectory_pickle: bool = False
    max_trajectories: bool = False
    enable_rejection_check: bool = True
    acceptance_rate: float = 0.01
    start_monitoring: int = 600
    af_params_dir: str = "/content/RLBind/bindcraft"
    dssp_path: str = ""
    dalphaball_path: str = ""
