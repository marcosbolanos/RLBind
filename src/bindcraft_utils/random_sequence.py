import random

_aa_supported = [
    "A",
    "R",
    "N",
    "D",
    "C",
    "Q",
    "E",
    "G",
    "H",
    "I",
    "L",
    "K",
    "M",
    "F",
    "P",
    "S",
    "T",
    "W",
    "Y",
    "V",
]


def random_aa_sequence(length: int | None):
    """
    Generate a random sequence of valid aminoacids
    Inputs: length (optional, will be randomly assigned if not given

    Outputs: a sequence of random aminoacids
    """
    if not length:
        length = random.randint(30, 100)

    AA_sequence = ""

    for i in range(length):
        n_aas = len(_aa_supported)
        aa_index = random.randint(0, n_aas - 1)

        random_aa = _aa_supported[aa_index]
        AA_sequence += random_aa
    return AA_sequence

