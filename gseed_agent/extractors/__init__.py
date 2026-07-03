from .material import extract as extract_material
from .energy import extract as extract_energy
from .landscape import extract as extract_landscape
from .construction import extract as extract_construction
from .drawing import extract as extract_drawing


EXTRACTORS = {
    "material": extract_material,
    "energy": extract_energy,
    "landscape": extract_landscape,
    "construction": extract_construction,
    "drawing": extract_drawing,
}

