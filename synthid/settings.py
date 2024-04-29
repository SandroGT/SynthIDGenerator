from pathlib import Path


# Dataset generics

SIZE_DOCUMENT_SAMPLES: int = 1000
"""Target number of samples to create for each personal document type."""

SIZE_DATASET: int = 1
"""Target number of images for the final dataset."""

OUTPUT_EXTENSION: str = 'PNG'
"""Output extension of images."""

MAX_CONCURRENT_DOCS: int = 3
"""Maximum number of personal documents appearing in the same page."""

DOCUMENT_LABELS: dict = {
    'none': 0,
}
"""Document labels in the mask/annotation image. Other labels are dynamically created in this setting initialization,
 according to the type of documents stored in the template folders."""

# Templates data paths

TEMPLATES_PATH: Path = Path(__file__).parent.joinpath('templates').resolve()
"""Location of the folder with the base personal documents and other templates."""

TEMPLATES_DOCUMENTS_PATH: Path = TEMPLATES_PATH.joinpath('documents')
"""Location of the folder with the personal documents templates."""

TEMPLATES_PERSON_PICTURES_PATH: Path = TEMPLATES_PATH.joinpath('face_pictures').resolve()
"""Location of the folder with face pictures."""

TEMPLATES_FONTS_PATH: Path = TEMPLATES_PATH.joinpath('fonts').resolve()
"""Location of the folder with custom fonts."""

TEMPLATES_PERSON_NAMES_PATH: Path = TEMPLATES_PATH.joinpath('names.json').resolve()
"""Location of the JSON file with names and surnames."""
TEMPLATES_PERSON_RESIDENCE_PATH: Path = TEMPLATES_PATH.joinpath("communes.json").resolve()
"""Location of the JSON file with italian communes"""
TEMPLATES_WORDS_PATH: Path = TEMPLATES_PATH.joinpath('words.txt').resolve()
"""Location of the TXT file with italian words."""
TEMPLATES_DOCUMENT_DATA_FILE_NAME: str = 'compilation_data.json'
"""Name of the file containing the data needed to compile a personal document template in the form of a dictionary.
The first key distinguishes between the front ('front') or back ('back') of the personal document. The following keys
identify the type of information (name, surname, gender, face picture, ...) along with other formatting settings."""

TEMPLATE_SIDES_NAMES = ['front', 'back']
"""Names/keys used to refer to each side of a personal document."""

TEMPLATES_DOCUMENT_FRONT_FILE_NAME: str = f'{TEMPLATE_SIDES_NAMES[0]}.png'
"""The name of the template file of the front of a personal document."""

TEMPLATES_DOCUMENT_BACK_FILE_NAME: str = f'{TEMPLATE_SIDES_NAMES[1]}.png'
"""The name of the template file of the back of a personal document."""


# Created data paths

CREATED_DOCS_PATH: Path = Path(__file__).parent.joinpath('synth_documents').resolve()
"""Location of the folder with the procedurally created personal documents."""

DOCUMENT_LABELS_OUTPUT_PATH: Path = CREATED_DOCS_PATH.joinpath('labels.json')
"""Location of the JSON dictionary with the corresponding document labels in the mask/annotation image."""


# Custom data automatically loaded
TEMPLATES_PERSON_NAMES_DICT: dict = dict()
"""Dictionary of the JSON file with names and surnames."""
TEMPLATES_PERSON_RESIDENCE_DICT: dict = dict()
"""Dictionary of the JSON file with italian communes."""
TEMPLATES_WORDS_LIST: list = list()
"""List of words from the TXT file with italian words."""
TEMPLATES_PERSON_PICTURES: list[Path]
"""List of face pictures file paths."""


# Initialization

def init():
    import json
    import shutil

    global DOCUMENT_LABELS, TEMPLATES_PERSON_NAMES_DICT, TEMPLATES_PERSON_RESIDENCE_DICT, TEMPLATES_WORDS_LIST, \
        TEMPLATES_PERSON_PICTURES
    # Clear output folders
    if CREATED_DOCS_PATH.exists():
        shutil.rmtree(CREATED_DOCS_PATH)
    # Create base output folders
    CREATED_DOCS_PATH.mkdir()
    # Initialize document labels dynamically
    i = 1
    for doc_path in TEMPLATES_DOCUMENTS_PATH.iterdir():
        for doc_side in TEMPLATE_SIDES_NAMES:
            doc_k = f'{doc_path.stem}-{doc_side}'
            DOCUMENT_LABELS[doc_k] = i
            i += 1
    with DOCUMENT_LABELS_OUTPUT_PATH.open('w', encoding='utf8') as f:
        json.dump(DOCUMENT_LABELS, f)
    with TEMPLATES_PERSON_NAMES_PATH.open('r', encoding='utf8') as f:
        TEMPLATES_PERSON_NAMES_DICT = json.load(f)
    with TEMPLATES_PERSON_RESIDENCE_PATH.open('r', encoding='utf8') as f:
        TEMPLATES_PERSON_RESIDENCE_DICT = json.load(f)
    with TEMPLATES_WORDS_PATH.open('r', encoding='utf8') as f:
        TEMPLATES_WORDS_LIST = f.read().split('\n')
    TEMPLATES_PERSON_PICTURES = [
        p for p in TEMPLATES_PERSON_PICTURES_PATH.iterdir()
        if p.suffix.lower() in {'.jpg', '.png'}
    ]


init()
