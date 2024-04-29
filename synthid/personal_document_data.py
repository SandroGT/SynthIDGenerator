"""
Each personal document reports a set of personal and non-personal information, that may also be unique of that document.
We identify each information with a key (like "name", "surname", "age", "gender", "ID number", ...).
This module provides a function that given the key retrieves a random value to be used when compiling that field. This
 value is either a string or an image (PIL.Image).
Which keys are of interest for each document can be found in the template folder of the same document.
Actually, we do not care about consistency, and it's not a problem to have a male name with a female picture and female
 gender, so these functions don't need any input to operate with constraints.
"""

from __future__ import annotations
from pathlib import Path
from PIL import Image
import random
import re
import string
import sys

from synthid.settings import (TEMPLATES_PERSON_PICTURES, TEMPLATES_PERSON_NAMES_DICT,
                              TEMPLATES_DOCUMENT_FRONT_FILE_NAME, TEMPLATES_DOCUMENT_BACK_FILE_NAME,
                              TEMPLATES_PERSON_RESIDENCE_DICT, TEMPLATES_WORDS_LIST)


def load_template_image(doc_type_path: Path) -> tuple[Image, Image]:
    template_front_image_path = doc_type_path.joinpath(TEMPLATES_DOCUMENT_FRONT_FILE_NAME)
    template_back_image_path = doc_type_path.joinpath(TEMPLATES_DOCUMENT_BACK_FILE_NAME)
    template_front_image = Image.open(template_front_image_path) if template_front_image_path.exists() else None
    template_back_image = Image.open(template_back_image_path) if template_back_image_path.exists() else None
    return template_front_image, template_back_image


def retrieve_random_data(key: str) -> str | Image:
    """Retrieves a random value for the field identified by key."""
    search_key = key.replace(' ', '_').replace('-', '_').lower()
    field_retrieval_function = getattr(sys.modules[__name__], f'get_random_{search_key}', None)
    if field_retrieval_function is None:
        raise ValueError(f'The key "{key}" is not a registered key with a retrieval function.')
    else:
        return field_retrieval_function()


def get_random_date() -> str:
    """Get a random date."""
    # !!! Do not care about consistency, and may return 31st the of February.
    day = random.randint(1, 31)
    month = random.randint(1, 12)
    year = random.randint(2010, 2030)
    return f'{day:02}/{month:02}/{year:04}'


def get_random_expiration_date() -> str:
    """Get a random date."""
    return get_random_date()


def get_random_emission_date() -> str:
    """Get a random date."""
    return get_random_date()


def get_random_name() -> str:
    """Get a random name from the JSON file of names."""
    return random.choice(TEMPLATES_PERSON_NAMES_DICT['names'])


def get_random_surname() -> str:
    """Get a random surname from the JSON file of names."""
    return random.choice(TEMPLATES_PERSON_NAMES_DICT['surnames'])


def get_random_face_picture() -> Image:
    """Get a random face picture from the dataset of face pictures."""
    return Image.open(random.choice(TEMPLATES_PERSON_PICTURES)).convert('RGBA')


def get_random_gender() -> str:
    """Get a random gender, either 'M' or 'F'. LGBTQ+ community may hate us, but we need to avoid complexity."""
    return random.choice(['M', 'F'])


def get_random_date_of_birth() -> str:
    """Get a random birthday date"""
    return get_random_date()


def get_random_birth_certificate_details() -> str:
    n1 = random_digits(3)
    n2 = random_digits(4)
    n3 = random_digits(6)
    s1 = random_letters(2, lower=True, upper=True)
    s2 = random_letters(2, lower=True, upper=True)
    return f'{n1} {s1} {s2} - {n2} {n3}'


def get_random_place_and_date_of_birth() -> str:
    return f'{get_random_commune()} {get_random_date_of_birth()}'


def get_random_act_n() -> str:
    return str(random.randint(1, 300))


def get_random_act_p() -> str:
    characters = string.ascii_letters + string.digits
    choice = random.randint(0, 1)
    if choice == 0:
        return str(random.randint(1, 10))
    if choice == 1:
        return random.choice(characters)


def get_random_act_s() -> str:
    characters = string.ascii_letters + string.digits
    return random.choice(characters)


def get_random_citizenship() -> str:
    return 'italian'


def get_random_nationality_short() -> str:
    return random.sample([
        'ITA', 'USA', 'GER', 'FRA', 'ESP', 'GBR', 'CAN', 'AUS', 'JPN', 'CHN', 'BRA', 'RUS', 'IND', 'MEX', 'ARG', 'SWE'
    ], 1)[0]


def get_random_commune() -> str:
    return random_geographic_info(commune=True, province=False)


def get_random_commune_with_province() -> str:
    return random_geographic_info(commune=True, province=False)


def get_random_residence() -> str:
    return random_geographic_info(commune=True, province=True)


def get_random_place_of_birth() -> str:
    return random_geographic_info(commune=True, province=True)


def get_random_town() -> str:
    return random_geographic_info(commune=True, province=False)


def get_random_province() -> str:
    return random_geographic_info(commune=False, province=True)


def get_random_address() -> str:
    def random_address_type() -> str:
        return random.choice([
            '', 'Via', 'Viale', 'Vicolo', 'V.', 'Largo', 'Corso',
            'Piazza', 'P.zza', 'Piazzale', 'P.zzale', 'P.',
            'Borgo', 'Rione', 'Localita\'', 'Località'
        ])

    def random_address_name() -> str:
        choice = random.randint(0, 3)
        if choice == 0:  # Random word
            return random.choice(TEMPLATES_WORDS_LIST)
        elif choice == 1:  # Random name
            return get_random_name()
        elif choice == 2:  # Random surname
            return get_random_surname()
        elif choice == 3:  # Random name and surname
            return f'{get_random_name()} {get_random_surname()}'
        else:
            assert False

    def random_civic_number() -> str:
        if random.uniform(0, 1) < 0.02:  # Probability of no-civic number
            return random.choice(['SNC', 'snc', '/SNC', '/snc'])
        else:
            civic_number = random.randint(1, 500)
            if random.uniform(0, 1) < 0.15:  # Probability of adding a letter
                civic_number_letter = random_letters(1, lower=True, upper=True)
                if random.uniform(0, 1) < 0.50:  # Probability of adding some letter separator
                    letter_separator = \
                        ' ' if random.uniform(0, 1) < 0.50 else '' + \
                        '/' + \
                        ' ' if random.uniform(0, 1) < 0.50 else ''
                    civic_number_letter = f'{letter_separator}{civic_number_letter}'
                civic_number = f'{civic_number}{civic_number_letter}'
            if random.uniform(0, 1) < 0.15:  # Probability of adding some other reference
                other_ref_type = random.choice(['Piano', 'P', 'P.', 'Corte', 'C.', 'Appartamento', 'App.'])
                other_ref_num = random.randint(1, 20)
                other_ref = f'{other_ref_type} {other_ref_num}'
                civic_number = f'{civic_number} {other_ref}'
            return str(civic_number)

    add_type = random_address_type()
    add_name = random_address_name()
    add_cnum = random_civic_number()
    sep = random.choice([' ', ', '])

    do_name_upper = random.uniform(0, 1) < 0.50
    if do_name_upper:
        do_all_upper = random.uniform(0, 1) < 0.50
    else:
        do_all_upper = False

    add_type = add_type.upper() if do_all_upper else add_type
    add_name = add_name.upper() if do_name_upper else add_name
    add_cnum = add_cnum.upper() if do_all_upper else add_cnum

    return f'{add_type} {add_name}{sep}{add_cnum}'


def get_random_complete_address() -> str:
    base_address = get_random_address()
    residence = get_random_residence()
    residence = residence.upper() if random.uniform(0, 1) < 0.50 else residence
    return f'{base_address}, {residence}'


def get_random_civil_status() -> str:
    status = ['Celibe/Nubile', 'Coniugato/a', 'Vedovo/a', 'Divorzato/a', '****']
    return random.choice(status)


def get_random_profession() -> str:
    professions = [
        'Studente', 'Ricercatore', 'Medico', 'Contabile', 'Attore', 'Architetto', 'Artista', 'Businessman',
        'Chef', 'Dentista', 'Infermiere', 'Contadino', 'Elettricista', 'Artigiano', 'Libero Professionista', '****']
    return random.choice(professions)


def get_random_height_cm() -> str:
    return f'{get_random_height_cm_no_measure()}cm'


def get_random_height_cm_no_measure() -> str:
    return str(random.randint(145, 210))


def get_random_hair() -> str:
    hair_types = ['neri', 'biondi', 'castani', 'brizzolati']
    return random.choice(hair_types)


def get_random_eyes() -> str:
    eyes_color = ['verdi', 'blu', 'grigi', 'castani']
    return random.choice(eyes_color)


def get_random_particular_signs() -> str:
    return '***'


def get_random_releasing_entity() -> str:
    place = random_geographic_info(commune=True, province=True)
    while len(place) > 14:
        place = random_geographic_info(commune=True, province=True)

    match = re.findall(r'\((.*?)\)', place)[0]
    place = place.replace(f'({match})', '')
    return place


def get_random_release_date() -> str:
    return get_random_date()


def get_random_commune() -> str:
    return random_geographic_info(commune=True, province=True)


def get_random_card_n() -> str:
    characters = string.ascii_letters
    random_char = random.choice(characters)
    random_numbers = ''.join([str(random.randint(1, 9)) for i in range(7)])
    return f'A{random_char}{random_numbers}'


def get_random_sex() -> str:
    return random.choice(['M', 'F'])


def get_random_digital_id_code() -> str:
    characters = string.ascii_letters
    random_chars = ''.join([random.choice(characters) for i in range(2)])
    random_numbers = ''.join([str(random.randint(1, 9)) for i in range(5)])
    return f'CA{random_numbers}{random_chars}'


def get_random_cie_back_codes() -> str:
    name = get_random_name().upper()
    surname = get_random_surname().upper()
    line1 = f'{random_letters(1, upper=True)}' \
            f'{"<"}' \
            f'{random_letters(5, upper=True)}' \
            f'{random_digits(5)}' \
            f'{random_letters(2, upper=True)}' \
            f'{random_digits(1)}' \
            f'{"<" * 15}'
    line2 = f'{random_digits(7)}' \
            f'{random_letters(1, upper=True)}' \
            f'{random_digits(7)}' \
            f'{random_letters(3, upper=True)}' \
            f'{"<" * 11}' \
            f'{random_digits(1)}'
    line3 = f'{surname.replace(" ", "<")}' \
            f'{"<<"}' \
            f'{name.replace(" ", "<")}' \
            f'{"<" * max(0, 30 - len(name) - len(surname) - 2)}' \
            [:30]
    return f'{line1}\n{line2}\n{line3}'


def get_random_digital_id_number() -> str:
    return ''.join([str(random.randint(1, 9)) for i in range(6)])


def get_random_fiscal_code() -> str:
    characters = string.ascii_letters
    code = ''
    for i in range(16):
        if i < 6:
            code += random.choice(characters)
        if (6 <= i < 8) or (i == 9):
            code += str(random.randint(1, 9))
        if (i == 8) or (i == 10):
            code += random.choice(characters)
        if i > 10:
            choice = random.randint(0, 1)
            if choice == 1:
                code += random.choice(characters)
            else:
                code += str(random.randint(1, 9))
    return code.upper()


def get_random_drv_card_id() -> str:
    characters = string.ascii_letters + string.digits
    random_chars = ''.join([random.choice(characters) for _ in range(10)])
    return random_chars


def get_random_releasing_entity_drv() -> str:
    chars = 'MIT-UCO'
    if random.randint(0, 1) == 0:
        return chars
    random_province = ''.join([random.choice(string.ascii_letters) for _ in range(2)])
    return f'MC-{random_province}'


def get_random_hc_institution_id_number() -> str:
    return 'SSN-MIN SALUTE - 500001'


def get_random_hc_card_id_number() -> str:
    return str(random_digits(20))


def random_digits(n_digits: int) -> int:
    return random.randint(10**(n_digits-1), 10**n_digits-1)


def random_letters(n_letters: int, lower: bool = False, upper: bool = False) -> str:
    letters_set = []
    if lower:
        letters_set += string.ascii_letters
    if upper:
        letters_set += string.ascii_letters.upper()
    return ''.join([random.sample(letters_set, 1)[0] for _ in range(n_letters)])


def random_geographic_info(commune: bool = False, province: bool = False) -> str:
    data = TEMPLATES_PERSON_RESIDENCE_DICT

    n_region = random.randint(0, len(data['regioni']) - 1)
    province_data = data['regioni'][n_region]['province']

    n_province = random.randint(0, len(province_data) - 1)
    n_commune = random.randint(0, len(province_data[n_province]['comuni']) - 1)

    commune_name = province_code = None
    if commune:
        commune_name = province_data[n_province]['comuni'][n_commune]['nome']
    if province:
        province_code = province_data[n_province]['code']

    if commune and province:
        return f'{commune_name} ({province_code})'
    elif commune:
        return commune_name
    elif province:
        return province_code
    else:
        raise ValueError('At least one of `commune` and `province` should be `True`')


# TODO add all the other methods for all the other needed information
...
