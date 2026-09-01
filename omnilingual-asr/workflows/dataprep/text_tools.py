# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import re
import unicodedata

import norm_config_module as norm_config_module
from unidecode import unidecode

norm_config = norm_config_module.norm_config  # type: ignore


def text_normalize(
    text, iso_code, lower_case=True, remove_numbers=True, remove_brackets=False
):
    """Given a text, normalize it by changing to lower case, removing punctuations, removing words that only contain digits and removing extra spaces

    Args:
        text : The string to be normalized
        iso_code :
        remove_numbers : Boolean flag to specify if words containing only digits should be removed

    Returns:
        normalized_text : the string after all normalization

    """

    config = norm_config.get(iso_code, norm_config["*"])

    for field in [
        "lower_case",
        "punc_set",
        "del_set",
        "mapping",
        "digit_set",
        "unicode_norm",
    ]:
        if field not in config:
            config[field] = norm_config["*"][field]

    text = unicodedata.normalize(config["unicode_norm"], text)

    # Convert to lower case

    if config["lower_case"] and lower_case:
        text = text.lower()

    # brackets

    # always text inside brackets with numbers in them. Usually corresponds to "(Sam 23:17)"
    text = re.sub(r"\([^\)]*\d[^\)]*\)", " ", text)
    if remove_brackets:
        text = re.sub(r"\([^\)]*\)", " ", text)

    # Apply mappings

    for old, new in config["mapping"].items():
        text = re.sub(old, new, text)

    # Replace punctutations with space

    punct_pattern = r"[" + config["punc_set"]

    punct_pattern += "]"

    normalized_text = re.sub(punct_pattern, " ", text)

    # remove characters in delete list

    delete_patten = r"[" + config["del_set"] + "]"

    normalized_text = re.sub(delete_patten, "", normalized_text)

    # Remove words containing only digits
    # We check for 3 cases  a)text starts with a number b) a number is present somewhere in the middle of the text c) the text ends with a number
    # For each case we use lookaround regex pattern to see if the digit pattern in preceded and followed by whitespaces, only then we replace the numbers with space
    # The lookaround enables overlapping pattern matches to be replaced

    if remove_numbers:

        digits_pattern = "[" + config["digit_set"]

        digits_pattern += "]+"

        complete_digit_pattern = (
            r"^"
            + digits_pattern
            + r"(?=\s)|(?<=\s)"
            + digits_pattern
            + r"(?=\s)|(?<=\s)"
            + digits_pattern
            + "$"
        )

        normalized_text = re.sub(complete_digit_pattern, " ", normalized_text)

    if config["rm_diacritics"]:
        normalized_text = unidecode(normalized_text)

    # Remove extra spaces
    normalized_text = re.sub(r"\s+", " ", normalized_text).strip()

    return normalized_text

def _lowercase_sentence_initial(match):
    """Lowercase a sentence-initial letter.

    Keeps the letter uppercase when the next letter is also uppercase, which marks
    the start of an acronym or initialism.

    Input: a `re.Match` from the SENTENCE_INITIAL pattern with three groups:
        1. sentence delimiter (start-of-string or `.!?—` plus optional space)
        2. first letter of the sentence
        3. next letter (may be empty)
    Output: the joined string with group 2 possibly lowercased.
    """
    delimiter, first, second = match.group(1), match.group(2), match.group(3)
    if first.isupper() and not (second and second.isupper()):
        first = first.lower()
    return delimiter + first + second


def normalize_text_mozilla(text, is_lower=True):
    """Normalize a transcript to a canonical form for scoring.

    Removes bracketed annotations (`[...]`), unintelligible markers (`(?)`),
    parenthetical wrappers, and stray punctuation (`¿¡";:!?`). Lowercases
    sentence-initial letters. Rewrites em dashes as commas, then drops commas
    and periods (but preserves `...`). Collapses runs of whitespace.

    Input: a raw transcript string.
    Output: the cleaned transcript string.
    """
    if is_lower:
        text = text.lower()
    BRACKETED = re.compile(r"\[[^\]]+\]")
    UNINTELLIGIBLE_PAREN = re.compile(r"\(\?+\)")
    WORD_PAREN = re.compile(r"\(([^()]*)\)")
    PUNCTUATION_OTHER = re.compile('[¿¡";:]+')
    COMMA = re.compile(",+")
    # [^\W\d_] matches any Unicode letter (word char minus digits/underscore),
    # including sentence-initial accented capitals (e.g. Spanish Á/É/Í/Ó/Ú/Ñ)
    SENTENCE_INITIAL = re.compile(r"(^\s*|[.!?—]\s*)([^\W\d_])([^\W\d_]?)")
    SENTENCE_END = re.compile("[!?]+")
    MULTISPACE = re.compile("  +")

    text = text.replace("~", "")
    text = re.sub(BRACKETED, " ", text)
    text = re.sub(UNINTELLIGIBLE_PAREN, " ", text)
    text = re.sub(WORD_PAREN, r"\1", text)
    text = text.replace("#x27;", "'")
    text = re.sub(PUNCTUATION_OTHER, " ", text)
    text = re.sub(SENTENCE_INITIAL, _lowercase_sentence_initial, text)
    # self-interruption em dash becomes a comma + space, same as any other comma, so it
    # collapses away to a single space by the time normalization finishes
    text = text.replace("—", ", ")
    text = re.sub(COMMA, " ", text)
    text = re.sub(SENTENCE_END, " ", text)
    text = text.replace("...", "!ELLIPSIS!").replace(".", " ").replace("!ELLIPSIS!", "...")
    while " ... " in text:
        text = text.replace(" ... ", " ")
    text = re.sub(MULTISPACE, " ", text)
    return text