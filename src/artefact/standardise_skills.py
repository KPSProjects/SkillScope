import re


def normalise_text(text):
    """
    Normalize text for consistent skill matching.
    Handles common tech abbreviations and special characters.
    """
    text = str(text).lower().strip()

    # Tech-specific replacements
    replacements = {
        "c#": "csharp",
        "f#": "fsharp",
        ".net": "dotnet",
        "asp.net core": "aspdotnet core",
        "asp.net": "aspdotnet",
        "node.js": "nodejs",
        "react.js": "reactjs",
        "vue.js": "vuejs",
        "angular.js": "angularjs",
        "next.js": "nextjs",
        "express.js": "expressjs",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    # Remove special characters except word chars, spaces, and common tech symbols
    text = re.sub(r"[^\w\s/+-]", " ", text)

    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def standardise_skill_label(label):
    """
    Standardize a skill label to canonical form.
    This is used when creating dictionary keys.
    """
    label = str(label).strip().lower()

    # Collapse whitespace
    label = re.sub(r"\s+", " ", label)

    return label


def find_canonical_match(user_label, skill_dictionary):
    """
    Find the canonical dictionary key for a user-provided label.

    This function enables alias resolution by:
    1. Normalizing the user label
    2. Checking if it matches any alias in the dictionary
    3. Returning the canonical key if found

    Args:
        user_label: The label to normalize (e.g., "python", "Python", "SQL")
        skill_dictionary: The full ESCO dictionary with aliases

    Returns:
        Canonical key if found, or the normalized user_label if not found
    """
    user_norm = normalise_text(user_label)

    # Direct check: is the normalized label already a canonical key?
    if user_norm in skill_dictionary:
        return user_norm

    # Alias check: does it match any alias in the dictionary?
    for canonical_key, skill_info in skill_dictionary.items():
        aliases = skill_info.get("aliases", [])

        for alias in aliases:
            alias_norm = normalise_text(alias)

            if user_norm == alias_norm:
                return canonical_key

    # No match found - return the normalized label as-is
    # This allows graceful handling of skills not in the dictionary
    return user_norm


def normalise_skill_list(skill_labels, skill_dictionary=None):
    """
    Normalize a list of skill labels.

    If skill_dictionary is provided, performs alias resolution to canonical keys.
    Otherwise, just normalizes the text.

    Args:
        skill_labels: List of skill label strings
        skill_dictionary: Optional dictionary for alias resolution

    Returns:
        List of normalized/canonical skill labels
    """
    normalized = []

    for label in skill_labels:
        if not label or not str(label).strip():
            continue

        if skill_dictionary:
            # Use alias resolution
            canonical = find_canonical_match(label, skill_dictionary)
            normalized.append(canonical)
        else:
            # Just normalize
            norm_label = normalise_text(label)
            if norm_label:
                normalized.append(norm_label)

    # Remove duplicates and sort
    return sorted(set(normalized))


def extract_short_form(label):
    """
    Extract short-form from parenthetical labels.

    Example: "python (computer programming)" -> "python"

    This is used when building dictionaries to create additional aliases.
    """
    label = str(label).strip()

    if "(" in label:
        short_form = label.split("(")[0].strip()
        if short_form and len(short_form) >= 2:
            return short_form

    return None