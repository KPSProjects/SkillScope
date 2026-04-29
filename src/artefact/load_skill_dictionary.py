import json

from src.config import ARTEFACT_GENERATED_DICTIONARY_PATH
from src.artefact.skill_dictionary import SKILL_DICTIONARY
from src.artefact.standardise_skills import normalise_text


CUSTOM_ALIASES = {
    "api development": [
        "build apis",
        "building apis",
        "develop apis",
        "developing apis",
        "api design",
        "api integration",
        "web api",
        "web apis",
    ],
    "rest api": [
        "restful services",
        "restful api",
        "restful apis",
        "rest services",
        "rest endpoints",
        "restful endpoints",
    ],
    "version control": [
        "source control",
        "git workflow",
        "git workflows",
        "branching",
        "pull requests",
        "merge requests",
    ],
    "automated testing": [
        "unit testing",
        "unit tests",
        "integration testing",
        "integration tests",
        "test automation",
        "automated tests",
    ],
    "software development": [
        "develop software",
        "building software",
        "software projects",
        "application development",
        "app development",
    ],
    "data analysis": [
        "analyse data",
        "analyze data",
        "data insights",
        "insight generation",
        "data reporting",
    ],
    "data visualisation": [
        "data visualization",
        "dashboards",
        "dashboarding",
        "visual reports",
        "plotly",
        "power bi",
        "powerbi",
        "tableau",
    ],
    "postgresql": [
        "postgres",
        "postgre sql",
    ],
    "c#": [
        "c sharp",
        "csharp",
    ],
    ".net": [
        "dotnet",
        "dot net",
        ".net framework",
        ".net core",
    ],
    "reactjs": [
        "react",
        "react.js",
        "react js",
    ],
}


def merge_aliases(existing_aliases, new_aliases):
    aliases = set()

    for alias in existing_aliases:
        if alias and str(alias).strip():
            aliases.add(str(alias).strip().lower())

    for alias in new_aliases:
        if alias and str(alias).strip():
            aliases.add(str(alias).strip().lower())

    return sorted(aliases)


def load_generated_esco_dictionary():
    if not ARTEFACT_GENERATED_DICTIONARY_PATH.exists():
        return {}

    with open(ARTEFACT_GENERATED_DICTIONARY_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def merge_manual_and_esco_dictionaries():
    """
    Builds the active matching dictionary.

    Manual dictionary is treated as the high-trust core.
    Generated ESCO is used as supporting coverage.
    Custom aliases are added to catch real CV/job advert wording.
    """
    generated_esco = load_generated_esco_dictionary()

    active_dictionary = {}

    # 1. Start with generated ESCO first as broad background coverage
    for skill, info in generated_esco.items():
        canonical = normalise_text(skill)

        active_dictionary[canonical] = {
            "aliases": merge_aliases(info.get("aliases", []), [skill]),
            "category": info.get("category", "general"),
            "weight": info.get("weight", 1),
            "esco_labels": info.get("esco_labels", [skill]),
            "source": "esco_generated",
        }

    # 2. Overlay manual dictionary as trusted project-specific knowledge
    for skill, info in SKILL_DICTIONARY.items():
        canonical = normalise_text(skill)

        existing = active_dictionary.get(canonical, {})

        existing_aliases = existing.get("aliases", [])
        manual_aliases = info.get("aliases", [])

        existing_esco_labels = existing.get("esco_labels", [])
        manual_esco_labels = info.get("esco_labels", [])

        active_dictionary[canonical] = {
            "aliases": merge_aliases(existing_aliases, manual_aliases + [skill]),
            "category": info.get("category", existing.get("category", "general")),
            "weight": max(info.get("weight", 1), existing.get("weight", 1)),
            "esco_labels": sorted(set(existing_esco_labels + manual_esco_labels)),
            "source": "manual_priority",
        }

    # 3. Add custom real-world aliases
    for canonical_skill, aliases in CUSTOM_ALIASES.items():
        canonical = normalise_text(canonical_skill)

        if canonical not in active_dictionary:
            active_dictionary[canonical] = {
                "aliases": merge_aliases([], aliases + [canonical_skill]),
                "category": "custom",
                "weight": 2,
                "esco_labels": [canonical_skill],
                "source": "custom_alias",
            }
        else:
            active_dictionary[canonical]["aliases"] = merge_aliases(
                active_dictionary[canonical].get("aliases", []),
                aliases + [canonical_skill],
            )

            if active_dictionary[canonical].get("source") == "esco_generated":
                active_dictionary[canonical]["source"] = "esco_with_custom_aliases"

    return active_dictionary


def load_active_skill_dictionary():
    return merge_manual_and_esco_dictionaries()