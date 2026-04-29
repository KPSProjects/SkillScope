import re

from src.artefact.load_skill_dictionary import load_active_skill_dictionary
from src.artefact.advert_requirement_classifier import (
    classify_advert_lines,
    REQUIREMENT_CONFIDENCE,
)


def normalise_text(text):
    text = str(text).lower().strip()

    replacements = {
        "c#": "csharp",
        "f#": "fsharp",
        ".net": "dotnet",
        "asp.net core": "aspdotnet core",
        "asp.net": "aspdotnet",
        "node.js": "nodejs",
        "react.js": "reactjs",
        "vue.js": "vuejs",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    text = re.sub(r"[^\w\s/+-]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def find_alias_evidence(text, aliases):
    text_norm = normalise_text(text)

    # Aliases that cause bad false positives in CV/job text
    blocked_aliases = {
        "did",           # matched normal word "did"
        "chef",          # matched your TGI Fridays chef role as a software tool
        "transformer",   # matched NLP transformer to electrical/security ESCO skills
        "transformers",
        "it",            # too broad, matches normal text
        "team",          # too broad, creates weak teamwork matches everywhere
        "design",        # too broad, creates think creatively/design false positives
    }

    # Short aliases only allowed if they are clearly technical/useful
    short_alias_whitelist = {
        "ai",
        "ml",
        "sql",
        "api",
        "css",
        "html",
        "git",
        "aws",
        "gcp",
        "php",
        "ui",
        "ux",
        "qa",
        "bi",
        "3d",
        "c#",
        "c++",
    }

    for alias in aliases:
        alias_norm = normalise_text(alias)

        if not alias_norm:
            continue

        if alias_norm in blocked_aliases:
            continue

        # Avoid random short-word matches
        if len(alias_norm) <= 3 and alias_norm not in short_alias_whitelist:
            continue

        pattern = r"\b" + re.escape(alias_norm) + r"\b"
        match = re.search(pattern, text_norm)

        if match:
            start = max(0, match.start() - 40)
            end = min(len(text_norm), match.end() + 40)
            snippet = text_norm[start:end].strip()

            return {
                "matched_alias": alias,
                "snippet": snippet,
            }

    return None


def extract_evidence_skills_from_cv(text):
    skill_dictionary = load_active_skill_dictionary()
    found_skills = {}

    for canonical_skill, skill_info in skill_dictionary.items():
        evidence = find_alias_evidence(text, skill_info["aliases"])

        if evidence is None:
            continue

        found_skills[canonical_skill] = {
            "category": skill_info["category"],
            "weight": skill_info["weight"],
            "esco_labels": skill_info["esco_labels"],
            "matched_alias": evidence["matched_alias"],
            "snippet": evidence["snippet"],
            "requirement_type": "cv_evidence",
            "confidence": 1.0,
        }

    return found_skills


def extract_evidence_skills_from_job(text):
    skill_dictionary = load_active_skill_dictionary()
    classified_lines = classify_advert_lines(text)
    found_skills = {}

    for line_info in classified_lines:
        line = line_info["line"]
        requirement_type = line_info["line_type"]
        confidence = REQUIREMENT_CONFIDENCE.get(requirement_type, 0.4)

        # completely ignore training content, future-role content, and benefits
        if requirement_type in {"training_provided", "future_role", "benefit"}:
            continue

        for canonical_skill, skill_info in skill_dictionary.items():
            evidence = find_alias_evidence(line, skill_info["aliases"])

            if evidence is None:
                continue

            # keep the strongest confidence if the same skill appears multiple times
            if canonical_skill in found_skills:
                if confidence > found_skills[canonical_skill]["confidence"]:
                    found_skills[canonical_skill] = {
                        "category": skill_info["category"],
                        "weight": skill_info["weight"],
                        "esco_labels": skill_info["esco_labels"],
                        "matched_alias": evidence["matched_alias"],
                        "snippet": line.strip(),
                        "requirement_type": requirement_type,
                        "confidence": confidence,
                    }
            else:
                found_skills[canonical_skill] = {
                    "category": skill_info["category"],
                    "weight": skill_info["weight"],
                    "esco_labels": skill_info["esco_labels"],
                    "matched_alias": evidence["matched_alias"],
                    "snippet": line.strip(),
                    "requirement_type": requirement_type,
                    "confidence": confidence,
                }

    return found_skills


def extract_evidence_skills(text, mode="job"):
    if mode == "job":
        return extract_evidence_skills_from_job(text)

    return extract_evidence_skills_from_cv(text)