from src.artefact.evidence_matcher import extract_evidence_skills


def extract_cv_skills_from_text(text):
    skill_matches = extract_evidence_skills(text, mode="cv")
    return sorted(skill_matches.keys())