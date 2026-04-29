from src.artefact.load_inputs import load_text_file
from src.artefact.evidence_matcher import extract_evidence_skills
from src.artefact.score_match import calculate_weighted_match
from src.artefact.load_skill_dictionary import load_active_skill_dictionary
from src.artefact.compare_skills import (
    save_match_results,
    save_match_report,
)
from src.artefact.generate_advice import save_advice_report
from src.config import (
    ARTEFACT_MATCH_RESULTS_PATH,
    ARTEFACT_MATCH_REPORT_PATH,
    ARTEFACT_ADVICE_REPORT_PATH,
    ARTEFACT_JOB_INPUT_PATH,
    ARTEFACT_CV_INPUT_PATH,
    ARTEFACT_JOB_SKILLS_PATH,
    ARTEFACT_CV_SKILLS_PATH,
)


def run_real_matching_demo():
    print("Running real CV to job matching demo...")

    skill_dictionary = load_active_skill_dictionary()

    job_text = load_text_file(ARTEFACT_JOB_INPUT_PATH)
    cv_text = load_text_file(ARTEFACT_CV_INPUT_PATH)

    print(f"Loaded job text length: {len(job_text)}")
    print(f"Loaded CV text length: {len(cv_text)}")

    job_skill_matches = extract_evidence_skills(job_text, mode="job")
    cv_skill_matches = extract_evidence_skills(cv_text, mode="cv")

    job_skills = sorted(job_skill_matches.keys())
    cv_skills = sorted(cv_skill_matches.keys())

    print(f"Extracted job skills: {len(job_skills)}")
    print(f"Extracted CV skills: {len(cv_skills)}")

    match_results = calculate_weighted_match(
        job_skill_matches=job_skill_matches,
        cv_skill_matches=cv_skill_matches,
        skill_dictionary=skill_dictionary,
    )

    save_match_results(match_results, ARTEFACT_MATCH_RESULTS_PATH)
    save_match_report(match_results, ARTEFACT_MATCH_REPORT_PATH)
    save_advice_report(match_results, ARTEFACT_ADVICE_REPORT_PATH)

    ARTEFACT_JOB_SKILLS_PATH.parent.mkdir(parents=True, exist_ok=True)
    ARTEFACT_CV_SKILLS_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(ARTEFACT_JOB_SKILLS_PATH, "w", encoding="utf-8") as file:
        for skill in job_skills:
            file.write(f"{skill}\n")

    with open(ARTEFACT_CV_SKILLS_PATH, "w", encoding="utf-8") as file:
        for skill in cv_skills:
            file.write(f"{skill}\n")

    print(f"Saved job skills to: {ARTEFACT_JOB_SKILLS_PATH}")
    print(f"Saved CV skills to: {ARTEFACT_CV_SKILLS_PATH}")
    print(f"Saved match results to: {ARTEFACT_MATCH_RESULTS_PATH}")
    print(f"Saved match report to: {ARTEFACT_MATCH_REPORT_PATH}")
    print(f"Saved advice report to: {ARTEFACT_ADVICE_REPORT_PATH}")


if __name__ == "__main__":
    run_real_matching_demo()