from src.artefact.compare_skills import (
    compare_skill_sets,
    save_match_report,
    save_match_results,
)
from src.artefact.generate_advice import save_advice_report
from src.config import (
    ARTEFACT_MATCH_RESULTS_PATH,
    ARTEFACT_MATCH_REPORT_PATH,
    ARTEFACT_ADVICE_REPORT_PATH,
)


def run_matching_demo():
    print("Running CV to job matching demo...")

    job_skills = [
        "python",
        "sql",
        "project management",
        "data protection",
        "risk management",
        "javascript",
    ]

    cv_skills = [
        "python",
        "sql",
        "tableau",
        "customer service",
        "project management",
    ]

    match_results = compare_skill_sets(job_skills=job_skills, cv_skills=cv_skills)

    save_match_results(match_results, ARTEFACT_MATCH_RESULTS_PATH)
    save_match_report(match_results, ARTEFACT_MATCH_REPORT_PATH)
    save_advice_report(match_results, ARTEFACT_ADVICE_REPORT_PATH)

    print(f"Saved match results to: {ARTEFACT_MATCH_RESULTS_PATH}")
    print(f"Saved match report to: {ARTEFACT_MATCH_REPORT_PATH}")
    print(f"Saved advice report to: {ARTEFACT_ADVICE_REPORT_PATH}")


if __name__ == "__main__":
    run_matching_demo()