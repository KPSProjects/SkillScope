import json
from pathlib import Path

from src.artefact.evidence_matcher import extract_evidence_skills
from src.artefact.score_match import calculate_weighted_match
from src.artefact.load_skill_dictionary import load_active_skill_dictionary
from src.artefact.load_inputs import load_text_file
from src.artefact.save_evidence_report import save_evidence_report
from src.artefact.generate_advice import save_advice_report

from src.config import (
    ARTEFACT_MULTI_JOB_SUMMARY_PATH,
    ARTEFACT_MULTI_JOB_RESULTS_PATH,
)


def ask_for_multiline_text(prompt_text):
    # Lets the user paste long CV/job advert text into the terminal
    print(prompt_text)
    print("Paste the text below. When finished, type END on a new line.")

    lines = []

    while True:
        line = input()

        if line.strip().upper() == "END":
            break

        lines.append(line)

    return "\n".join(lines)


def ask_for_file_text(prompt_text):
    # Lets the user provide a text file path
    file_path = Path(input(prompt_text).strip())

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    return load_text_file(file_path)


def get_cv_text():
    print("\nChoose CV input method:")
    print("1 - Use CV text file path")
    print("2 - Paste CV text manually")

    choice = input("Enter 1 or 2: ").strip()

    if choice == "1":
        return ask_for_file_text("Enter CV text file path: ")

    if choice == "2":
        return ask_for_multiline_text("Enter CV text")

    raise ValueError("Invalid CV input option. Please enter 1 or 2.")


def get_job_texts():
    print("\nChoose job advert input method:")
    print("1 - Use folder of .txt job adverts")
    print("2 - Paste one job advert manually")
    print("3 - Paste multiple job adverts manually")

    choice = input("Enter 1, 2, or 3: ").strip()

    job_texts = []

    if choice == "1":
        folder_path = Path(input("Enter job adverts folder path: ").strip())

        if not folder_path.exists():
            raise FileNotFoundError(f"Folder not found: {folder_path}")

        job_files = sorted(folder_path.glob("*.txt"))

        if not job_files:
            raise FileNotFoundError(f"No .txt job adverts found in: {folder_path}")

        for job_file in job_files:
            job_texts.append(
                {
                    "job_name": job_file.name,
                    "job_text": load_text_file(job_file),
                }
            )

        return job_texts

    if choice == "2":
        job_text = ask_for_multiline_text("Enter job advert text")

        job_texts.append(
            {
                "job_name": "pasted_job_1.txt",
                "job_text": job_text,
            }
        )

        return job_texts

    if choice == "3":
        number_of_jobs = int(input("How many job adverts do you want to paste? ").strip())

        for i in range(1, number_of_jobs + 1):
            job_text = ask_for_multiline_text(f"Enter job advert {i} text")

            job_texts.append(
                {
                    "job_name": f"pasted_job_{i}.txt",
                    "job_text": job_text,
                }
            )

        return job_texts

    raise ValueError("Invalid job input option. Please enter 1, 2, or 3.")


def save_interactive_summary(results, output_path: Path):
    # Save ranked summary of user-provided CV/job comparisons
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as file:
        file.write("INTERACTIVE CV TO JOB MATCH SUMMARY\n")
        file.write("===================================\n\n")

        for i, result in enumerate(results, start=1):
            file.write(f"{i}. {result['job_file_name']}\n")
            file.write(f"   Match score: {result['match_score_percent']}%\n")
            file.write(f"   Job skill count: {result['job_skill_count']}\n")
            file.write(f"   CV skill count: {result['cv_skill_count']}\n")
            file.write(f"   Matched skills: {result['matched_skill_count']}\n")
            file.write(f"   Missing skills: {result['missing_skill_count']}\n")
            file.write(f"   Extra skills: {result['extra_skill_count']}\n")
            file.write(f"   Matched weight total: {result['matched_weight_total']}\n")
            file.write(f"   Total job weight: {result['job_weight_total']}\n")

            if result["matched_skills"]:
                file.write(f"   Top matched: {', '.join(result['matched_skills'][:5])}\n")

            if result["missing_skills"]:
                file.write(f"   Top missing: {', '.join(result['missing_skills'][:5])}\n")

            file.write("\n")


def save_interactive_results(results, output_path: Path):
    # Save full structured results as JSON
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, ensure_ascii=False)


def run_interactive_matching_demo():
    print("Running interactive CV to job matching demo...")

    skill_dictionary = load_active_skill_dictionary()

    cv_text = get_cv_text()
    job_texts = get_job_texts()

    cv_skill_matches = extract_evidence_skills(cv_text, mode="cv")
    cv_skills = sorted(cv_skill_matches.keys())

    print(f"\nExtracted CV skills: {len(cv_skills)}")
    print(f"Active dictionary size: {len(skill_dictionary)}")
    print(f"Jobs to compare: {len(job_texts)}\n")

    all_results = []

    output_folder = ARTEFACT_MULTI_JOB_SUMMARY_PATH.parent

    for job_data in job_texts:
        job_name = job_data["job_name"]
        job_text = job_data["job_text"]

        print(f"Processing: {job_name}")

        job_skill_matches = extract_evidence_skills(job_text, mode="job")

        match_results = calculate_weighted_match(
            job_skill_matches=job_skill_matches,
            cv_skill_matches=cv_skill_matches,
            skill_dictionary=skill_dictionary,
        )

        match_results["job_file_name"] = job_name

        safe_job_name = Path(job_name).stem.replace(" ", "_")

        evidence_report_path = output_folder / f"{safe_job_name}_interactive_evidence_report.txt"
        advice_report_path = output_folder / f"{safe_job_name}_interactive_advice_report.txt"

        save_evidence_report(
            job_skill_matches=job_skill_matches,
            cv_skill_matches=cv_skill_matches,
            output_path=evidence_report_path,
        )

        save_advice_report(
            match_results=match_results,
            output_path=advice_report_path,
        )

        result_row = {
            "job_file_name": job_name,
            "job_skill_count": match_results["job_skill_count"],
            "cv_skill_count": match_results["cv_skill_count"],
            "matched_skill_count": match_results["matched_skill_count"],
            "missing_skill_count": match_results["missing_skill_count"],
            "extra_skill_count": match_results["extra_skill_count"],
            "matched_weight_total": match_results["matched_weight_total"],
            "job_weight_total": match_results["job_weight_total"],
            "match_score_percent": match_results["match_score_percent"],
            "matched_skills": match_results["matched_skills"],
            "missing_skills": match_results["missing_skills"],
            "extra_skills": match_results["extra_skills"],
        }

        all_results.append(result_row)

    all_results.sort(key=lambda x: x["match_score_percent"], reverse=True)

    interactive_summary_path = output_folder / "interactive_match_summary.txt"
    interactive_results_path = output_folder / "interactive_match_results.json"

    save_interactive_summary(all_results, interactive_summary_path)
    save_interactive_results(all_results, interactive_results_path)

    print("\nInteractive matching complete.")
    print(f"Saved summary to: {interactive_summary_path}")
    print(f"Saved results to: {interactive_results_path}")
    print(f"Saved evidence and advice reports to: {output_folder}")


if __name__ == "__main__":
    run_interactive_matching_demo()