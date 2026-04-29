from pathlib import Path
import json


def normalise_skill_list(skills):
    # clean and standardise a skill list into unique lowercase values
    cleaned_skills = []

    for skill in skills:
        skill_text = str(skill).strip().lower()
        if skill_text:
            cleaned_skills.append(skill_text)

    return sorted(set(cleaned_skills))


def compare_skill_sets(job_skills, cv_skills):
    # compare standardised job and cv skill lists
    job_set = set(normalise_skill_list(job_skills))
    cv_set = set(normalise_skill_list(cv_skills))

    matched_skills = sorted(job_set & cv_set)
    missing_skills = sorted(job_set - cv_set)
    extra_skills = sorted(cv_set - job_set)

    if len(job_set) > 0:
        match_score = (len(matched_skills) / len(job_set)) * 100
    else:
        match_score = 0.0

    return {
        "job_skill_count": len(job_set),
        "cv_skill_count": len(cv_set),
        "matched_skill_count": len(matched_skills),
        "missing_skill_count": len(missing_skills),
        "extra_skill_count": len(extra_skills),
        "match_score_percent": round(match_score, 2),
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "extra_skills": extra_skills,
    }


def save_match_results(match_results, output_path: Path):
    # save match results as json
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(match_results, file, indent=2, ensure_ascii=False)


def save_match_report(match_results, output_path: Path):
    # save a readable text report
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as file:
        file.write("CV TO JOB SKILL MATCH REPORT\n")
        file.write("============================\n\n")

        file.write(f"Job skill count: {match_results['job_skill_count']}\n")
        file.write(f"CV skill count: {match_results['cv_skill_count']}\n")
        file.write(f"Matched skills: {match_results['matched_skill_count']}\n")
        file.write(f"Missing skills: {match_results['missing_skill_count']}\n")
        file.write(f"Extra skills: {match_results['extra_skill_count']}\n")
        file.write(f"Matched weight total: {match_results['matched_weight_total']}\n")
        file.write(f"Total job weight: {match_results['job_weight_total']}\n")
        file.write(f"Match score: {match_results['match_score_percent']}%\n\n")


        file.write("Matched skills\n")
        file.write("--------------\n")
        for skill in match_results["matched_skills"]:
            file.write(f"- {skill}\n")

        file.write("\nMissing skills\n")
        file.write("--------------\n")
        for skill in match_results["missing_skills"]:
            file.write(f"- {skill}\n")

        file.write("\nExtra skills\n")
        file.write("------------\n")
        for skill in match_results["extra_skills"]:
            file.write(f"- {skill}\n")