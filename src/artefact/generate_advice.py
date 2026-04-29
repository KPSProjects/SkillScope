from pathlib import Path


def get_match_band(match_score):
    if match_score >= 75:
        return "strong"
    if match_score >= 50:
        return "moderate"
    return "weak"


def build_advice_lines(match_results):
    advice_lines = []

    job_file_name = match_results.get("job_file_name", "selected job")
    match_score = match_results.get("match_score_percent", 0)
    matched_skills = match_results.get("matched_skills", [])
    missing_skills = match_results.get("missing_skills", [])
    extra_skills = match_results.get("extra_skills", [])
    job_skill_count = match_results.get("job_skill_count", 0)

    if job_skill_count == 0:
        return [
            "No job skills were extracted from the advert, so this result should not be treated as reliable.",
            "The advert may use wording that is not currently covered by the dictionary or ESCO alias rules.",
            "The extractor should be reviewed using the evidence report before using this score for decision-making.",
        ]

    match_band = get_match_band(match_score)

    if match_band == "strong":
        advice_lines.append(
            f"{job_file_name} appears to be a strong match based on the extracted skill evidence."
        )
    elif match_band == "moderate":
        advice_lines.append(
            f"{job_file_name} appears to be a moderate match. The CV has relevant overlap, but some important job requirements are still missing."
        )
    else:
        advice_lines.append(
            f"{job_file_name} appears to be a weaker match. The CV contains some relevant skills, but several extracted job requirements are not clearly evidenced."
        )

    if matched_skills:
        top_matched = matched_skills[:7]
        advice_lines.append(
            "The strongest areas of alignment are: "
            + ", ".join(top_matched)
            + "."
        )
    else:
        advice_lines.append(
            "No direct skill overlaps were found between the CV and this advert."
        )

    if missing_skills:
        top_missing = missing_skills[:7]
        advice_lines.append(
            "The main missing or weakly evidenced areas are: "
            + ", ".join(top_missing)
            + "."
        )
        advice_lines.append(
            "These should be checked manually against the CV. Some may be genuinely missing, while others may be present but not written clearly enough for the matcher to detect."
        )
        advice_lines.append(
            "If the candidate has experience in these areas, the CV should be tailored by adding clearer evidence in the skills section, project descriptions, or work experience bullet points."
        )
    else:
        advice_lines.append(
            "No missing skills were identified, suggesting strong coverage of the extracted advert requirements."
        )

    if extra_skills:
        top_extra = extra_skills[:7]
        advice_lines.append(
            "The CV also includes additional skills not directly extracted from this advert, such as: "
            + ", ".join(top_extra)
            + "."
        )
        advice_lines.append(
            "These additional skills may still be valuable, but the CV should prioritise the skills that are directly requested by the advert."
        )

    if match_band == "strong":
        advice_lines.append(
            "Recommended action: keep the CV focused and make sure the strongest matched skills are visible near the top."
        )
    elif match_band == "moderate":
        advice_lines.append(
            "Recommended action: tailor the CV before applying by adding stronger wording for the missing skills that the candidate genuinely has."
        )
    else:
        advice_lines.append(
            "Recommended action: only apply with this CV if the missing skills are not essential, or rewrite the CV substantially to evidence the missing requirements."
        )

    return advice_lines


def save_advice_report(match_results, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    advice_lines = build_advice_lines(match_results)

    with open(output_path, "w", encoding="utf-8") as file:
        file.write("CV TO JOB MATCH ADVICE REPORT\n")
        file.write("=============================\n\n")

        file.write(f"Job file: {match_results.get('job_file_name', 'N/A')}\n")
        file.write(f"Match score: {match_results.get('match_score_percent', 0)}%\n")
        file.write(f"Job skill count: {match_results.get('job_skill_count', 0)}\n")
        file.write(f"CV skill count: {match_results.get('cv_skill_count', 0)}\n")
        file.write(f"Matched skills: {match_results.get('matched_skill_count', 0)}\n")
        file.write(f"Missing skills: {match_results.get('missing_skill_count', 0)}\n")
        file.write(f"Extra skills: {match_results.get('extra_skill_count', 0)}\n")
        file.write(f"Matched weight total: {match_results.get('matched_weight_total', 0)}\n")
        file.write(f"Total job weight: {match_results.get('job_weight_total', 0)}\n\n")

        file.write("Matched skills\n")
        file.write("--------------\n")
        for skill in match_results.get("matched_skills", []):
            file.write(f"- {skill}\n")

        file.write("\nMissing skills\n")
        file.write("--------------\n")
        for skill in match_results.get("missing_skills", []):
            file.write(f"- {skill}\n")

        file.write("\nExtra CV skills\n")
        file.write("---------------\n")
        for skill in match_results.get("extra_skills", []):
            file.write(f"- {skill}\n")

        file.write("\nAdvice\n")
        file.write("------\n")
        for line in advice_lines:
            file.write(f"- {line}\n")