from pathlib import Path


def build_advice_lines(match_results):
    # create simple advice based on the match results
    advice_lines = []

    match_score = match_results.get("match_score_percent", 0)
    matched_skills = match_results.get("matched_skills", [])
    missing_skills = match_results.get("missing_skills", [])
    extra_skills = match_results.get("extra_skills", [])

    # overall judgement
    if match_score >= 75:
        advice_lines.append(
            "Overall, this CV appears to be a strong match for the job based on the extracted skills."
        )
    elif match_score >= 50:
        advice_lines.append(
            "Overall, this CV is a moderate match for the job, but there are still some missing skills that should be addressed."
        )
    else:
        advice_lines.append(
            "Overall, this CV is currently a weak match for the job based on the extracted skills."
        )

    # matched skills advice
    if matched_skills:
        top_matched = matched_skills[:5]
        advice_lines.append(
            "The CV already aligns with the advert in areas such as: "
            + ", ".join(top_matched)
            + "."
        )

    # missing skills advice
    if missing_skills:
        top_missing = missing_skills[:5]
        advice_lines.append(
            "The main missing skills identified were: "
            + ", ".join(top_missing)
            + "."
        )
        advice_lines.append(
            "These missing skills should be reviewed to decide whether they are genuinely absent, only implied, or simply not written clearly enough in the CV."
        )
        advice_lines.append(
            "If the candidate has experience in these areas, the CV should be tailored to mention them more directly using clearer wording."
        )
    else:
        advice_lines.append(
            "No missing skills were identified from the extracted comparison, which suggests strong alignment with the advert."
        )

    # extra skills advice
    if extra_skills:
        top_extra = extra_skills[:5]
        advice_lines.append(
            "The CV also contains additional skills not directly requested in the advert, such as: "
            + ", ".join(top_extra)
            + "."
        )
        advice_lines.append(
            "These extra skills may still be useful, but they should not take attention away from the most relevant job requirements."
        )

    # final practical advice
    if missing_skills:
        advice_lines.append(
            "A stronger final version of the CV should emphasise evidence of the missing skills through work history, projects, tools used, and achievements."
        )
    else:
        advice_lines.append(
            "The final CV should still be tailored carefully so that the strongest matching skills are clearly visible near the top of the document."
        )

    return advice_lines


def save_advice_report(match_results, output_path: Path):
    # save a readable advice report
    output_path.parent.mkdir(parents=True, exist_ok=True)

    advice_lines = build_advice_lines(match_results)

    with open(output_path, "w", encoding="utf-8") as file:
        file.write("CV TO JOB MATCH ADVICE REPORT\n")
        file.write("=============================\n\n")

        file.write(f"Match score: {match_results.get('match_score_percent', 0)}%\n")
        file.write(f"Matched skills: {match_results.get('matched_skill_count', 0)}\n")
        file.write(f"Missing skills: {match_results.get('missing_skill_count', 0)}\n")
        file.write(f"Extra skills: {match_results.get('extra_skill_count', 0)}\n\n")

        file.write("Advice\n")
        file.write("------\n")
        for line in advice_lines:
            file.write(f"- {line}\n")