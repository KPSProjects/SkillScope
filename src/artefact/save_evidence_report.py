from pathlib import Path


def save_evidence_report(job_skill_matches, cv_skill_matches, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as file:
        file.write("SKILL MATCH EVIDENCE REPORT\n")
        file.write("===========================\n\n")

        file.write("JOB SKILL EVIDENCE\n")
        file.write("------------------\n\n")

        for skill, info in sorted(job_skill_matches.items()):
            file.write(f"Skill: {skill}\n")
            file.write(f"Category: {info.get('category')}\n")
            file.write(f"Weight: {info.get('weight')}\n")
            file.write(f"Requirement type: {info.get('requirement_type')}\n")
            file.write(f"Confidence: {info.get('confidence')}\n")
            file.write(f"Matched alias: {info.get('matched_alias')}\n")
            file.write(f"Evidence: {info.get('snippet')}\n")
            file.write("\n")

        file.write("\nCV SKILL EVIDENCE\n")
        file.write("-----------------\n\n")

        for skill, info in sorted(cv_skill_matches.items()):
            file.write(f"Skill: {skill}\n")
            file.write(f"Category: {info.get('category')}\n")
            file.write(f"Weight: {info.get('weight')}\n")
            file.write(f"Matched alias: {info.get('matched_alias')}\n")
            file.write(f"Evidence: {info.get('snippet')}\n")
            file.write("\n")