def calculate_weighted_match(job_skill_matches, cv_skill_matches, skill_dictionary):
    # calculate weighted score using confidence-adjusted job weights
    job_skills = sorted(job_skill_matches.keys())
    cv_skills = sorted(cv_skill_matches.keys())

    matched_skills = sorted(set(job_skills) & set(cv_skills))
    missing_skills = sorted(set(job_skills) - set(cv_skills))
    extra_skills = sorted(set(cv_skills) - set(job_skills))

    total_job_weight = 0.0
    matched_weight = 0.0

    for skill in job_skills:
        base_weight = skill_dictionary.get(skill, {}).get("weight", 1)
        confidence = job_skill_matches.get(skill, {}).get("confidence", 0.5)
        effective_weight = base_weight * confidence

        total_job_weight += effective_weight

        if skill in matched_skills:
            matched_weight += effective_weight

    if total_job_weight > 0:
        weighted_score_percent = round((matched_weight / total_job_weight) * 100, 2)
    else:
        weighted_score_percent = 0.0

    return {
        "job_skill_count": len(job_skills),
        "cv_skill_count": len(cv_skills),
        "matched_skill_count": len(matched_skills),
        "missing_skill_count": len(missing_skills),
        "extra_skill_count": len(extra_skills),
        "matched_weight_total": round(matched_weight, 2),
        "job_weight_total": round(total_job_weight, 2),
        "match_score_percent": weighted_score_percent,
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "extra_skills": extra_skills,
    }