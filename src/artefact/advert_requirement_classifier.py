REQUIREMENT_CONFIDENCE = {
    "required": 1.0,
    "desirable": 0.6,
    "uncertain": 0.4,
    "training_provided": 0.0,
    "future_role": 0.0,
    "benefit": 0.0,
    "trait_description": 0.1,
    "assessment_method": 0.0,
}


def split_text_into_lines(text):
    # Split advert into cleaned non-empty lines
    raw_lines = str(text).splitlines()
    cleaned_lines = []

    for line in raw_lines:
        line = line.strip()
        if line:
            cleaned_lines.append(line)

    return cleaned_lines


def classify_line_type(line):
    # Classify a job advert line using keyword rules
    line_norm = line.lower().strip()

    trait_description_clues = [
        "demonstrate four key traits",
        "passion, curiosity, commitment",
        "our most successful applicants demonstrate",
        "key traits:",
        "what we look for",
        "ideal candidate",
        "personality",
        "motivated individual",
        "enthusiastic",
        "proactive approach",
        "positive attitude",
        "team spirit",
        "cultural fit",
    ]

    assessment_method_clues = [
        "skills assessment",
        "assessed through",
        "assessment will include",
        "written, maths and python coding challenges",
        "series of challenges",
        "screening interview",
        "final interview",
        "application process",
        "interview process",
        "you'll be tested on",
    ]

    required_clues = [
        "required",
        "requirements",
        "must have",
        "what we're looking for",
        "what we are looking for",
        "skills and experience required",
        "required skills and qualifications",
        "essential",
        "entry requirement",
        "entry requirements",
        "must be willing",
        "you must",
        "must have the full right to work",

        # More general job-action wording
        "you will:",
        "you will",
        "you'll",
        "you will have experience",
        "you will experience",

        # Hospitality wording
        "as chef de partie",
        "as junior sous chef",
        "as chef de partie / junior sous chef",
    ]

    desirable_clues = [
        "desirable",
        "advantageous",
        "preferred",
        "plus",
        "familiarity with",
        "exposure to",
        "not essential",
        "a plus",
        "highly desirable",
    ]

    training_clues = [
        "training covers",
        "training modules",
        "you'll achieve",
        "you will achieve",
        "certification",
        "certifications",
        "academy",
        "immersive training",
        "industry-recognised",
        "industry-recognized",
        "your journey begins",
        "the programme follows",
        "full-time training",
        "training period",
        "instructors",
        "graduate from our programme",
        "as part of your graduation",
    ]

    future_role_clues = [
        "role types",
        "could be deployed into",
        "deployed into",
        "after completing your first deployment",
        "future role",
        "future roles",
        "once deployed",
        "client site",
        "permanent employee of digital futures ready for deployment",
    ]

    benefit_clues = [
        "benefits",
        "salary",
        "bonus",
        "pension",
        "why join",
        "we offer",
        "package",
        "availability",
        "eligibility",
        "year 1:",
        "year 2:",
        "fully remote",
        "employee referral scheme",
        "bank holidays",
        "annual leave",
        "free tea",
        "free coffee",
        "on site parking",
        "parking",
    ]

    for clue in trait_description_clues:
        if clue in line_norm:
            return "trait_description"

    for clue in assessment_method_clues:
        if clue in line_norm:
            return "assessment_method"

    for clue in training_clues:
        if clue in line_norm:
            return "training_provided"

    for clue in future_role_clues:
        if clue in line_norm:
            return "future_role"

    for clue in benefit_clues:
        if clue in line_norm:
            return "benefit"

    for clue in required_clues:
        if clue in line_norm:
            return "required"

    for clue in desirable_clues:
        if clue in line_norm:
            return "desirable"

    return "uncertain"


def classify_advert_lines(text):
    # Classify each advert line and carry section meaning forward
    lines = split_text_into_lines(text)
    classified_lines = []

    current_section_type = "uncertain"

    required_headings = {
        "what we're looking for",
        "what we are looking for",
        "required skills and qualifications",
        "skills and experience required",
        "required skills",
        "requirements",
        "how to apply",

        # Hospitality heading
        "as chef de partie / junior sous chef you will:",
    }

    training_headings = {
        "frontier ai programme",
        "1. immersive training",
        "2. graduation",
    }

    future_role_headings = {
        "3. employment",
    }

    benefit_headings = {
        "benefits",
        "salary",
        "key programme information",
        "why join digital futures?",
    }

    for line in lines:
        lower_line = line.lower().strip()
        explicit_type = classify_line_type(line)

        if lower_line in required_headings:
            current_section_type = "required"
            classified_lines.append({"line": line, "line_type": "required"})
            continue

        if lower_line in training_headings:
            current_section_type = "training_provided"
            classified_lines.append({"line": line, "line_type": "training_provided"})
            continue

        if lower_line in future_role_headings:
            current_section_type = "future_role"
            classified_lines.append({"line": line, "line_type": "future_role"})
            continue

        if lower_line in benefit_headings:
            current_section_type = "benefit"
            classified_lines.append({"line": line, "line_type": "benefit"})
            continue

        if explicit_type != "uncertain":
            classified_lines.append({"line": line, "line_type": explicit_type})

            if explicit_type in {
                "required",
                "training_provided",
                "future_role",
                "benefit",
                "trait_description",
                "assessment_method",
            }:
                current_section_type = explicit_type

            continue

        if current_section_type == "training_provided":
            classified_lines.append({"line": line, "line_type": "training_provided"})
            continue

        if current_section_type == "future_role":
            classified_lines.append({"line": line, "line_type": "future_role"})
            continue

        if current_section_type == "benefit":
            classified_lines.append({"line": line, "line_type": "benefit"})
            continue

        classified_lines.append({"line": line, "line_type": current_section_type})

    return classified_lines