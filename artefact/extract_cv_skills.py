import re
import pandas as pd

from src.config import ESCO_SKILLS_CLEANED_PATH


def normalise_text(text):
    text = str(text).lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def is_useful_skill_phrase(phrase):
    phrase_norm = normalise_text(phrase)

    banned_exact_phrases = {
        "assume responsibility",
        "adapt to change",
        "work independently",
        "cope with stress",
        "show commitment",
        "meet deadlines",
        "manage time",
        "time management",
        "problem solving",
        "communication",
        "management",
        "leadership",
        "customer satisfaction",
        "team working",
        "team player",
        "working closely",
        "attention to detail",
        "multitasking",
        "interpersonal skills",
        "organisational skills",
        "organizational skills",
        "motivated",
        "organised",
        "organized",
        "support managers",
        "lead a team",
        "manage a team",
        "work in an organised manner",
        "work in an organized manner",
        "communicate with customers",
        "maintain professional records",
        "build business relationships",
        "create solutions to problems",
        "perform project management",
        "manage personal professional development",
        "manage tasks",
        "perform office routine activities",
        "perform administrative duties",
        "carry out sales analysis",
        "implement strategic planning",
        "apply company policies",
        "perform services in a flexible manner",
        "maintenance operations",
        "product comprehension",
        "application process",
    }

    strong_single_words = {
        "sql",
        "java",
        "javascript",
        "python",
        "excel",
        "linux",
        "windows",
        "aws",
        "azure",
        "sap",
        "payroll",
        "tax",
        "accounting",
        "recruitment",
        "auditing",
        "compliance",
        "salesforce",
        "networking",
        "troubleshooting",
        "bookkeeping",
        "procurement",
        "cad",
        "autocad",
        "matlab",
        "powerbi",
        "tableau",
        "statistics",
    }

    allowed_multiword_exceptions = {
        "customer service",
        "human resource management",
        "project management",
        "employment law",
        "labour legislation",
        "lean manufacturing",
        "quality assurance procedures",
        "sap data services",
        "data protection",
        "international law",
        "quality standards",
        "risk management",
    }

    if phrase_norm in banned_exact_phrases:
        return False

    word_count = len(phrase_norm.split())

    if word_count == 1:
        return phrase_norm in strong_single_words

    if word_count >= 2:
        if phrase_norm in allowed_multiword_exceptions:
            return True
        return True

    return False


def load_esco_phrases():
    esco_df = pd.read_csv(ESCO_SKILLS_CLEANED_PATH)
    phrase_rows = []

    for _, row in esco_df.iterrows():
        preferred_label = row.get("preferred_label")
        alt_labels = row.get("alt_labels")

        if pd.notna(preferred_label):
            phrase = str(preferred_label).strip()
            if phrase and is_useful_skill_phrase(phrase):
                phrase_rows.append((normalise_text(phrase), preferred_label))

        if pd.notna(alt_labels):
            for alt_label in str(alt_labels).split("\n"):
                alt_label = alt_label.strip()
                if alt_label and is_useful_skill_phrase(alt_label):
                    phrase_rows.append((normalise_text(alt_label), preferred_label))

    return list(set(phrase_rows))


def extract_cv_skills_from_text(text):
    text_norm = normalise_text(text)
    esco_phrases = load_esco_phrases()

    found_skills = set()

    for phrase_norm, preferred_label in esco_phrases:
        pattern = r"\b" + re.escape(phrase_norm) + r"\b"
        if re.search(pattern, text_norm):
            found_skills.add(preferred_label.lower())

    return sorted(found_skills)