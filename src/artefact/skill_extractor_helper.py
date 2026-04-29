import re
import pandas as pd

from src.config import ESCO_SKILLS_CLEANED_PATH


def normalise_text(text):
    # make text easier to match
    text = str(text).lower().strip()

    replacements = {
        "c#": "csharp",
        "f#": "fsharp",
        ".net": "dotnet",
        "asp.net core": "aspdotnet core",
        "asp.net": "aspdotnet",
        "node.js": "nodejs",
        "react.js": "reactjs",
        "vue.js": "vuejs",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def is_useful_skill_phrase(phrase, mode="job"):
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

    strict_technical_single_words = {
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
        "salesforce",
        "networking",
        "troubleshooting",
        "cad",
        "autocad",
        "matlab",
        "powerbi",
        "tableau",
        "statistics",
        "postgresql",
        "wpf",
        "agile",
        "git",
        "csharp",
        "dotnet",
        "mysql",
        "sqlite",
        "mongodb",
        "html",
        "css",
    }

    useful_general_single_words = {
        "accounting",
        "recruitment",
        "auditing",
        "compliance",
        "bookkeeping",
        "procurement",
        "payroll",
        "tax",
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
        "database management systems",
        "distributed version control",
        "version control",
        "automated testing",
        "code review",
        "agile development",
        "computer programming",
        "machine learning",
        "natural language processing",
        "data analysis",
        "data cleaning",
        "data visualisation",
        "software development",
        "software engineering",
        "information security",
        "security engineering",
        "human computer interaction",
        "digital data processing",
        "process data",
    }

    if phrase_norm in banned_exact_phrases:
        return False

    word_count = len(phrase_norm.split())

    if word_count == 1:
        return phrase_norm in strict_technical_single_words or phrase_norm in useful_general_single_words

    if word_count >= 2:
        if mode == "cv":
            # stricter on CVs to avoid pretending someone has stack they never wrote
            return phrase_norm in allowed_multiword_exceptions
        else:
            # broader on job adverts
            return True

    return False


def load_esco_phrases(mode="job"):
    esco_df = pd.read_csv(ESCO_SKILLS_CLEANED_PATH)
    phrase_rows = []

    for _, row in esco_df.iterrows():
        preferred_label = row.get("preferred_label")
        alt_labels = row.get("alt_labels")

        if pd.notna(preferred_label):
            phrase = str(preferred_label).strip()
            if phrase and is_useful_skill_phrase(phrase, mode=mode):
                phrase_rows.append((normalise_text(phrase), preferred_label))

        if pd.notna(alt_labels):
            for alt_label in str(alt_labels).split("\n"):
                alt_label = alt_label.strip()
                if alt_label and is_useful_skill_phrase(alt_label, mode=mode):
                    phrase_rows.append((normalise_text(alt_label), preferred_label))

    # manual boosts for technical adverts and CVs
    manual_phrases = [
        ("csharp", "c#"),
        ("dotnet", ".net"),
        ("aspdotnet core", "asp.net core"),
        ("wpf", "wpf"),
        ("postgresql", "postgresql"),
        ("version control", "version control"),
        ("distributed version control", "distributed version control"),
        ("automated tests", "automated testing"),
        ("automated test", "automated testing"),
        ("code reviews", "code review"),
        ("code review", "code review"),
        ("agile", "agile development"),
        ("machine learning", "machine learning"),
        ("natural language processing", "natural language processing"),
        ("data analysis", "data analysis"),
        ("data cleaning", "data cleaning"),
        ("data visualisation", "data visualisation"),
        ("software development", "software development"),
        ("software engineering", "software engineering"),
        ("information security", "information security"),
        ("security engineering", "security engineering"),
        ("human computer interaction", "human computer interaction"),
        ("database management systems", "database management systems"),
        ("process data", "process data"),
        ("digital data processing", "digital data processing"),
    ]

    phrase_rows.extend(manual_phrases)

    return list(set(phrase_rows))


def extract_skills_from_text(text, mode="job"):
    text_norm = normalise_text(text)
    esco_phrases = load_esco_phrases(mode=mode)

    found_skills = set()

    for phrase_norm, preferred_label in esco_phrases:
        pattern = r"\b" + re.escape(phrase_norm) + r"\b"
        if re.search(pattern, text_norm):
            found_skills.add(preferred_label.lower())

    return sorted(found_skills)