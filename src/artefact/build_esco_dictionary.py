import json
import re
from collections import defaultdict

import pandas as pd

from src.config import (
    ESCO_SKILLS_CLEANED_PATH,
    ARTEFACT_GENERATED_DICTIONARY_PATH,
)


def normalise_text(text):
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

    text = re.sub(r"[^\w\s/+-]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def make_canonical_label(label):
    label = str(label).strip().lower()
    label = re.sub(r"\s+", " ", label)
    return label


def standardise_canonical_skill(label):
    label_norm = normalise_text(label)

    canonical_map = {
        # Programming languages
        "python computer programming": "python",
        "python": "python",
        "java computer programming": "java",
        "java": "java",
        "javascript computer programming": "javascript",
        "javascript": "javascript",
        "typescript computer programming": "typescript",
        "typescript": "typescript",
        "csharp computer programming": "c#",
        "csharp": "c#",
        "c programming": "c",
        "c": "c",
        "c++ computer programming": "c++",
        "c++": "c++",
        "sql computer programming": "sql",
        "sql": "sql",
        "r computer programming": "r",
        "r": "r",
        "php computer programming": "php",
        "php": "php",
        "ruby computer programming": "ruby",
        "ruby": "ruby",
        "computer programming": "programming",
        "programming language": "programming",

        # Data / AI clean-up
        "ml computer programming": "machine learning",
        "machine learning": "machine learning",
        "artificial intelligence": "artificial intelligence",
        "natural language processing": "natural language processing",
        "predictive modelling": "predictive modelling",
        "predictive modeling": "predictive modelling",
        "data wrangling": "data wrangling",
        "analytics": "analytics",

        # Databases
        "postgres": "postgresql",
        "postgre sql": "postgresql",
        "postgresql": "postgresql",
        "mysql": "mysql",
        "sqlite": "sqlite",
        "mongodb": "mongodb",

        # Web / software
        "dotnet": ".net",
        "aspdotnet": "asp.net",
        "aspdotnet core": "asp.net core",
        "reactjs": "reactjs",
        "react js": "reactjs",
        "react": "reactjs",
        "vuejs": "vuejs",
        "vue js": "vuejs",
        "vue": "vuejs",
        "nodejs": "nodejs",
        "node js": "nodejs",
        "style sheet languages": "css",
        "use markup languages": "html",

        # Tools
        "tools for software configuration management": "git",
        "software configuration management": "git",
        "distributed version control": "version control",

        # Project / process
        "project management": "project management",
        "agile project management": "agile project management",
    }

    return canonical_map.get(label_norm, make_canonical_label(label))


def extract_short_form(label):
    label = str(label).strip()

    if "(" in label:
        short_form = label.split("(")[0].strip()
        if short_form and len(short_form) >= 2:
            return short_form

    return None


def is_blocked_phrase(label):
    label_norm = normalise_text(label)

    blocked_exact = {
        # Generic behaviour / personality / vague phrases
        "assume responsibility",
        "adapt to change",
        "work independently",
        "cope with stress",
        "show commitment",
        "meet deadlines",
        "manage time",
        "time management",
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

        # ESCO noise found during testing
        "pregnancy",
        "yoga",
        "breaking",
        "morality",
        "ethics",
        "logic",
        "plan",
        "dies",
        "less",
        "call",
        "brands",
        "patterns",
        "source",
        "cooperate",
        "reconstruct",
        "purchase",
        "hire people",
        "be responsible",
        "enjoy working",
        "lead",
        "stay focused",
        "trademarks",
        "assertiveness",
        "demonstrate curiosity",
        "curiosity",
        "passion",
        "commitment",
        "ambition",
        "traits",
        "reverse engineering",
        "perform warehousing operations",

        # Remaining false positives from evidence testing
        "project commissioning",
        "systems theory",
        "direct inward dialing",
        "electrical machines",
        "security panels",
        "chef tools for software configuration management",
        "computer technology",
        "think creatively",
    }

    blocked_starts = [
        "ability to ",
        "knowledge of ",
        "understanding of ",
        "experience of ",
        "experience in ",
        "responsible for ",
        "ensure ",
        "maintain ",
        "support ",
        "assist ",
        "help ",
        "provide ",
        "deliver ",
        "monitor ",
        "report ",
        "manage ",
        "lead ",
        "coordinate ",
        "organise ",
        "organize ",
        "demonstrate ",
        "perform ",
    ]

    too_generic_single_words = {
        "management",
        "support",
        "delivery",
        "operations",
        "planning",
        "services",
        "development",
        "training",
        "monitoring",
        "care",
        "teaching",
        "sales",
        "marketing",
        "leadership",
        "responsibility",
        "change",
        "commitment",
        "stress",
        "service",
        "work",
        "skills",
    }

    if label_norm in blocked_exact:
        return True

    if any(label_norm.startswith(prefix) for prefix in blocked_starts):
        return True

    words = label_norm.split()

    if len(words) == 1 and label_norm in too_generic_single_words:
        return True

    if len(label_norm) < 2:
        return True

    return False


def infer_category(label):
    label_norm = normalise_text(label)

    keyword_categories = [
        (
            {
                "python", "java", "javascript", "typescript", "php", "csharp",
                "dotnet", "html", "css", "jquery", "mvc", "react", "angular",
                "vue", "ruby", "golang", "rust", "swift", "kotlin", "nodejs",
                "programming", "software development",
            },
            "software",
        ),
        (
            {
                "sql", "mysql", "postgresql", "sqlite", "mongodb", "nosql",
                "database", "redis", "cassandra", "dynamodb",
            },
            "database",
        ),
        (
            {
                "azure", "aws", "cloud", "gcp", "google cloud", "kubernetes",
                "docker", "containerization",
            },
            "cloud",
        ),
        (
            {
                "devops", "ci", "cd", "deployment", "version control", "git",
                "jenkins", "gitlab", "github",
            },
            "devops",
        ),
        (
            {
                "security", "oauth", "rbac", "data protection", "gdpr",
                "encryption", "penetration testing", "cybersecurity",
                "secure coding",
            },
            "security",
        ),
        (
            {
                "machine learning", "artificial intelligence", "ai", "nlp",
                "predictive", "analytics", "data wrangling", "deep learning",
                "neural network", "tensorflow", "pytorch",
            },
            "data_ai",
        ),
        (
            {
                "excel", "reporting", "documentation", "record keeping",
                "administration", "file archiving",
            },
            "business_support",
        ),
        (
            {
                "driving", "delivery", "vehicle", "manual handling", "lifting",
                "warehouse", "logistics", "customs", "import export",
            },
            "logistics",
        ),
        (
            {
                "customer service", "customer support", "communication",
                "stakeholder",
            },
            "general_professional",
        ),
    ]

    for keywords, category in keyword_categories:
        for keyword in keywords:
            if keyword in label_norm:
                return category

    return "general"


def infer_weight(label, category):
    label_norm = normalise_text(label)

    high_priority_keywords = {
        "python", "sql", "java", "javascript", "typescript", "csharp", "dotnet",
        "azure", "aws", "postgresql", "graphql", "rest api", "devops", "ci/cd",
        "oauth2", "role based access control", "react", "angular", "vue",
        "kubernetes", "docker",
    }

    medium_priority_keywords = {
        "documentation", "reporting", "git", "version control", "automated testing",
        "code review", "machine learning", "natural language processing",
        "data cleaning", "data visualisation", "excel", "customer service",
        "record keeping", "compliance", "driving", "vehicle inspection",
        "manual handling", "problem solving", "troubleshooting", "administration",
        "customs", "logistics", "import export administration", "file archiving",
        "css", "html",
    }

    for keyword in high_priority_keywords:
        if keyword in label_norm:
            return 3

    for keyword in medium_priority_keywords:
        if keyword in label_norm:
            return 2

    if category in {"software", "database", "cloud", "security", "data_ai"}:
        return 2

    return 1


def build_esco_dictionary():
    esco_df = pd.read_csv(ESCO_SKILLS_CLEANED_PATH)

    grouped = defaultdict(
        lambda: {
            "aliases": set(),
            "esco_labels": set(),
            "category": "general",
            "weight": 1,
        }
    )

    for _, row in esco_df.iterrows():
        preferred_label = row.get("preferred_label")
        alt_labels = row.get("alt_labels")

        if pd.isna(preferred_label):
            continue

        preferred_label = str(preferred_label).strip()

        if not preferred_label or is_blocked_phrase(preferred_label):
            continue

        canonical = standardise_canonical_skill(preferred_label)

        if is_blocked_phrase(canonical):
            continue

        category = infer_category(canonical)
        weight = infer_weight(canonical, category)

        grouped[canonical]["aliases"].add(preferred_label)
        grouped[canonical]["aliases"].add(canonical)
        grouped[canonical]["esco_labels"].add(preferred_label)
        grouped[canonical]["category"] = category
        grouped[canonical]["weight"] = max(grouped[canonical]["weight"], weight)

        short_form = extract_short_form(preferred_label)
        if short_form and not is_blocked_phrase(short_form):
            standardised_short_form = standardise_canonical_skill(short_form)

            if not is_blocked_phrase(standardised_short_form):
                grouped[canonical]["aliases"].add(short_form)
                grouped[canonical]["aliases"].add(standardised_short_form)

        if pd.notna(alt_labels):
            for alt_label in str(alt_labels).split("\n"):
                alt_label = alt_label.strip()

                if not alt_label:
                    continue

                if is_blocked_phrase(alt_label):
                    continue

                standardised_alt = standardise_canonical_skill(alt_label)

                if is_blocked_phrase(standardised_alt):
                    continue

                grouped[canonical]["aliases"].add(alt_label)
                grouped[canonical]["aliases"].add(standardised_alt)

                short_form_alt = extract_short_form(alt_label)
                if short_form_alt and not is_blocked_phrase(short_form_alt):
                    standardised_short_form_alt = standardise_canonical_skill(short_form_alt)

                    if not is_blocked_phrase(standardised_short_form_alt):
                        grouped[canonical]["aliases"].add(short_form_alt)
                        grouped[canonical]["aliases"].add(standardised_short_form_alt)

    final_dict = {}

    for canonical, info in grouped.items():
        if is_blocked_phrase(canonical):
            continue

        aliases = sorted(
            {
                str(alias).strip().lower()
                for alias in info["aliases"]
                if str(alias).strip() and not is_blocked_phrase(alias)
            }
        )

        esco_labels = sorted(
            {
                str(label).strip()
                for label in info["esco_labels"]
                if str(label).strip()
            }
        )

        if not aliases:
            continue

        final_dict[canonical] = {
            "aliases": aliases,
            "category": info["category"],
            "weight": info["weight"],
            "esco_labels": esco_labels,
        }

    return final_dict


def save_dictionary(dictionary_data):
    ARTEFACT_GENERATED_DICTIONARY_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(ARTEFACT_GENERATED_DICTIONARY_PATH, "w", encoding="utf-8") as file:
        json.dump(dictionary_data, file, indent=2, ensure_ascii=False)


def run_build_esco_dictionary():
    print("Building filtered and standardised ESCO-informed artefact dictionary...")

    dictionary_data = build_esco_dictionary()
    save_dictionary(dictionary_data)

    print(f"Generated dictionary entries: {len(dictionary_data)}")
    print(f"Saved generated dictionary to: {ARTEFACT_GENERATED_DICTIONARY_PATH}")


if __name__ == "__main__":
    run_build_esco_dictionary()