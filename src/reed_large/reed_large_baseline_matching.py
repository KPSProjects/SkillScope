import re
import time
import pandas as pd
from numpy.distutils.fcompiler import none

from src.config import (
    REED_UK_LARGE_CLEANED_PATH,
    ESCO_SKILLS_CLEANED_PATH,
    REED_LARGE_MATCHED_SKILLS_V3_PATH,
    REED_LARGE_SKILL_FREQUENCY_V3_PATH,
    REED_LARGE_SKILL_FREQUENCY_FILTERED_V3_PATH,
)


def normalise_text(text):
    text = str(text).lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def simple_tokenise(text):
    return re.findall(r"\w+", str(text).lower())


def load_reed_large_cleaned():
    return pd.read_csv(REED_UK_LARGE_CLEANED_PATH)


def load_esco_data():
    return pd.read_csv(ESCO_SKILLS_CLEANED_PATH)


def is_allowed_preferred_label(preferred_label):
    label_norm = normalise_text(preferred_label)

    banned_preferred_labels = {
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
        "company policies",
        "meet commitments",
        "build trust",
        "demonstrate trustworthiness",
        "demonstrate willingness to learn",
    }

    return label_norm not in banned_preferred_labels


def is_strong_baseline_phrase(phrase):
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
        "company policies",
        "meet commitments",
        "build trust",
        "demonstrate trustworthiness",
        "demonstrate willingness to learn",
    }

    banned_starts = [
        "ability to ",
        "knowledge of ",
        "understanding of ",
        "experience of ",
        "experience in ",
        "work in ",
        "working in ",
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
        "build ",
        "meet ",
    ]

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

    weak_single_words = {
        "communication",
        "management",
        "support",
        "delivery",
        "operations",
        "planning",
        "services",
        "development",
        "training",
        "monitoring",
        "reporting",
        "design",
        "analysis",
        "care",
        "teaching",
        "sales",
        "marketing",
        "leadership",
        "responsibility",
        "change",
        "commitment",
        "stress",
        "trust",
        "trustworthiness",
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
        "health and safety in the workplace",
        "production processes",
        "food engineering",
        "mechanical engineering",
    }

    if phrase_norm in banned_exact_phrases:
        return False

    if any(phrase_norm.startswith(prefix) for prefix in banned_starts):
        return False

    word_count = len(phrase_norm.split())

    if word_count == 1:
        if phrase_norm in strong_single_words:
            return True
        if phrase_norm in weak_single_words:
            return False
        return False

    if word_count >= 2:
        if phrase_norm in allowed_multiword_exceptions:
            return True

        vague_words = {
            "responsibility",
            "change",
            "commitment",
            "stress",
            "professional",
            "personal",
            "independently",
            "efficiently",
            "team",
            "deadline",
            "deadlines",
            "flexible",
            "manner",
            "customers",
            "trust",
            "trustworthiness",
            "willingness",
            "learn",
            "commitments",
        }

        if any(word in phrase_norm.split() for word in vague_words):
            return False

        return True

    return False


def build_esco_phrase_table(esco_df):
    phrase_rows = []

    for _, row in esco_df.iterrows():
        preferred_label = row.get("preferred_label")
        alt_labels = row.get("alt_labels")

        if pd.isna(preferred_label):
            continue

        preferred_label = str(preferred_label).strip()

        if not is_allowed_preferred_label(preferred_label):
            continue

        if preferred_label and is_strong_baseline_phrase(preferred_label):
            phrase_rows.append(
                {
                    "phrase": preferred_label,
                    "preferred_label": preferred_label,
                    "match_type": "preferred",
                }
            )

        if pd.notna(alt_labels):
            for alt_label in str(alt_labels).split("\n"):
                alt_label = alt_label.strip()

                if alt_label and is_strong_baseline_phrase(alt_label):
                    phrase_rows.append(
                        {
                            "phrase": alt_label,
                            "preferred_label": preferred_label,
                            "match_type": "alt",
                        }
                    )

    phrase_df = pd.DataFrame(phrase_rows).drop_duplicates()

    if phrase_df.empty:
        return pd.DataFrame(columns=[
            "phrase",
            "preferred_label",
            "match_type",
            "phrase_norm",
            "word_count",
            "first_token",
        ])

    phrase_df["phrase_norm"] = phrase_df["phrase"].apply(normalise_text)
    phrase_df["word_count"] = phrase_df["phrase_norm"].apply(lambda x: len(x.split()))
    phrase_df["first_token"] = phrase_df["phrase_norm"].apply(
        lambda x: x.split()[0] if x else ""
    )

    phrase_df = phrase_df[phrase_df["phrase_norm"].str.len() >= 3].copy()
    phrase_df = phrase_df.sort_values(by="word_count", ascending=False).reset_index(drop=True)

    return phrase_df


def build_phrase_index(phrase_df):
    phrase_index = {}

    for _, row in phrase_df.iterrows():
        first_token = row["first_token"]

        if first_token not in phrase_index:
            phrase_index[first_token] = []

        phrase_index[first_token].append(
            {
                "phrase": row["phrase"],
                "phrase_norm": row["phrase_norm"],
                "preferred_label": row["preferred_label"],
                "match_type": row["match_type"],
            }
        )

    return phrase_index


def choose_text(row):
    requirements = row.get("job_requirements")
    description = row.get("job_description")

    if pd.notna(requirements):
        requirements = str(requirements).strip()
        if len(requirements) > 30:
            return requirements

    if pd.notna(description):
        return str(description)

    return ""


def phrase_in_text(phrase_norm, text_norm):
    pattern = r"\b" + re.escape(phrase_norm) + r"\b"
    return re.search(pattern, text_norm) is not None


def match_reed_large_to_esco(reed_df, phrase_index, max_rows=1000):
    matched_rows = []

    working_df = reed_df.copy()
    if max_rows is not None:
        working_df = working_df.head(max_rows).copy()

    total_rows = len(working_df)
    start_time = time.time()

    print(f"Rows selected for baseline matching: {total_rows}")

    for i, (_, row) in enumerate(working_df.iterrows(), start=1):
        text = choose_text(row)
        text_norm = normalise_text(text)

        if len(text_norm) < 30:
            continue

        job_title = row.get("job_title", "")
        category = row.get("category", "")
        post_date = row.get("post_date", "")

        seen_labels = set()

        text_tokens = set(simple_tokenise(text_norm))
        candidate_phrases = []

        for token in text_tokens:
            if token in phrase_index:
                candidate_phrases.extend(phrase_index[token])

        unique_candidates = {}
        for phrase_row in candidate_phrases:
            key = (phrase_row["phrase_norm"], phrase_row["preferred_label"])
            if key not in unique_candidates:
                unique_candidates[key] = phrase_row

        candidate_list = list(unique_candidates.values())
        candidate_list.sort(key=lambda x: len(x["phrase_norm"].split()), reverse=True)

        for phrase_row in candidate_list:
            phrase_norm = phrase_row["phrase_norm"]
            preferred_label = phrase_row["preferred_label"]
            match_type = phrase_row["match_type"]

            if preferred_label in seen_labels:
                continue

            if phrase_in_text(phrase_norm, text_norm):
                matched_rows.append(
                    {
                        "job_title": job_title,
                        "category": category,
                        "post_date": post_date,
                        "matched_phrase": phrase_row["phrase"],
                        "preferred_label": preferred_label,
                        "match_type": match_type,
                    }
                )
                seen_labels.add(preferred_label)

        if i % 100 == 0 or i == total_rows:
            elapsed = time.time() - start_time
            avg_per_row = elapsed / i
            remaining_rows = total_rows - i
            eta_seconds = avg_per_row * remaining_rows

            print(
                f"Processed {i}/{total_rows} rows | "
                f"elapsed: {elapsed:.2f}s | "
                f"rough ETA: {eta_seconds:.2f}s | "
                f"matches so far: {len(matched_rows)}"
            )

    return pd.DataFrame(matched_rows)


def create_skill_frequency_summary(matches_df):
    if matches_df.empty:
        return pd.DataFrame(columns=["preferred_label", "match_count"])

    summary_df = (
        matches_df.groupby("preferred_label")
        .size()
        .reset_index(name="match_count")
        .sort_values(by="match_count", ascending=False)
        .reset_index(drop=True)
    )

    return summary_df


def filter_skill_frequency_summary(summary_df, min_count=2):
    if summary_df.empty:
        return summary_df.copy()

    filtered_df = summary_df[summary_df["match_count"] >= min_count].copy()
    filtered_df = filtered_df.sort_values(by="match_count", ascending=False).reset_index(drop=True)

    return filtered_df


def run_reed_large_baseline_matching():
    print("Starting faster baseline ESCO matching on Reed large cleaned data...")

    load_start = time.time()
    reed_df = load_reed_large_cleaned()
    esco_df = load_esco_data()
    load_end = time.time()

    print(f"Reed large cleaned rows loaded: {len(reed_df)}")
    print(f"ESCO rows loaded: {len(esco_df)}")
    print(f"Loading time: {load_end - load_start:.2f} seconds")

    phrase_start = time.time()
    phrase_df = build_esco_phrase_table(esco_df)
    phrase_index = build_phrase_index(phrase_df)
    phrase_end = time.time()

    print(f"Filtered ESCO phrase rows prepared: {len(phrase_df)}")
    print(f"Unique first-token buckets: {len(phrase_index)}")
    print(f"Phrase prep time: {phrase_end - phrase_start:.2f} seconds")

    bad_labels_to_check = [
        "assume responsibility",
        "cope with stress",
        "show commitment",
        "work independently",
        "maintenance operations",
        "product comprehension",
        "company policies",
        "meet commitments",
        "build trust",
        "demonstrate trustworthiness",
        "demonstrate willingness to learn",
    ]

    for bad_label in bad_labels_to_check:
        check_df = phrase_df[phrase_df["preferred_label"].str.lower() == bad_label]
        print(f"{bad_label}: {len(check_df)} rows in phrase table")

    match_start = time.time()
    matches_df = match_reed_large_to_esco(
        reed_df=reed_df,
        phrase_index=phrase_index,
        max_rows=None
    )
    match_end = time.time()

    print(f"Matched rows created: {len(matches_df)}")
    print(f"Matching time: {match_end - match_start:.2f} seconds")

    summary_df = create_skill_frequency_summary(matches_df)
    filtered_summary_df = filter_skill_frequency_summary(summary_df, min_count=2)

    REED_LARGE_MATCHED_SKILLS_V3_PATH.parent.mkdir(parents=True, exist_ok=True)
    REED_LARGE_SKILL_FREQUENCY_V3_PATH.parent.mkdir(parents=True, exist_ok=True)
    REED_LARGE_SKILL_FREQUENCY_FILTERED_V3_PATH.parent.mkdir(parents=True, exist_ok=True)

    matches_df.to_csv(REED_LARGE_MATCHED_SKILLS_V3_PATH, index=False)
    summary_df.to_csv(REED_LARGE_SKILL_FREQUENCY_V3_PATH, index=False)
    filtered_summary_df.to_csv(REED_LARGE_SKILL_FREQUENCY_FILTERED_V3_PATH, index=False)

    print(f"Saved matched skills to: {REED_LARGE_MATCHED_SKILLS_V3_PATH}")
    print(f"Saved skill frequency summary to: {REED_LARGE_SKILL_FREQUENCY_V3_PATH}")
    print(f"Saved filtered skill frequency summary to: {REED_LARGE_SKILL_FREQUENCY_FILTERED_V3_PATH}")