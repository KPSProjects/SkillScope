import re
import pandas as pd

from src.config import (
    ESCO_SKILLS_CLEANED_PATH,
    DISTILBERT_ESCO_MAPPING_PATH,
)


def normalise_text(text):
    # make text easier to compare
    text = str(text).lower().strip()
    text = re.sub(r"[^a-z0-9+#/. -]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_esco_skills():
    # load cleaned ESCO file
    return pd.read_csv(ESCO_SKILLS_CLEANED_PATH)


def build_esco_lookup(esco_df):
    # build a simple lookup using preferred and alt labels
    lookup_rows = []

    for _, row in esco_df.iterrows():
        concept_uri = row["concept_uri"]
        preferred_label = row["preferred_label"]
        alt_labels = row["alt_labels"]

        # add preferred label
        if pd.notna(preferred_label):
            lookup_rows.append(
                {
                    "concept_uri": concept_uri,
                    "preferred_label": preferred_label,
                    "match_label": preferred_label,
                    "match_type": "preferred",
                    "match_label_norm": normalise_text(preferred_label),
                }
            )

        # add alt labels if they exist
        if pd.notna(alt_labels):
            for alt_label in str(alt_labels).split("\n"):
                alt_label = alt_label.strip()
                if alt_label:
                    lookup_rows.append(
                        {
                            "concept_uri": concept_uri,
                            "preferred_label": preferred_label,
                            "match_label": alt_label,
                            "match_type": "alt",
                            "match_label_norm": normalise_text(alt_label),
                        }
                    )

    lookup_df = pd.DataFrame(lookup_rows)

    # remove duplicates
    lookup_df = lookup_df.drop_duplicates(
        subset=["concept_uri", "match_label_norm"]
    ).reset_index(drop=True)

    return lookup_df


def extract_spans_from_bio(tokens, tags):
    # turn B/I/O labels into extracted skill phrases
    spans = []
    current_tokens = []

    for token, tag in zip(tokens, tags):
        tag = str(tag)

        if tag == "B":
            if current_tokens:
                spans.append(" ".join(current_tokens))
            current_tokens = [str(token)]

        elif tag == "I":
            if current_tokens:
                current_tokens.append(str(token))
            else:
                current_tokens = [str(token)]

        else:
            if current_tokens:
                spans.append(" ".join(current_tokens))
                current_tokens = []

    if current_tokens:
        spans.append(" ".join(current_tokens))

    return spans


def map_phrase_to_esco(phrase, esco_lookup_df):
    # try exact normalised match first
    phrase_norm = normalise_text(phrase)

    exact_matches = esco_lookup_df[
        esco_lookup_df["match_label_norm"] == phrase_norm
    ]

    if not exact_matches.empty:
        first_match = exact_matches.iloc[0]
        return {
            "input_phrase": phrase,
            "matched": True,
            "preferred_label": first_match["preferred_label"],
            "concept_uri": first_match["concept_uri"],
            "match_label": first_match["match_label"],
            "match_type": first_match["match_type"],
        }

    # if nothing matches, return unmapped
    return {
        "input_phrase": phrase,
        "matched": False,
        "preferred_label": None,
        "concept_uri": None,
        "match_label": None,
        "match_type": None,
    }


def run_esco_mapping_demo():
    print("Starting DistilBERT to ESCO mapping demo...")

    # sample extracted phrases for demo
    extracted_phrases = [
        "java",
        "javascript",
        "powershell",
        "cloud-based application",
        "agile methodologies",
        "continuous delivery",
        "troubleshooting",
        "support developers",
        "sql",
        "machine learning",
    ]

    esco_df = load_esco_skills()
    esco_lookup_df = build_esco_lookup(esco_df)

    print(f"Loaded ESCO rows: {len(esco_df)}")
    print(f"Built ESCO lookup rows: {len(esco_lookup_df)}")

    results = []
    for phrase in extracted_phrases:
        result = map_phrase_to_esco(phrase, esco_lookup_df)
        results.append(result)

    results_df = pd.DataFrame(results)

    print("\n--- DISTILBERT TO ESCO MAPPING PREVIEW ---")
    print(results_df)

    DISTILBERT_ESCO_MAPPING_PATH.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(DISTILBERT_ESCO_MAPPING_PATH, index=False)

    print(f"\nSaved mapping preview to: {DISTILBERT_ESCO_MAPPING_PATH}")