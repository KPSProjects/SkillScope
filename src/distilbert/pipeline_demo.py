import re
from pathlib import Path
import pandas as pd

from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

from src.config import (
    DISTILBERT_ROW_MODEL_DIR,
    ESCO_SKILLS_CLEANED_PATH,
    DISTILBERT_PIPELINE_DEMO_PATH,
)

# original tokenizer name
MODEL_NAME = "distilbert-base-uncased"


def normalise_text(text):
    # make text easier to compare
    text = str(text).lower().strip()
    text = re.sub(r"[^a-z0-9+#/. -]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_esco_skills():
    # load cleaned ESCO skills
    return pd.read_csv(ESCO_SKILLS_CLEANED_PATH)


def build_esco_lookup(esco_df):
    # build lookup from preferred and alt labels
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

        # add alt labels
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


def map_phrase_to_esco(phrase, esco_lookup_df):
    # try exact normalised matching first
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

    # return unmatched if no match found
    return {
        "input_phrase": phrase,
        "matched": False,
        "preferred_label": None,
        "concept_uri": None,
        "match_label": None,
        "match_type": None,
    }


def clean_pipeline_phrase(phrase):
    # clean Hugging Face output text
    phrase = str(phrase).replace("##", "")
    phrase = re.sub(r"\s+", " ", phrase).strip()
    return phrase


def load_distilbert_pipeline():
    # load tokenizer from original base model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # find checkpoint folders
    model_dir = Path(DISTILBERT_ROW_MODEL_DIR)
    checkpoint_dirs = sorted(
        [p for p in model_dir.iterdir() if p.is_dir() and p.name.startswith("checkpoint-")],
        key=lambda p: int(p.name.split("-")[-1])
    )

    if not checkpoint_dirs:
        raise FileNotFoundError(
            "No checkpoint folders were found in the row-level model output directory."
        )

    # use the latest checkpoint
    latest_checkpoint = checkpoint_dirs[-1]
    print(f"Loading model from checkpoint: {latest_checkpoint}")

    # load trained model
    model = AutoModelForTokenClassification.from_pretrained(latest_checkpoint)

    # build token classification pipeline
    ner_pipeline = pipeline(
        task="token-classification",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="simple",
    )

    return ner_pipeline


def extract_skill_phrases(text, ner_pipeline):
    # run the model on the input text
    predictions = ner_pipeline(text)

    extracted_phrases = []

    for pred in predictions:
        phrase = pred.get("word", "").strip()
        phrase = clean_pipeline_phrase(phrase)

        # skip empty phrases
        if phrase:
            extracted_phrases.append(phrase)

    # remove duplicates but keep original order
    seen = set()
    unique_phrases = []

    for phrase in extracted_phrases:
        phrase_norm = normalise_text(phrase)

        if phrase_norm and phrase_norm not in seen:
            seen.add(phrase_norm)
            unique_phrases.append(phrase)

    return unique_phrases


def build_demo_text(input_text, extracted_phrases, mapping_df):
    lines = []
    lines.append("DISTILBERT + ESCO PIPELINE DEMO")
    lines.append("==============================")
    lines.append("")
    lines.append("INPUT TEXT")
    lines.append("----------")
    lines.append(input_text)
    lines.append("")
    lines.append("EXTRACTED PHRASES")
    lines.append("-----------------")
    for phrase in extracted_phrases:
        lines.append(f"- {phrase}")
    lines.append("")
    lines.append("ESCO MAPPING RESULTS")
    lines.append("--------------------")
    lines.append(mapping_df.to_string(index=False))
    lines.append("")

    return "\n".join(lines)


def run_pipeline_demo():
    # start demo
    print("Starting DistilBERT pipeline demo...")

    # sample text for testing the pipeline
    input_text = (
        "Full-time software engineer role. "
        "Required skills include Java, JavaScript, SQL, machine learning, troubleshooting, and PowerShell. "
        "Experience with agile methodologies and continuous delivery is also useful."
    )

    # load trained model pipeline
    ner_pipeline = load_distilbert_pipeline()

    # load ESCO data
    esco_df = load_esco_skills()
    esco_lookup_df = build_esco_lookup(esco_df)

    # run extraction
    extracted_phrases = extract_skill_phrases(input_text, ner_pipeline)

    print("\n--- EXTRACTED PHRASES ---")
    print(extracted_phrases)

    # map extracted phrases to ESCO
    mapping_results = []
    for phrase in extracted_phrases:
        mapping_results.append(map_phrase_to_esco(phrase, esco_lookup_df))

    mapping_df = pd.DataFrame(mapping_results)

    print("\n--- ESCO MAPPING RESULTS ---")
    print(mapping_df)

    # save a text version of the demo output
    demo_text = build_demo_text(input_text, extracted_phrases, mapping_df)

    DISTILBERT_PIPELINE_DEMO_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(DISTILBERT_PIPELINE_DEMO_PATH, "w", encoding="utf-8") as file:
        file.write(demo_text)

    print(f"\nSaved pipeline demo to: {DISTILBERT_PIPELINE_DEMO_PATH}")