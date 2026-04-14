import ast
import re
import pandas as pd

from src.config import (
    SKILLSPAN_TEST_PATH,
    ESCO_SKILLS_CLEANED_PATH,
    BASELINE_EVAL_SUMMARY_PATH,
    BASELINE_EVAL_EXAMPLES_PATH,
)


def load_skillspan_file(path) -> pd.DataFrame:
    """
    Loads one SkillSpan JSON Lines file into a dataframe.
    """
    return pd.read_json(path, lines=True)


def load_esco_cleaned() -> pd.DataFrame:
    """
    Loads the cleaned ESCO skills file.
    """
    return pd.read_csv(ESCO_SKILLS_CLEANED_PATH)


def safe_parse_list(value):
    """
    Tries to turn a stored value into a Python list.
    """
    if isinstance(value, list):
        return value

    if pd.isna(value):
        return []

    if isinstance(value, str):
        value = value.strip()

        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            pass

        return [value]

    return [value]


def normalise_text(text: str) -> str:
    """
    Normalises text for simple matching/comparison.
    """
    if pd.isna(text):
        return ""

    text = str(text).lower()
    text = re.sub(r"[^a-z0-9+#/. -]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def combine_rows_by_idx(df: pd.DataFrame) -> pd.DataFrame:
    """
    Combines multiple SkillSpan rows with the same idx into one document-level row.
    """
    combined_rows = []

    for idx_value, group in df.groupby("idx", sort=True):
        all_tokens = []
        all_skill_tags = []

        source_value = group["source"].iloc[0] if "source" in group.columns else "unknown"

        for _, row in group.iterrows():
            tokens = safe_parse_list(row["tokens"])
            skill_tags = safe_parse_list(row["tags_skill"])

            all_tokens.extend(tokens)
            all_skill_tags.extend(skill_tags)

        combined_rows.append(
            {
                "idx": idx_value,
                "source": source_value,
                "tokens": all_tokens,
                "tags_skill": all_skill_tags,
            }
        )

    return pd.DataFrame(combined_rows)


def extract_bio_spans(tokens, tags):
    """
    Extracts spans from BIO tags.
    Supports:
    - B / I / O
    - B-SKILL / I-SKILL / O
    """
    spans = []
    current_tokens = []

    for token, tag in zip(tokens, tags):
        tag_str = str(tag).upper()

        is_begin = tag_str == "B" or tag_str.startswith("B-")
        is_inside = tag_str == "I" or tag_str.startswith("I-")
        is_outside = tag_str == "O"

        if is_begin:
            if current_tokens:
                spans.append(" ".join(current_tokens))
            current_tokens = [str(token)]

        elif is_inside:
            if current_tokens:
                current_tokens.append(str(token))
            else:
                current_tokens = [str(token)]

        elif is_outside:
            if current_tokens:
                spans.append(" ".join(current_tokens))
                current_tokens = []

        else:
            if current_tokens:
                spans.append(" ".join(current_tokens))
                current_tokens = []

    if current_tokens:
        spans.append(" ".join(current_tokens))

    return spans


def build_esco_phrase_lookup(esco_df: pd.DataFrame) -> set[str]:
    """
    Builds a set of normalised ESCO phrases from preferred and alt labels.
    Keeps only phrases with length >= 3 to match the improved baseline logic.
    """
    phrases = set()

    for _, row in esco_df.iterrows():
        preferred_label = row.get("preferred_label", "")
        alt_labels = row.get("alt_labels", "")

        if pd.notna(preferred_label):
            norm_pref = normalise_text(preferred_label)
            if len(norm_pref) >= 3:
                phrases.add(norm_pref)

        if pd.notna(alt_labels):
            alt_labels_str = str(alt_labels).strip()
            if alt_labels_str:
                for label in alt_labels_str.split("\n"):
                    norm_alt = normalise_text(label)
                    if len(norm_alt) >= 3:
                        phrases.add(norm_alt)

    return phrases


def predict_skills_from_text(text: str, esco_phrases: set[str]) -> list[str]:
    """
    Very simple baseline prediction:
    returns ESCO phrases that appear in the text using exact boundary matching.
    """
    text_norm = normalise_text(text)
    matched = []

    for phrase in esco_phrases:
        pattern = rf"(?<!\w){re.escape(phrase)}(?!\w)"
        if re.search(pattern, text_norm):
            matched.append(phrase)

    return sorted(set(matched))


def evaluate_documents(doc_df: pd.DataFrame, esco_phrases: set[str], max_docs: int = 20):
    """
    Evaluates a limited number of SkillSpan documents against the baseline matcher.
    """
    results = []

    for _, row in doc_df.head(max_docs).iterrows():
        tokens = row["tokens"]
        tags_skill = row["tags_skill"]

        gold_skills_raw = extract_bio_spans(tokens, tags_skill)
        gold_skills = sorted(set(normalise_text(skill) for skill in gold_skills_raw if normalise_text(skill)))

        text = " ".join(str(token) for token in tokens)
        predicted_skills = predict_skills_from_text(text, esco_phrases)

        gold_set = set(gold_skills)
        pred_set = set(predicted_skills)
        overlap_set = gold_set.intersection(pred_set)

        precision = len(overlap_set) / len(pred_set) if pred_set else 0.0
        recall = len(overlap_set) / len(gold_set) if gold_set else 0.0

        results.append(
            {
                "idx": row["idx"],
                "source": row["source"],
                "gold_count": len(gold_set),
                "predicted_count": len(pred_set),
                "overlap_count": len(overlap_set),
                "precision": precision,
                "recall": recall,
                "gold_skills": sorted(gold_set),
                "predicted_skills": sorted(pred_set),
                "overlap_skills": sorted(overlap_set),
                "text_sample": text[:500],
            }
        )

    return results


def build_summary_text(results: list[dict], evaluated_docs: int) -> str:
    """
    Builds a readable summary text for the evaluation.
    """
    avg_precision = sum(r["precision"] for r in results) / len(results) if results else 0.0
    avg_recall = sum(r["recall"] for r in results) / len(results) if results else 0.0
    total_gold = sum(r["gold_count"] for r in results)
    total_pred = sum(r["predicted_count"] for r in results)
    total_overlap = sum(r["overlap_count"] for r in results)

    lines = []
    lines.append("BASELINE VS SKILLSPAN EVALUATION SUMMARY")
    lines.append("=======================================")
    lines.append("")
    lines.append(f"Documents evaluated: {evaluated_docs}")
    lines.append(f"Total gold skills: {total_gold}")
    lines.append(f"Total predicted skills: {total_pred}")
    lines.append(f"Total overlap skills: {total_overlap}")
    lines.append(f"Average precision: {avg_precision:.4f}")
    lines.append(f"Average recall: {avg_recall:.4f}")
    lines.append("")
    lines.append("Note: this is an approximate exact-string-overlap evaluation.")
    lines.append("SkillSpan gold spans and ESCO dictionary phrases may express the same skill differently.")
    lines.append("")

    return "\n".join(lines)


def build_examples_text(results: list[dict], max_examples: int = 10) -> str:
    """
    Builds readable example outputs for inspection.
    """
    lines = []
    lines.append("BASELINE VS SKILLSPAN EXAMPLES")
    lines.append("==============================")
    lines.append("")

    for result in results[:max_examples]:
        lines.append(f"idx: {result['idx']}")
        lines.append(f"source: {result['source']}")
        lines.append(f"gold_count: {result['gold_count']}")
        lines.append(f"predicted_count: {result['predicted_count']}")
        lines.append(f"overlap_count: {result['overlap_count']}")
        lines.append(f"precision: {result['precision']:.4f}")
        lines.append(f"recall: {result['recall']:.4f}")
        lines.append(f"gold_skills: {result['gold_skills'][:15]}")
        lines.append(f"predicted_skills: {result['predicted_skills'][:15]}")
        lines.append(f"overlap_skills: {result['overlap_skills'][:15]}")
        lines.append(f"text_sample: {result['text_sample']}")
        lines.append("")

    return "\n".join(lines)


def run_baseline_evaluation():
    """
    Main function for the first small-scale baseline evaluation.
    """
    print("Starting baseline vs SkillSpan evaluation...")

    skillspan_test_df = load_skillspan_file(SKILLSPAN_TEST_PATH)
    skillspan_test_docs = combine_rows_by_idx(skillspan_test_df)

    esco_df = load_esco_cleaned()
    esco_phrases = build_esco_phrase_lookup(esco_df)

    print(f"SkillSpan test rows: {len(skillspan_test_df)}")
    print(f"SkillSpan test documents: {len(skillspan_test_docs)}")
    print(f"ESCO phrase count: {len(esco_phrases)}")

    # Evaluate on a small subset first
    results = evaluate_documents(skillspan_test_docs, esco_phrases, max_docs=20)

    summary_text = build_summary_text(results, evaluated_docs=len(results))
    examples_text = build_examples_text(results, max_examples=10)

    print("\n--- EVALUATION SUMMARY ---")
    print(summary_text)

    print("\n--- EVALUATION EXAMPLES PREVIEW ---")
    print(examples_text[:5000])

    BASELINE_EVAL_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(BASELINE_EVAL_SUMMARY_PATH, "w", encoding="utf-8") as file:
        file.write(summary_text)

    with open(BASELINE_EVAL_EXAMPLES_PATH, "w", encoding="utf-8") as file:
        file.write(examples_text)

    print(f"\nSaved summary to: {BASELINE_EVAL_SUMMARY_PATH}")
    print(f"Saved examples to: {BASELINE_EVAL_EXAMPLES_PATH}")