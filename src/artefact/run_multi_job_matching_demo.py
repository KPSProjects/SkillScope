import json
import re
import time
from pathlib import Path

from sklearn.metrics import precision_score, recall_score, f1_score

from src.artefact.load_skill_dictionary import load_active_skill_dictionary


SKILLSPAN_TEST_PATH = Path("data/raw/skillspan/test.json")
OUTPUT_PATH = Path("data/processed/evaluation/baseline_skillspan_results.txt")


def normalise_text(text):
    # Convert text into a consistent format for matching
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


def load_skillspan_jsonl(path):
    # Load SkillSpan rows from the raw JSONL-style file
    rows = []

    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))

    return rows


def combine_true_tags(tags_skill, tags_knowledge):
    # SkillSpan has separate skill and knowledge tags.
    # For this evaluation, both are treated as valid positive labels.
    combined_tags = []

    for skill_tag, knowledge_tag in zip(tags_skill, tags_knowledge):
        if skill_tag != "O":
            combined_tags.append(skill_tag)
        elif knowledge_tag != "O":
            combined_tags.append(knowledge_tag)
        else:
            combined_tags.append("O")

    return combined_tags


def build_alias_lookup(skill_dictionary):
    # Build a faster alias lookup so we do not loop through the whole dictionary for every row
    alias_lookup = {}

    for skill_name, skill_data in skill_dictionary.items():
        aliases = skill_data.get("aliases", [])

        for alias in aliases:
            alias_norm = normalise_text(alias)

            if not alias_norm:
                continue

            alias_parts = tuple(alias_norm.split())

            if not alias_parts:
                continue

            alias_lookup[alias_parts] = skill_name

    return alias_lookup


def predict_keyword_tags(tokens, alias_lookup):
    # Predict BIO labels by checking if token windows match known aliases
    predicted_tags = ["O"] * len(tokens)
    normalised_tokens = [normalise_text(token) for token in tokens]

    alias_lengths = sorted(
        {len(alias_parts) for alias_parts in alias_lookup.keys()},
        reverse=True,
    )

    for alias_length in alias_lengths:
        if alias_length > len(normalised_tokens):
            continue

        for i in range(0, len(normalised_tokens) - alias_length + 1):
            token_window = tuple(normalised_tokens[i:i + alias_length])

            if token_window in alias_lookup:
                predicted_tags[i] = "B"

                for j in range(i + 1, i + alias_length):
                    predicted_tags[j] = "I"

    return predicted_tags


def bio_to_binary(tags):
    # Convert BIO labels into binary labels:
    # 1 = skill/knowledge token
    # 0 = not a skill/knowledge token
    return [0 if tag == "O" else 1 for tag in tags]


def evaluate_baseline_on_skillspan():
    start_time = time.time()

    if not SKILLSPAN_TEST_PATH.exists():
        raise FileNotFoundError(f"SkillSpan test file not found: {SKILLSPAN_TEST_PATH}")

    print("Evaluating keyword baseline on SkillSpan test set...")
    print(f"Using file: {SKILLSPAN_TEST_PATH}")

    skill_dictionary = load_active_skill_dictionary()
    alias_lookup = build_alias_lookup(skill_dictionary)
    rows = load_skillspan_jsonl(SKILLSPAN_TEST_PATH)

    print(f"Loaded SkillSpan rows: {len(rows)}")
    print(f"Loaded dictionary skills: {len(skill_dictionary)}")
    print(f"Loaded aliases for matching: {len(alias_lookup)}")

    all_true_binary = []
    all_pred_binary = []

    rows_used = 0

    for row in rows:
        tokens = row.get("tokens", [])
        tags_skill = row.get("tags_skill", [])
        tags_knowledge = row.get("tags_knowledge", [])

        if not tokens or not tags_skill or not tags_knowledge:
            continue

        if not (len(tokens) == len(tags_skill) == len(tags_knowledge)):
            continue

        true_tags = combine_true_tags(tags_skill, tags_knowledge)
        predicted_tags = predict_keyword_tags(tokens, alias_lookup)

        all_true_binary.extend(bio_to_binary(true_tags))
        all_pred_binary.extend(bio_to_binary(predicted_tags))

        rows_used += 1

    precision = precision_score(all_true_binary, all_pred_binary, zero_division=0)
    recall = recall_score(all_true_binary, all_pred_binary, zero_division=0)
    f1 = f1_score(all_true_binary, all_pred_binary, zero_division=0)

    elapsed_time = time.time() - start_time

    print("\nBASELINE KEYWORD EVALUATION RESULTS")
    print("===================================")
    print(f"Rows evaluated: {rows_used}")
    print(f"Tokens evaluated: {len(all_true_binary)}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Runtime: {elapsed_time:.2f} seconds")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
        file.write("BASELINE KEYWORD EVALUATION ON SKILLSPAN\n")
        file.write("========================================\n\n")
        file.write(f"Rows evaluated: {rows_used}\n")
        file.write(f"Tokens evaluated: {len(all_true_binary)}\n")
        file.write(f"Precision: {precision:.4f}\n")
        file.write(f"Recall: {recall:.4f}\n")
        file.write(f"F1 Score: {f1:.4f}\n")
        file.write(f"Runtime: {elapsed_time:.2f} seconds\n")

    print(f"\nSaved results to: {OUTPUT_PATH}")


if __name__ == "__main__":
    evaluate_baseline_on_skillspan()