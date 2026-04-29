import json
import re
from pathlib import Path

from sklearn.metrics import precision_score, recall_score, f1_score

from src.artefact.load_skill_dictionary import load_active_skill_dictionary


SKILLSPAN_TEST_PATH = Path("data/raw/skillspan/test.json")


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


def load_skillspan_jsonl(path):
    rows = []

    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))

    return rows


def combine_true_tags(tags_skill, tags_knowledge):
    """
    SkillSpan has two label columns:
    - tags_skill
    - tags_knowledge

    For this evaluation, both are treated as positive skill/knowledge evidence.
    """
    combined = []

    for skill_tag, knowledge_tag in zip(tags_skill, tags_knowledge):
        if skill_tag != "O":
            combined.append(skill_tag)
        elif knowledge_tag != "O":
            combined.append(knowledge_tag)
        else:
            combined.append("O")

    return combined


def predict_keyword_tags(tokens, skill_dictionary):
    """
    Creates simple BIO predictions from the active keyword/manual/ESCO-informed dictionary.

    This evaluates whether the keyword matcher can identify labelled SkillSpan tokens.
    """
    predicted = ["O"] * len(tokens)

    normalised_tokens = [normalise_text(token) for token in tokens]

    for skill_name, skill_data in skill_dictionary.items():
        aliases = skill_data.get("aliases", [])

        for alias in aliases:
            alias_norm = normalise_text(alias)

            if not alias_norm:
                continue

            alias_parts = alias_norm.split()

            if not alias_parts:
                continue

            alias_length = len(alias_parts)

            for i in range(0, len(normalised_tokens) - alias_length + 1):
                token_window = normalised_tokens[i:i + alias_length]

                if token_window == alias_parts:
                    predicted[i] = "B"

                    for j in range(i + 1, i + alias_length):
                        predicted[j] = "I"

    return predicted


def bio_to_binary(tags):
    """
    Converts BIO tags to binary:
    1 = skill/knowledge token
    0 = non-skill token
    """
    return [0 if tag == "O" else 1 for tag in tags]


def evaluate_baseline_on_skillspan():
    if not SKILLSPAN_TEST_PATH.exists():
        raise FileNotFoundError(f"SkillSpan test file not found: {SKILLSPAN_TEST_PATH}")

    print("Evaluating keyword baseline on SkillSpan test set...")
    print(f"Using file: {SKILLSPAN_TEST_PATH}")

    skill_dictionary = load_active_skill_dictionary()
    rows = load_skillspan_jsonl(SKILLSPAN_TEST_PATH)

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
        predicted_tags = predict_keyword_tags(tokens, skill_dictionary)

        all_true_binary.extend(bio_to_binary(true_tags))
        all_pred_binary.extend(bio_to_binary(predicted_tags))

        rows_used += 1

    precision = precision_score(all_true_binary, all_pred_binary, zero_division=0)
    recall = recall_score(all_true_binary, all_pred_binary, zero_division=0)
    f1 = f1_score(all_true_binary, all_pred_binary, zero_division=0)

    print("\nBASELINE KEYWORD EVALUATION RESULTS")
    print("===================================")
    print(f"Rows evaluated: {rows_used}")
    print(f"Tokens evaluated: {len(all_true_binary)}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    output_path = Path("data/processed/evaluation/baseline_skillspan_results.txt")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as file:
        file.write("BASELINE KEYWORD EVALUATION ON SKILLSPAN\n")
        file.write("========================================\n\n")
        file.write(f"Rows evaluated: {rows_used}\n")
        file.write(f"Tokens evaluated: {len(all_true_binary)}\n")
        file.write(f"Precision: {precision:.4f}\n")
        file.write(f"Recall: {recall:.4f}\n")
        file.write(f"F1 Score: {f1:.4f}\n")

    print(f"\nSaved results to: {output_path}")


if __name__ == "__main__":
    evaluate_baseline_on_skillspan()