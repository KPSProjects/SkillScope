import json
from pathlib import Path

from src.config import (
    SKILLSPAN_TRAIN_PATH,
    SKILLSPAN_DEV_PATH,
    SKILLSPAN_TEST_PATH,
    SPACY_TRAIN_DATA_PATH,
    SPACY_DEV_DATA_PATH,
    SPACY_TEST_DATA_PATH,
    SPACY_TRAINING_SUMMARY_PATH,
)


def load_skillspan_json(path: Path):
    # load skillspan data
    # this handles both json arrays and jsonl style files
    with open(path, "r", encoding="utf-8") as file:
        raw_text = file.read().strip()

    if not raw_text:
        return []

    # if it starts with [ then it is a normal json list
    if raw_text.startswith("["):
        return json.loads(raw_text)

    # otherwise read it as jsonl
    rows = []
    for line in raw_text.splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))

    return rows


def join_tokens(tokens):
    # rebuild the text from tokens and track character positions
    text_parts = []
    token_char_spans = []
    current_pos = 0

    for i, token in enumerate(tokens):
        token_text = str(token)

        if i > 0:
            text_parts.append(" ")
            current_pos += 1

        start_char = current_pos
        text_parts.append(token_text)
        current_pos += len(token_text)
        end_char = current_pos

        token_char_spans.append((start_char, end_char))

    text = "".join(text_parts)
    return text, token_char_spans


def bio_to_spans(tokens, tags, label_name="SKILL"):
    # turn BIO labels into spaCy character spans
    text, token_char_spans = join_tokens(tokens)
    entities = []

    current_start = None
    current_end = None

    for i, tag in enumerate(tags):
        if tag == "B":
            if current_start is not None:
                entities.append((current_start, current_end, label_name))

            current_start = token_char_spans[i][0]
            current_end = token_char_spans[i][1]

        elif tag == "I":
            if current_start is not None:
                current_end = token_char_spans[i][1]

        else:
            if current_start is not None:
                entities.append((current_start, current_end, label_name))
                current_start = None
                current_end = None

    if current_start is not None:
        entities.append((current_start, current_end, label_name))

    return text, {"entities": entities}


def convert_skillspan_split(skillspan_rows):
    # convert one dataset split into spaCy training format
    spacy_rows = []

    for row in skillspan_rows:
        tokens = row.get("tokens", [])
        tags_skill = row.get("tags_skill", [])

        if not tokens or not tags_skill:
            continue

        if len(tokens) != len(tags_skill):
            continue

        text, annotations = bio_to_spans(tokens, tags_skill, label_name="SKILL")
        spacy_rows.append([text, annotations])

    return spacy_rows


def save_spacy_json(data, output_path: Path):
    # save converted spaCy data
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def count_skill_entities(spacy_rows):
    # count all skill entities in a split
    total = 0
    for _, annotations in spacy_rows:
        total += len(annotations.get("entities", []))
    return total


def save_summary(train_data, dev_data, test_data):
    # save a small summary file
    SPACY_TRAINING_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(SPACY_TRAINING_SUMMARY_PATH, "w", encoding="utf-8") as file:
        file.write("SPACY SKILLSPAN PREPARATION SUMMARY\n")
        file.write("==================================\n\n")

        file.write(f"Train documents: {len(train_data)}\n")
        file.write(f"Train skill spans: {count_skill_entities(train_data)}\n\n")

        file.write(f"Dev documents: {len(dev_data)}\n")
        file.write(f"Dev skill spans: {count_skill_entities(dev_data)}\n\n")

        file.write(f"Test documents: {len(test_data)}\n")
        file.write(f"Test skill spans: {count_skill_entities(test_data)}\n\n")

        file.write("Sample training example\n")
        file.write("-----------------------\n")
        if train_data:
            sample_text, sample_ann = train_data[0]
            file.write(f"Text: {sample_text[:800]}\n\n")
            file.write(f"Entities: {sample_ann.get('entities', [])[:20]}\n")


def run_prepare_skillspan_spacy():
    print("Preparing SkillSpan data for spaCy...")

    train_rows = load_skillspan_json(SKILLSPAN_TRAIN_PATH)
    dev_rows = load_skillspan_json(SKILLSPAN_DEV_PATH)
    test_rows = load_skillspan_json(SKILLSPAN_TEST_PATH)

    print(f"Loaded SkillSpan train rows: {len(train_rows)}")
    print(f"Loaded SkillSpan dev rows: {len(dev_rows)}")
    print(f"Loaded SkillSpan test rows: {len(test_rows)}")

    train_data = convert_skillspan_split(train_rows)
    dev_data = convert_skillspan_split(dev_rows)
    test_data = convert_skillspan_split(test_rows)

    print(f"Converted train rows: {len(train_data)}")
    print(f"Converted dev rows: {len(dev_data)}")
    print(f"Converted test rows: {len(test_data)}")

    save_spacy_json(train_data, SPACY_TRAIN_DATA_PATH)
    save_spacy_json(dev_data, SPACY_DEV_DATA_PATH)
    save_spacy_json(test_data, SPACY_TEST_DATA_PATH)
    save_summary(train_data, dev_data, test_data)

    print(f"Saved spaCy train data to: {SPACY_TRAIN_DATA_PATH}")
    print(f"Saved spaCy dev data to: {SPACY_DEV_DATA_PATH}")
    print(f"Saved spaCy test data to: {SPACY_TEST_DATA_PATH}")
    print(f"Saved summary to: {SPACY_TRAINING_SUMMARY_PATH}")


if __name__ == "__main__":
    run_prepare_skillspan_spacy()