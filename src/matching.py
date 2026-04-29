import json
import re
import pandas as pd

from src.config import (
    REED_CLEANED_PATH,
    ESCO_SKILLS_CLEANED_PATH,
    WEAK_LABELLED_REED_PATH,
    WEAK_LABELLED_REED_PREVIEW_PATH,
)


def normalise_text(text):
    # make text easier to compare
    text = str(text).lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def simple_tokenise(text):
    # split text into word-like tokens and punctuation
    return re.findall(r"\w+|[^\w\s]", str(text), flags=re.UNICODE)


def load_reed_data():
    # load cleaned Reed data
    return pd.read_csv(REED_CLEANED_PATH)


def load_esco_data():
    # load cleaned ESCO data
    return pd.read_csv(ESCO_SKILLS_CLEANED_PATH)


def find_text_column(df):
    # try common text column names first
    possible_columns = [
        "job_description",
        "description",
        "advert_text",
        "text",
        "body",
        "content",
        "summary",
    ]

    for column in possible_columns:
        if column in df.columns:
            return column

    # fallback: pick the longest average text-like column
    best_column = None
    best_length = 0

    for column in df.columns:
        if df[column].dtype == "object":
            sample = df[column].dropna().astype(str).head(100)
            if not sample.empty:
                avg_len = sample.str.len().mean()
                if avg_len > best_length:
                    best_length = avg_len
                    best_column = column

    if best_column is None:
        raise ValueError("Could not find a suitable text column in the Reed dataset.")

    return best_column


def build_esco_phrase_list(esco_df):
    # build one list of ESCO preferred and alt labels
    phrases = set()

    for _, row in esco_df.iterrows():
        preferred_label = row.get("preferred_label")
        alt_labels = row.get("alt_labels")

        if pd.notna(preferred_label):
            phrase = str(preferred_label).strip()
            if phrase:
                phrases.add(phrase)

        if pd.notna(alt_labels):
            for alt_label in str(alt_labels).split("\n"):
                alt_label = alt_label.strip()
                if alt_label:
                    phrases.add(alt_label)

    # keep phrases that are not tiny
    cleaned_phrases = []
    for phrase in phrases:
        if len(phrase.strip()) >= 3:
            cleaned_phrases.append(phrase)

    # longest first helps multi-word phrases match before tiny ones
    cleaned_phrases = sorted(cleaned_phrases, key=lambda x: len(x.split()), reverse=True)

    return cleaned_phrases


def label_tokens_with_esco(tokens, esco_phrases):
    # start with everything outside a skill
    labels = ["O"] * len(tokens)

    token_norms = [normalise_text(token) for token in tokens]

    for phrase in esco_phrases:
        phrase_tokens = simple_tokenise(phrase)
        phrase_norms = [normalise_text(token) for token in phrase_tokens]

        if not phrase_norms:
            continue

        phrase_length = len(phrase_norms)

        for i in range(len(token_norms) - phrase_length + 1):
            window = token_norms[i:i + phrase_length]

            # only label if the full phrase matches
            if window == phrase_norms:
                # skip if this area is already labelled
                if any(label != "O" for label in labels[i:i + phrase_length]):
                    continue

                labels[i] = "B"
                for j in range(1, phrase_length):
                    labels[i + j] = "I"

    return labels


def build_weak_labelled_rows(df, text_column, esco_phrases, max_rows=2000):
    # build weak-labelled rows in SkillSpan-like format
    output_rows = []
    row_id = 1

    # keep only rows with real text
    working_df = df.dropna(subset=[text_column]).copy()
    working_df[text_column] = working_df[text_column].astype(str)

    # remove short rows
    working_df = working_df[working_df[text_column].str.len() > 30].copy()

    # cap the size
    working_df = working_df.head(max_rows)

    for _, row in working_df.iterrows():
        text = row[text_column]
        tokens = simple_tokenise(text)

        if len(tokens) < 3:
            continue

        tags_skill = label_tokens_with_esco(tokens, esco_phrases)

        # skip rows with no skill labels
        if "B" not in tags_skill:
            continue

        output_rows.append(
            {
                "idx": row_id,
                "tokens": tokens,
                "tags_skill": tags_skill,
                "tags_knowledge": ["O"] * len(tokens),
                "source": "reed_weak",
            }
        )

        row_id += 1

    return output_rows


def save_jsonl(rows, output_path):
    # save rows as json lines
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_preview(rows, preview_path, text_column):
    # save a readable preview
    preview_path.parent.mkdir(parents=True, exist_ok=True)

    with open(preview_path, "w", encoding="utf-8") as file:
        file.write("WEAK LABELLED REED PREVIEW\n")
        file.write("==========================\n\n")
        file.write(f"Text column used: {text_column}\n")
        file.write(f"Rows saved: {len(rows)}\n\n")

        for i, row in enumerate(rows[:5], start=1):
            file.write(f"Row {i}\n")
            file.write(f"idx: {row['idx']}\n")
            file.write(f"source: {row['source']}\n")
            file.write(f"token count: {len(row['tokens'])}\n")
            file.write(f"skill label count: {len(row['tags_skill'])}\n")
            file.write(f"tokens sample: {row['tokens'][:40]}\n")
            file.write(f"tags sample: {row['tags_skill'][:40]}\n")
            file.write("\n")


def inspect_reed_columns():
    # inspect Reed columns
    print("Inspecting Reed dataset columns...")

    reed_df = load_reed_data()

    print(f"Row count: {len(reed_df)}")
    print("\nColumns:")
    print(list(reed_df.columns))

    print("\nSample values from first row:")
    first_row = reed_df.iloc[0]

    for column in reed_df.columns:
        value = first_row[column]
        print(f"\nCOLUMN: {column}")
        print(str(value)[:500])


def run_build_weak_labels():
    print("Starting weak label generation from Reed data...")

    reed_df = load_reed_data()
    esco_df = load_esco_data()

    print(f"Reed rows loaded: {len(reed_df)}")
    print(f"ESCO rows loaded: {len(esco_df)}")

    text_column = find_text_column(reed_df)
    print(f"Using Reed text column: {text_column}")

    esco_phrases = build_esco_phrase_list(esco_df)
    print(f"ESCO phrases prepared: {len(esco_phrases)}")

    weak_rows = build_weak_labelled_rows(
        df=reed_df,
        text_column=text_column,
        esco_phrases=esco_phrases,
        max_rows=2000,
    )

    print(f"Weak-labelled rows created: {len(weak_rows)}")

    save_jsonl(weak_rows, WEAK_LABELLED_REED_PATH)
    save_preview(weak_rows, WEAK_LABELLED_REED_PREVIEW_PATH, text_column)

    print(f"Saved weak-labelled JSONL to: {WEAK_LABELLED_REED_PATH}")
    print(f"Saved preview to: {WEAK_LABELLED_REED_PREVIEW_PATH}")