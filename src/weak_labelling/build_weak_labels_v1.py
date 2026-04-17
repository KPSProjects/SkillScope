import json
import re
import time
import pandas as pd

from src.config import (
    REED_UK_LARGE_CLEANED_PATH,
    ESCO_SKILLS_CLEANED_PATH,
    WEAK_LABELLED_REED_LARGE_V1_PATH,
    WEAK_LABELLED_REED_LARGE_V1_PREVIEW_PATH,
)


def normalise_text(text):
    # make text easier to compare
    text = str(text).lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def simple_tokenise(text):
    # split text into words and punctuation
    return re.findall(r"\w+|[^\w\s]", str(text), flags=re.UNICODE)


def split_into_chunks(text):
    # split text into smaller chunks instead of using one giant advert block
    text = str(text)

    # split on full stops, semicolons, bullet-like separators, and line breaks
    raw_chunks = re.split(r"[.\n;•]+", text)

    chunks = []
    for chunk in raw_chunks:
        chunk = " ".join(str(chunk).split()).strip()
        if chunk:
            chunks.append(chunk)

    return chunks


def looks_like_bad_chunk(chunk):
    # skip title-like or useless advert chunks
    chunk_norm = normalise_text(chunk)

    if len(chunk_norm) < 20:
        return True

    bad_starts = [
        "apply now",
        "job title",
        "location",
        "salary",
        "client details",
        "description",
        "about us",
        "job details",
    ]

    for bad in bad_starts:
        if chunk_norm.startswith(bad):
            return True

    # skip chunks that look like title/location headers
    title_like_patterns = [
        r"^[a-z ]+ - [a-z ]+$",
        r"^[a-z ]+\([a-z ]+\)$",
    ]

    for pattern in title_like_patterns:
        if re.match(pattern, chunk_norm):
            return True

    return False


def load_reed_large_cleaned():
    # load cleaned large Reed dataset
    return pd.read_csv(REED_UK_LARGE_CLEANED_PATH)


def load_esco_data():
    # load cleaned ESCO data
    return pd.read_csv(ESCO_SKILLS_CLEANED_PATH)


def is_strong_skill_phrase(phrase):
    # keep stronger phrases and avoid broad useless one-word matches
    phrase_norm = normalise_text(phrase)

    strong_single_words = {
        "sql",
        "java",
        "javascript",
        "python",
        "powershell",
        "excel",
        "accounting",
        "troubleshooting",
        "communication",
        "management",
    }

    if len(phrase_norm.split()) >= 2:
        return True

    if phrase_norm in strong_single_words:
        return True

    return False


def build_esco_phrase_list(esco_df):
    # build one filtered list of ESCO preferred and alt labels
    phrases = set()

    for _, row in esco_df.iterrows():
        preferred_label = row.get("preferred_label")
        alt_labels = row.get("alt_labels")

        if pd.notna(preferred_label):
            phrase = str(preferred_label).strip()
            if phrase and is_strong_skill_phrase(phrase):
                phrases.add(phrase)

        if pd.notna(alt_labels):
            for alt_label in str(alt_labels).split("\n"):
                alt_label = alt_label.strip()
                if alt_label and is_strong_skill_phrase(alt_label):
                    phrases.add(alt_label)

    cleaned_phrases = sorted(
        list(phrases),
        key=lambda x: len(x.split()),
        reverse=True,
    )

    return cleaned_phrases


def label_tokens_with_esco(tokens, esco_phrases):
    # start with all tokens outside a skill
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

            if window == phrase_norms:
                # skip already labelled spans
                if any(label != "O" for label in labels[i:i + phrase_length]):
                    continue

                labels[i] = "B"
                for j in range(1, phrase_length):
                    labels[i + j] = "I"

    return labels


def count_skill_spans(labels):
    # count how many B tags exist in a label row
    return sum(1 for label in labels if label == "B")


def choose_best_text(row):
    # use requirements first because they are usually more skill-heavy
    requirements = row.get("job_requirements")
    description = row.get("job_description")

    if pd.notna(requirements):
        requirements = str(requirements).strip()
        if len(requirements) > 30:
            return requirements, "job_requirements"

    return str(description), "job_description"


def build_weak_labelled_rows(df, esco_phrases, max_rows=1000):
    # build weak-labelled rows in SkillSpan-like format
    output_rows = []
    row_id = 1

    total_rows = min(len(df), max_rows)
    working_df = df.head(max_rows).reset_index(drop=True)

    build_start = time.time()

    print(f"Rows selected for weak labelling: {total_rows}")
    print("Starting filtered ESCO weak labelling...")

    for i, (_, row) in enumerate(working_df.iterrows(), start=1):
        text, source_field = choose_best_text(row)
        chunks = split_into_chunks(text)

        for chunk in chunks:
            if looks_like_bad_chunk(chunk):
                continue

            tokens = simple_tokenise(chunk)

            if len(tokens) < 3:
                continue

            tags_skill = label_tokens_with_esco(tokens, esco_phrases)

            # require at least 2 skill spans or 3 labelled tokens
            span_count = count_skill_spans(tags_skill)
            labelled_token_count = sum(1 for tag in tags_skill if tag in {"B", "I"})

            if span_count < 2 and labelled_token_count < 3:
                continue

            output_rows.append(
                {
                    "idx": row_id,
                    "tokens": tokens,
                    "tags_skill": tags_skill,
                    "tags_knowledge": ["O"] * len(tokens),
                    "source": f"reed_large_weak_{source_field}",
                }
            )

            row_id += 1

        # print progress every 25 rows
        if i % 25 == 0 or i == total_rows:
            elapsed = time.time() - build_start
            avg_per_row = elapsed / i
            remaining_rows = total_rows - i
            eta_seconds = avg_per_row * remaining_rows

            print(
                f"Processed {i}/{total_rows} rows | "
                f"elapsed: {elapsed:.2f}s | "
                f"rough ETA: {eta_seconds:.2f}s | "
                f"weak rows kept: {len(output_rows)}"
            )

    return output_rows


def save_jsonl(rows, output_path):
    # save rows as json lines
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_preview(rows, preview_path):
    # save a readable preview
    preview_path.parent.mkdir(parents=True, exist_ok=True)

    with open(preview_path, "w", encoding="utf-8") as file:
        file.write("WEAK LABELLED REED LARGE V1 PREVIEW\n")
        file.write("===================================\n\n")
        file.write(f"Rows saved: {len(rows)}\n\n")

        for i, row in enumerate(rows[:5], start=1):
            file.write(f"Row {i}\n")
            file.write(f"idx: {row['idx']}\n")
            file.write(f"source: {row['source']}\n")
            file.write(f"token count: {len(row['tokens'])}\n")
            file.write(f"skill label count: {len(row['tags_skill'])}\n")
            file.write(f"tokens sample: {row['tokens'][:50]}\n")
            file.write(f"tags sample: {row['tags_skill'][:50]}\n")
            file.write("\n")


def run_build_weak_labels_v1():
    # start total timer
    overall_start = time.time()
    print("Starting improved weak label generation from cleaned Reed large data...")

    # load data
    load_start = time.time()
    reed_df = load_reed_large_cleaned()
    esco_df = load_esco_data()
    load_end = time.time()

    print(f"Cleaned Reed large rows loaded: {len(reed_df)}")
    print(f"ESCO rows loaded: {len(esco_df)}")
    print(f"Loading time: {load_end - load_start:.2f} seconds")

    # build filtered ESCO phrase list
    esco_start = time.time()
    esco_phrases = build_esco_phrase_list(esco_df)
    esco_end = time.time()

    print(f"Filtered ESCO phrases prepared: {len(esco_phrases)}")
    print(f"ESCO phrase build time: {esco_end - esco_start:.2f} seconds")

    # build weak labels
    weak_start = time.time()
    weak_rows = build_weak_labelled_rows(
        df=reed_df,
        esco_phrases=esco_phrases,
        max_rows=500,
    )
    weak_end = time.time()

    print(f"Weak-labelled rows created: {len(weak_rows)}")
    print(f"Weak label build time: {weak_end - weak_start:.2f} seconds")

    # save outputs
    save_start = time.time()
    save_jsonl(weak_rows, WEAK_LABELLED_REED_LARGE_V1_PATH)
    save_preview(weak_rows, WEAK_LABELLED_REED_LARGE_V1_PREVIEW_PATH)
    save_end = time.time()

    print(f"Save time: {save_end - save_start:.2f} seconds")
    print(f"Saved weak-labelled JSONL to: {WEAK_LABELLED_REED_LARGE_V1_PATH}")
    print(f"Saved preview to: {WEAK_LABELLED_REED_LARGE_V1_PREVIEW_PATH}")

    # end total timer
    overall_end = time.time()
    print(f"Total elapsed time: {overall_end - overall_start:.2f} seconds")