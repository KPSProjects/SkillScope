import json
from pathlib import Path


FILES = [
    "data/processed/distilbert/weak_labelled_reed.jsonl",
    "data/processed/distilbert/weak_labelled_reed_large.jsonl",
    "data/processed/distilbert/weak_labelled_reed_large_v1.jsonl",
    "data/processed/distilbert/weak_labelled_reed_large_v2.jsonl",
    "data/processed/distilbert/weak_labelled_reed_large_v3.jsonl",
]


def count_labelled_tokens(tags):
    b_tags = sum(1 for tag in tags if str(tag).startswith("B"))
    labelled_tokens = sum(1 for tag in tags if str(tag).startswith(("B", "I")))
    return b_tags, labelled_tokens


def count_jsonl_file(path):
    path = Path(path)

    if not path.exists():
        return {
            "file": str(path),
            "exists": False,
        }

    row_count = 0
    total_tokens = 0

    total_skill_b_tags = 0
    total_skill_labelled_tokens = 0

    total_knowledge_b_tags = 0
    total_knowledge_labelled_tokens = 0

    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            row = json.loads(line)

            tokens = row.get("tokens", [])
            tags_skill = row.get("tags_skill", [])
            tags_knowledge = row.get("tags_knowledge", [])

            skill_b_tags, skill_labelled_tokens = count_labelled_tokens(tags_skill)
            knowledge_b_tags, knowledge_labelled_tokens = count_labelled_tokens(tags_knowledge)

            row_count += 1
            total_tokens += len(tokens)

            total_skill_b_tags += skill_b_tags
            total_skill_labelled_tokens += skill_labelled_tokens

            total_knowledge_b_tags += knowledge_b_tags
            total_knowledge_labelled_tokens += knowledge_labelled_tokens

    total_b_tags = total_skill_b_tags + total_knowledge_b_tags
    total_labelled_tokens = total_skill_labelled_tokens + total_knowledge_labelled_tokens

    return {
        "file": str(path),
        "exists": True,
        "rows": row_count,
        "avg_tokens": round(total_tokens / row_count, 2) if row_count else 0,

        "avg_skill_b_tags": round(total_skill_b_tags / row_count, 2) if row_count else 0,
        "avg_skill_labelled_tokens": round(total_skill_labelled_tokens / row_count, 2) if row_count else 0,

        "avg_knowledge_b_tags": round(total_knowledge_b_tags / row_count, 2) if row_count else 0,
        "avg_knowledge_labelled_tokens": round(total_knowledge_labelled_tokens / row_count, 2) if row_count else 0,

        "avg_total_b_tags": round(total_b_tags / row_count, 2) if row_count else 0,
        "avg_total_labelled_tokens": round(total_labelled_tokens / row_count, 2) if row_count else 0,
    }


def main():
    print("WEAK LABELLED FILE CHECK")
    print("========================\n")

    for file_path in FILES:
        result = count_jsonl_file(file_path)

        if not result["exists"]:
            print(f"{result['file']} - NOT FOUND")
            continue

        print(f"File: {result['file']}")
        print(f"Rows: {result['rows']}")
        print(f"Avg tokens: {result['avg_tokens']}")

        print(f"Avg skill B-tags: {result['avg_skill_b_tags']}")
        print(f"Avg skill labelled tokens: {result['avg_skill_labelled_tokens']}")

        print(f"Avg knowledge B-tags: {result['avg_knowledge_b_tags']}")
        print(f"Avg knowledge labelled tokens: {result['avg_knowledge_labelled_tokens']}")

        print(f"Avg total B-tags: {result['avg_total_b_tags']}")
        print(f"Avg total labelled tokens: {result['avg_total_labelled_tokens']}")
        print()


if __name__ == "__main__":
    main()