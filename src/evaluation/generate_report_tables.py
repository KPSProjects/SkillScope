from pathlib import Path
import csv


def save_csv(output_path, headers, rows):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerows(rows)


def make_markdown_table(headers, rows):
    header_line = "| " + " | ".join(headers) + " |"
    separator_line = "| " + " | ".join(["---"] * len(headers)) + " |"

    row_lines = []
    for row in rows:
        row_lines.append("| " + " | ".join(str(cell) for cell in row) + " |")

    return "\n".join([header_line, separator_line] + row_lines)


def main():
    output_dir = Path("data/processed/evaluation/report_tables")
    output_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------
    # TABLE 3.4 - Reed UK Large data quality summary
    # -------------------------------------------------
    table_34_headers = ["Metric", "Value"]
    table_34_rows = [
        ["Raw Reed UK Large rows", "50,000"],
        ["Rows after cleaning", "42,793"],
        ["Full duplicates removed", "5,179"],
        ["Additional text-based duplicates removed", "2,028"],
        ["Remaining rows with usable advert text", "42,793"],
    ]

    save_csv(output_dir / "table_3_4_reed_data_quality_summary.csv", table_34_headers, table_34_rows)

    table_34_markdown = make_markdown_table(table_34_headers, table_34_rows)

    # -------------------------------------------------
    # TABLE 6.1 - Evaluation summary table
    # -------------------------------------------------
    table_61_headers = ["Method", "Evaluation setup", "Precision", "Recall", "F1", "Notes"]
    table_61_rows = [
        ["Baseline keyword matcher", "SkillSpan token-level", "0.4617", "0.1334", "0.2070", "Transparent baseline, very low recall"],
        ["spaCy NER", "Row-level binary", "0.7043", "0.5883", "0.6411", "Strong row-level detection"],
        ["spaCy NER", "Exact span-level", "0.3901", "0.2312", "0.2903", "Stricter span matching"],
        ["DistilBERT", "Initial token classification, 3 epochs", "0.1392", "0.1494", "0.1441", "Weak early setup"],
        ["DistilBERT", "Row-level token classification, 3 epochs", "0.3981", "0.4189", "0.4082", "Large improvement after row-level redesign"],
        ["DistilBERT", "Row-level token classification, 5 epochs", "0.4150", "0.4253", "0.4201", "Best DistilBERT model"],
    ]

    save_csv(output_dir / "table_6_1_evaluation_summary.csv", table_61_headers, table_61_rows)

    table_61_markdown = make_markdown_table(table_61_headers, table_61_rows)

    # -------------------------------------------------
    # TABLE 6.2 - Manual artefact spot-check validation
    # -------------------------------------------------
    table_62_headers = ["Check area", "Observation"]
    table_62_rows = [
        ["Matched skills", "Most matched skills were genuinely present in both the CV and the job advert"],
        ["Missing skills", "Missing skills were generally plausible and reflected clear advert requirements"],
        ["False positives", "Earlier versions produced some broad or inflated matches, reduced later through controlled dictionary refinement and advert filtering"],
        ["Score realism", "Final match scores were more believable after requirement-aware parsing and evidence-based weighting"],
        ["Overall judgement", "The refined artefact was considered practically plausible for demonstration purposes"],
    ]

    save_csv(output_dir / "table_6_2_manual_artefact_spot_check.csv", table_62_headers, table_62_rows)

    table_62_markdown = make_markdown_table(table_62_headers, table_62_rows)

    # -------------------------------------------------
    # SAVE ONE REPORT-READY TXT FILE
    # -------------------------------------------------
    report_tables_path = output_dir / "report_ready_tables.txt"

    with open(report_tables_path, "w", encoding="utf-8") as file:
        file.write("TABLE 3.4 - REED UK LARGE DATA QUALITY SUMMARY\n")
        file.write("=============================================\n\n")
        file.write(table_34_markdown)
        file.write("\n\n")

        file.write("TABLE 6.1 - EVALUATION SUMMARY TABLE\n")
        file.write("====================================\n\n")
        file.write(table_61_markdown)
        file.write("\n\n")

        file.write("TABLE 6.2 - MANUAL ARTEFACT SPOT-CHECK VALIDATION\n")
        file.write("=================================================\n\n")
        file.write(table_62_markdown)
        file.write("\n")

    print("Saved report-ready tables to:")
    print(report_tables_path)
    print()
    print("Saved CSV copies to:")
    print(output_dir / "table_3_4_reed_data_quality_summary.csv")
    print(output_dir / "table_6_1_evaluation_summary.csv")
    print(output_dir / "table_6_2_manual_artefact_spot_check.csv")


if __name__ == "__main__":
    main()