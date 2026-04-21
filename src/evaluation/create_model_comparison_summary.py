from pathlib import Path

from src.config import (
    SPACY_EVALUATION_RESULTS_PATH,
)


def run_create_model_comparison_summary():
    base_dir = Path(__file__).resolve().parent.parent.parent
    output_dir = base_dir / "data" / "processed" / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "model_comparison_summary.txt"

    with open(output_path, "w", encoding="utf-8") as file:
        file.write("MODEL COMPARISON SUMMARY\n")
        file.write("========================\n\n")

        file.write("1. BASELINE V3\n")
        file.write("--------------\n")
        file.write("Type: ESCO-based keyword matching baseline\n")
        file.write("Strengths:\n")
        file.write("- very transparent\n")
        file.write("- easy to explain and reproduce\n")
        file.write("- useful for standardised taxonomy matching\n")
        file.write("Weaknesses:\n")
        file.write("- weak contextual understanding\n")
        file.write("- can miss real skills if wording changes\n")
        file.write("- earlier versions showed noisy behavioural phrases\n")
        file.write("Project verdict:\n")
        file.write("- good transparent baseline\n")
        file.write("- not strong enough to act as the main learned extractor\n\n")

        file.write("2. SPACY NER\n")
        file.write("------------\n")
        file.write("Type: supervised NER model trained on SkillSpan\n")
        file.write("Test results:\n")
        file.write("- test rows: 3569\n")
        file.write("- gold skill spans: 1090\n")
        file.write("- predicted skill spans: 646\n")
        file.write("- exact span matches: 252\n")
        file.write("- row precision: 0.7043\n")
        file.write("- row recall: 0.5883\n")
        file.write("- row F1: 0.6411\n")
        file.write("- exact span precision: 0.3901\n")
        file.write("- exact span recall: 0.2312\n")
        file.write("- exact span F1: 0.2903\n")
        file.write("Strengths:\n")
        file.write("- learns beyond fixed keywords\n")
        file.write("- clearly stronger than simple dictionary matching\n")
        file.write("- produced some exact matches on unseen test rows\n")
        file.write("Weaknesses:\n")
        file.write("- misses many gold spans\n")
        file.write("- boundary matching is still weak\n")
        file.write("- fragment-based dataset likely limits performance\n")
        file.write("Project verdict:\n")
        file.write("- better extractor than the keyword baseline\n")
        file.write("- useful learned comparison model\n")
        file.write("- not perfect, especially at exact span boundaries\n\n")

        file.write("3. DISTILBERT\n")
        file.write("-------------\n")
        file.write("Type: transformer-based token classification model\n")
        file.write("Status:\n")
        file.write("- add DistilBERT metrics here once confirmed from its evaluation output\n")
        file.write("Expected strengths:\n")
        file.write("- strongest contextual modelling\n")
        file.write("- better handling of varied wording than baseline methods\n")
        file.write("Expected weaknesses:\n")
        file.write("- more complex pipeline\n")
        file.write("- less transparent than keyword matching\n")
        file.write("Project verdict:\n")
        file.write("- likely strongest NLP model if evaluation supports it\n")
        file.write("- needs confirmed metrics before final comparison claim\n\n")

        file.write("CURRENT OVERALL CONCLUSION\n")
        file.write("--------------------------\n")
        file.write("Baseline v3 is the most transparent method and works as a defensible reference system.\n")
        file.write("spaCy performs as a real learned extractor and is stronger than the baseline, but still has recall and span-boundary weaknesses.\n")
        file.write("DistilBERT should be treated as the advanced comparison model once its final metrics are confirmed.\n")

    print(f"Saved model comparison summary to: {output_path}")


if __name__ == "__main__":
    run_create_model_comparison_summary()