import json
import random
from pathlib import Path

import spacy
from spacy.training import Example

from src.config import (
    SPACY_TRAIN_DATA_PATH,
    SPACY_DEV_DATA_PATH,
    SPACY_MODEL_DIR,
    SPACY_TRAINING_SUMMARY_PATH,
)


def load_spacy_json(path: Path):
    # load the converted spaCy training data
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def filter_valid_examples(data, nlp):
    # turn raw json rows into spaCy Example objects
    examples = []

    for text, annotations in data:
        try:
            doc = nlp.make_doc(text)
            example = Example.from_dict(doc, annotations)
            examples.append(example)
        except Exception:
            # skip broken rows if any appear
            continue

    return examples


def count_entities(data):
    # count how many skill spans are in a dataset split
    total = 0
    for _, annotations in data:
        total += len(annotations.get("entities", []))
    return total


def save_training_summary(
    train_docs_count,
    dev_docs_count,
    train_entity_count,
    dev_entity_count,
    valid_train_examples,
    valid_dev_examples,
    losses_by_epoch,
):
    # save a small training summary for the report
    SPACY_TRAINING_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(SPACY_TRAINING_SUMMARY_PATH, "a", encoding="utf-8") as file:
        file.write("\n")
        file.write("SPACY MODEL TRAINING SUMMARY\n")
        file.write("============================\n\n")
        file.write(f"Train documents loaded: {train_docs_count}\n")
        file.write(f"Dev documents loaded: {dev_docs_count}\n")
        file.write(f"Train skill spans: {train_entity_count}\n")
        file.write(f"Dev skill spans: {dev_entity_count}\n")
        file.write(f"Valid training examples used: {valid_train_examples}\n")
        file.write(f"Valid dev examples used: {valid_dev_examples}\n\n")
        file.write("Loss by epoch\n")
        file.write("-------------\n")

        for epoch_number, loss_value in losses_by_epoch:
            file.write(f"Epoch {epoch_number}: {loss_value:.4f}\n")

        file.write("\nModel saved to:\n")
        file.write(f"{SPACY_MODEL_DIR}\n")


def run_train_spacy_skill_ner():
    print("Starting spaCy skill NER training...")

    # load prepared train/dev data
    train_data = load_spacy_json(SPACY_TRAIN_DATA_PATH)
    dev_data = load_spacy_json(SPACY_DEV_DATA_PATH)

    print(f"Loaded train rows: {len(train_data)}")
    print(f"Loaded dev rows: {len(dev_data)}")
    print(f"Train skill spans: {count_entities(train_data)}")
    print(f"Dev skill spans: {count_entities(dev_data)}")

    # start with a blank english pipeline
    nlp = spacy.blank("en")

    # add ner if it does not exist
    if "ner" not in nlp.pipe_names:
        ner = nlp.add_pipe("ner")
    else:
        ner = nlp.get_pipe("ner")

    # add the only label we need
    ner.add_label("SKILL")

    # convert rows into Example objects
    train_examples = filter_valid_examples(train_data, nlp)
    dev_examples = filter_valid_examples(dev_data, nlp)

    print(f"Valid training examples: {len(train_examples)}")
    print(f"Valid dev examples: {len(dev_examples)}")

    # simple training settings
    epochs = 10
    batch_size = 16
    losses_by_epoch = []

    # initialise training
    optimizer = nlp.initialize(lambda: train_examples)

    # train only the ner pipe
    pipe_exceptions = ["ner"]
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]

    with nlp.disable_pipes(*other_pipes):
        for epoch in range(1, epochs + 1):
            random.shuffle(train_examples)
            losses = {}

            batches = spacy.util.minibatch(train_examples, size=batch_size)

            for batch in batches:
                nlp.update(
                    batch,
                    drop=0.2,
                    losses=losses,
                    sgd=optimizer,
                )

            epoch_loss = losses.get("ner", 0.0)
            losses_by_epoch.append((epoch, epoch_loss))
            print(f"Epoch {epoch}/{epochs} | NER loss: {epoch_loss:.4f}")

    # save trained model
    SPACY_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    nlp.to_disk(SPACY_MODEL_DIR)

    # save summary
    save_training_summary(
        train_docs_count=len(train_data),
        dev_docs_count=len(dev_data),
        train_entity_count=count_entities(train_data),
        dev_entity_count=count_entities(dev_data),
        valid_train_examples=len(train_examples),
        valid_dev_examples=len(dev_examples),
        losses_by_epoch=losses_by_epoch,
    )

    print(f"Saved trained spaCy model to: {SPACY_MODEL_DIR}")
    print(f"Updated training summary: {SPACY_TRAINING_SUMMARY_PATH}")


if __name__ == "__main__":
    run_train_spacy_skill_ner()