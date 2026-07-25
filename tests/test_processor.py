
from dataprocessors.preprocessors import DataPreProcessor


processor = DataPreProcessor()

samples = processor.process_file("./datasets/testing.json")

for sample in samples:
    print("====================")
    print(f"TEXT:{sample.text}")

    print(
        "\nINTENTS:"
    )
    for intent in sample.intents:
        print(f"- {intent.label}")
        print(
            "  intent text:",
            sample.text[
                intent.start_index:
                intent.end_index
            ]
        )
        print(
            "  ENTITIES:"
        )
        for entity in intent.entities:
            print(
                "   ",
                entity.label,
                "=>",
                sample.text[
                    entity.start_index:
                    entity.end_index
                ]
            )