from banking_nlu.utils.schemas import ExportedDataset


class DataValidator:

    def validate(self, item: ExportedDataset) -> None:
        self.validate_text(item.text)

        for intent in item.intents:
            self.validate_span(
                item.text,
                intent.start_index,
                intent.end_index,
                "intent"
            )

            for entity in intent.entities:
                self.validate_span(
                    item.text,
                    entity.start_index,
                    entity.end_index,
                    "entity"
                )

    def validate_text(self, text: str):
        if not text.strip():
            raise ValueError(
                "Text cannot be empty"
            )

    def validate_span(
        self,
        text: str,
        start: int,
        end: int,
        span_type: str
    ):
        if start < 0:
            raise ValueError(
                f"{span_type} start index invalid"
            )
        if end > len(text):
            raise ValueError(
                f"{span_type} end index out of range"
            )
        if start > end:
            raise ValueError(
                f"{span_type} start > end"
            )