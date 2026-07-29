from utils.schemas import SpanPrediction, TokenPrediction


import torch

from dataprocessors.encoders.bio import LabelBIOEncoder
from utils.schemas import TokenPrediction



class BIOLogitMapper:

    def __init__(self, encoder: LabelBIOEncoder):
        self.encoder = encoder

    def decode(self, logits:torch.Tensor, text:str, offset_mapping: torch.Tensor) -> list[str]:
        result = []
        token_ids = torch.argmax(
                    logits,
                    dim=-1
                )[0]

        for index, (token_id, offset) in enumerate(
            zip(token_ids, offset_mapping)
        ):
            label_id = int(token_id)
            label = self.encoder.id_to_label[label_id]
            result.append(label)

        return result
        
        

    def map(
        self,
        logits: torch.Tensor,
        text: str,
        offset_mapping: torch.Tensor
    ) -> list[TokenPrediction]:

        """
        Convert token classification logits into BIO predictions

        logits:
            (batch, seq_len, label_count)

        offset_mapping:
            (seq_len, 2)
        """


        predictions = []


        token_ids = torch.argmax(
            logits,
            dim=-1
        )[0]


        probabilities = torch.softmax(
            logits,
            dim=-1
        )[0]


        for index, (token_id, offset) in enumerate(
            zip(token_ids, offset_mapping)
        ):

            label_id = int(token_id)

            label = self.encoder.id_to_label[
                label_id
            ]


            start_index = int(offset[0])
            end_index = int(offset[1])


            # skip <s>, </s>, padding
            if start_index == end_index:
                continue


            # ignore outside
            if label == "O":
                continue


            predictions.append(
                TokenPrediction(
                    prediction_id=label_id,
                    label=label,
                    value=text[
                        start_index:end_index
                    ],
                    confidence=float(
                        probabilities[index][label_id]
                    ),
                    start_index=start_index,
                    end_index=end_index
                )
            )


        return predictions

class BIOCombiner:

    def combine(self, text: str, predictions: list[TokenPrediction]) -> list[SpanPrediction]:

        items: list[SpanPrediction] = []
        current: SpanPrediction | None = None

        for prediction in predictions:
            label = prediction.label
            # Ignore outside tokens
            if label == "O":
                if current is not None:
                    current.value = text[
                        current.start_index:
                        current.end_index
                    ]
                    items.append(current)
                    current = None
                continue

            # Beginning of new entity
            if label.startswith("B-"):

                if current is not None:
                    current.value = text[
                        current.start_index:
                        current.end_index
                    ]
                    items.append(current)

                current = SpanPrediction(
                    prediction_id=prediction.prediction_id,
                    label=label[2:],
                    confidence=prediction.confidence,
                    start_index=prediction.start_index,
                    end_index=prediction.end_index,
                    value=text[
                        prediction.start_index:
                        prediction.end_index
                    ]
                )
                continue

            # Inside entity
            if label.startswith("I-"):

                entity_label = label[2:]

                # No previous entity
                if current is None:

                    current = SpanPrediction(
                        prediction_id=prediction.prediction_id,
                        label=entity_label,
                        confidence=prediction.confidence,
                        start_index=prediction.start_index,
                        end_index=prediction.end_index,
                        value=text[
                            prediction.start_index:
                            prediction.end_index
                        ]
                    )

                    continue

                # Different entity type
                if current.label != entity_label:
                    current.value = text[
                        current.start_index:
                        current.end_index
                    ]
                    items.append(current)
                    current = SpanPrediction(
                        prediction_id=prediction.prediction_id,
                        label=entity_label,
                        confidence=prediction.confidence,
                        start_index=prediction.start_index,
                        end_index=prediction.end_index,
                        value=text[
                            prediction.start_index:
                            prediction.end_index
                        ]
                    )
                    continue


                # Same entity
                current.end_index = max(
                    current.end_index,
                    prediction.end_index
                )

                current.value = text[
                    current.start_index:
                    current.end_index
                ]

                current.confidence = max(
                    current.confidence,
                    prediction.confidence
                )


        # Flush last entity
        if current is not None:

            current.value = text[
                current.start_index:
                current.end_index
            ]

            items.append(current)


        return items
