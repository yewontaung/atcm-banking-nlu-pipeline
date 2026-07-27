from utils.schemas import EntityPrediction, EntityTokenPrediction


class EntityCombiner:

    def combine(self, text: str, predictions: list[EntityTokenPrediction]) -> list[EntityPrediction]:

        entities: list[EntityPrediction] = []
        current: EntityPrediction | None = None

        for prediction in predictions:
            label = prediction.label
            # Ignore outside tokens
            if label == "O":
                if current is not None:
                    current.value = text[
                        current.start_index:
                        current.end_index
                    ]
                    entities.append(current)
                    current = None
                continue

            # Beginning of new entity
            if label.startswith("B-"):

                if current is not None:
                    current.value = text[
                        current.start_index:
                        current.end_index
                    ]
                    entities.append(current)

                current = EntityPrediction(
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

                    current = EntityPrediction(
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
                    entities.append(current)
                    current = EntityPrediction(
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

            entities.append(current)


        return entities
