from dataprocessors.encoders import (
    IntentEncoder,
    EntityEncoder
)

# -----------------------
# Intent test
# -----------------------
intent_encoder = IntentEncoder(
    labels=[
        "transfer_money",
        "check_balance",
        "deposit_money"
    ]
)

encoded_intent = intent_encoder.encode(
    [
        "transfer_money",
        "check_balance"
    ]
)


print(
    "Intent:"
)

print(encoded_intent)

print(
    intent_encoder.decode(
        encoded_intent
    )
)

# -----------------------
# NER test
# -----------------------


entity_encoder = EntityEncoder(
    entities=[
        "receiver",
        "amount"
    ]
)

print(
    "\nNER mapping:"
)
print(
    entity_encoder.label_to_id
)
encoded_ner = entity_encoder.encode(
    [
        "B-receiver",
        "I-receiver",
        "O",
        "B-amount"
    ]
)
print(
    "\nNER:"
)

print(
    encoded_ner
)

print(
    entity_encoder.decode(
        encoded_ner
    )
)