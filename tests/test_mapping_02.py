from dataprocessors.encoders import IntentEncoder, EntityEncoder

intent_encoder = IntentEncoder.from_file("./metadata/intents.json")
entity_encoder = EntityEncoder.from_file("./metadata/entities.json")

print(intent_encoder.id_to_label)
print(entity_encoder.id_to_label)