from dataprocessors.encoders.classification import IntentClassificationEncoder
from dataprocessors.encoders.bio import LabelBIOEncoder

intent_encoder = IntentClassificationEncoder.from_file("./metadata/intents.json")
entity_encoder = LabelBIOEncoder.from_file("./metadata/entities.json")

print(intent_encoder.id_to_label)
print(entity_encoder.id_to_label)