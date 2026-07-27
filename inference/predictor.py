import torch


class Predictor:

    def __init__(
        self,
        model,
        tokenizer,
        mapper,
        device
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.mapper = mapper
        self.device = device


    def predict(self, text:str):

        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            return_offsets_mapping=True,
            padding=True,
            truncation=True
        )


        input_ids = encoded["input_ids"].to(
            self.device
        )

        attention_mask = encoded["attention_mask"].to(
            self.device
        )


        self.model.eval()

        with torch.no_grad():

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )


        return self.mapper.map(
            text=text,
            outputs=outputs,
            offset_mapping=encoded["offset_mapping"][0]
        )