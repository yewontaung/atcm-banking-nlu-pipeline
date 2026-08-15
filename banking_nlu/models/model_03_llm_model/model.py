import json
from typing import Any

from peft import PeftModel
import torch

from transformers import AutoModelForCausalLM


class Model03LLMBasedClassificationModel:

    def __init__(
        self,
        model_name:str,
        tokenizer,
        adapter_path:str | None,
        device:str = "cpu"):

        self.tokenizer = tokenizer
        self.device = device
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=(
                torch.float16 if self.device == "cuda" else torch.float32
            ),
        )

        if adaptor_path:
            self.model = PeftModel.from_pretrained(
                self.model,
                adaptor_path
            )

        self.model.to(self.device)
        self.model.eval()


    def predict(self, text:str) -> dict:
        prompt = f"""Extract the intent and entities from this banking request.
            Input:
            {text}
            Return JSON only.
            Answer:
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs =  self.model.generate(
                **inputs,
                max_new_tokens = 128,
                do_sample = False,
            )

        generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        generated_text = self.tokenizer.decode(
            generated_tokens,
            skip_special_tokens = True
        )
        
        return self._parse_prediction(text, generated_text)

    def _parse_prediction(self, text:str, generated_text:str) -> dict[str, Any]:
        try:
            result = json.loads(generated_text)
            predictions:list[dict[str, Any]] = []
            for intent, entity_data in result.items():
                entities = [
                    {"label": i["label"], "value": i["value"]}
                    for i in entity_data
                ]
                predictions.append({
                    "label":intent,
                    "entites": entities
                })

            return {
                "text": text,
                "intents": predictions
            }
        except json.JSONDecodeError:
            return {
                "text": text,
                "intents": [],
                "raw_output": generated_text,
            }