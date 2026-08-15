import json
from typing import Any

import torch
from torch import nn

from peft import (
    LoraConfig,
    PeftModel,
    get_peft_model,
)
from transformers import AutoModelForCausalLM


class Model03LLMBasedClassificationModel(nn.Module):

    def __init__(
        self,
        model_name: str,
        tokenizer,
        adapter_path: str | None = None,
        device: str = "cpu",
    ):
        super().__init__()

        self.tokenizer = tokenizer
        self.device = device

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=(
                torch.float16
                if device == "cuda"
                else torch.float32
            ),
        )

        if adapter_path:

            # Load existing LoRA adapter
            self.model = PeftModel.from_pretrained(
                self.model,
                adapter_path,
            )

        else:

            # Create new LoRA adapter
            lora_config = LoraConfig(
                r=8,
                lora_alpha=16,
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=[
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                ],
            )

            self.model = get_peft_model(
                self.model,
                lora_config,
            )

        self.model.to(self.device)

    def forward(
        self,
        input_ids,
        attention_mask,
        labels=None,
    ):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

    def predict(self, text: str) -> dict:

        prompt = f"""Extract the intent and entities from this banking request.

            Input:
            {text}

            Return JSON only.

            Answer:
            """

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
        ).to(self.device)

        self.eval()

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
            )

        generated_tokens = outputs[0][
            inputs["input_ids"].shape[1]:
        ]

        generated_text = self.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True,
        )

        return self._parse_prediction(
            text,
            generated_text,
        )

    def _parse_prediction(
        self,
        text: str,
        generated_text: str,
    ):
        # Your existing parser
        ...

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