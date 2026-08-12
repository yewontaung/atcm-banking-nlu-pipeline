import torch
from tqdm import tqdm


class NLUModelTrainer:

    def __init__(
        self,
        model,
        optimizer,
        loss_fn,
        device
    ):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device


    def train_epoch(
        self,
        dataloader
    ):

        self.model.train()

        total_loss = 0


        progress = tqdm(dataloader)


        for batch in progress:
            input_ids = batch["input_ids"].to(
                self.device
            )
            attention_mask = batch["attention_mask"].to(
                self.device
            )

            intent_labels = batch["intent_labels"].to(
                self.device
            )

            ner_labels = batch["ner_labels"].to(
                self.device
            )


            self.optimizer.zero_grad()


            output = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            losses = self.loss_fn(
                output,
                intent_labels,
                ner_labels
            )
            loss = losses["loss"]
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()


            progress.set_description(
                f"train_loss={loss.item():.4f}"
            )


        return total_loss / len(dataloader)

    def validate_epoch(self, dataloader):
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            progress = tqdm(dataloader)

            for batch in progress:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(
                    self.device
                )
                intent_labels = batch["intent_labels"].to(self.device)
                ner_labels = batch["ner_labels"].to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                losses = self.loss_fn(outputs, intent_labels, ner_labels)

                loss = losses["loss"]

                total_loss += loss.item()

                progress.set_description(
                    f"val_loss={loss.item():.4f}"
                )
            return total_loss / len(dataloader)
