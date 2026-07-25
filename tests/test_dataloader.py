from torch.utils.data import DataLoader

from dataloader.dataset import NLUDataset
from dataloader.collator import NLUCollator

from utils.schemas import TokenizedDataset



samples = [

    TokenizedDataset(

        text="hello",

        input_ids=[
            0,10,11,2
        ],

        attention_mask=[
            1,1,1,1
        ],

        intent_labels=[
            1,0,0
        ],

        ner_labels=[
            -100,1,0,-100
        ]

    ),


    TokenizedDataset(

        text="hello world",

        input_ids=[
            0,10,11,12,2
        ],

        attention_mask=[
            1,1,1,1,1
        ],

        intent_labels=[
            0,1,0
        ],

        ner_labels=[
            -100,1,2,0,-100
        ]

    )
]


dataset = NLUDataset(samples)


collator = NLUCollator(
    pad_token_id=1
)


loader = DataLoader(

    dataset,

    batch_size=2,

    shuffle=True,

    collate_fn=collator

)



batch = next(iter(loader))


for key,value in batch.items():

    print(key)
    print(value)
    print()