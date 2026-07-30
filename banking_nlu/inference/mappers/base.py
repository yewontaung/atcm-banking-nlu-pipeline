from abc import ABC, abstractmethod
from typing import TypeVar

from torch import Tensor

from banking_nlu.utils.schemas import ModelOutput

T = TypeVar("T", bound=ModelOutput)

class BasePredictionMapper(ABC):

    @abstractmethod
    def map(
        self,
        text:str,
        outputs:T,
        offset_mapping:Tensor
    ):
        pass