from abc import ABC, abstractmethod

class BaseChain(ABC):
    def __init__(self) -> None:
        super().__init__()
        self.chain_name: str = None

    def parser(self, chain_text: str):
        pass