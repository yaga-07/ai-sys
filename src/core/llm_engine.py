class BaseLLMEngine:
    def generate(self, prompt: str) -> str:
        raise NotImplementedError
