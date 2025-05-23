from core.llm_engine import BaseLLMEngine

class AnthropicEngine(BaseLLMEngine):
    def generate(self, prompt):
        # Dummy response
        return "Anthropic response to: " + prompt
