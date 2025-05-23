from core.llm_engine import BaseLLMEngine

class OpenAIEngine(BaseLLMEngine):
    def generate(self, prompt):
        # Dummy response
        return "OpenAI response to: " + prompt
