from core.llm_engine import BaseLLMEngine

class LocalEngine(BaseLLMEngine):
    def generate(self, prompt):
        # Dummy response
        return "Local LLM response to: " + prompt
