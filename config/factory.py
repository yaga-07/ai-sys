def create_llm_engine(provider: str):
    if provider == "openai":
        from llm_engines.openai_engine import OpenAIEngine
        return OpenAIEngine()
    elif provider == "anthropic":
        from llm_engines.anthropic_engine import AnthropicEngine
        return AnthropicEngine()
    elif provider == "local":
        from llm_engines.local_engine import LocalEngine
        return LocalEngine()
    raise ValueError(f"Unknown provider: {provider}")
