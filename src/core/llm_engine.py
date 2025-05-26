class BaseLLMEngine:
    def generate(self, messages: list) -> str:
        raise NotImplementedError
    
    def generate_from_pdf(self, pdf_bytes: list, messages: list) -> str:
        raise NotImplementedError

    def generate_from_image(self, image_bytes: list, messages: list) -> str:
        raise NotImplementedError
    
    def generate_from_audio(self, audio_bytes: list, messages: list) -> str:
        raise NotImplementedError
    
    def generate_from_video(self, video_bytes: list, messages: list) -> str:
        raise NotImplementedError