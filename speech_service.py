# speech_service.py
from speech_to_chat import AudioTranscriber

class SpeechService:
    def __init__(self, bot):
        self.bot = bot
        self.transcriber = AudioTranscriber(self.bot)
        self.is_running = False

    async def record_and_transcribe(self):
        """Record and transcribe audio."""
        self.is_running = True
        await self.transcriber.record_and_transcribe()