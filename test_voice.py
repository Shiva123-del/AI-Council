import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

text = """
Welcome to the AI Council. The Researcher will begin the discussion,
followed by the Domain Expert, Critical Analyst, and Final Judge.
"""

speech_file_path = "test_voice.mp3"

with client.audio.speech.with_streaming_response.create(
    model="gpt-4o-mini-tts",
    voice="alloy",
    input=text,
) as response:
    response.stream_to_file(speech_file_path)

print("Voice generated successfully.")
print(f"Audio file saved as: {speech_file_path}")