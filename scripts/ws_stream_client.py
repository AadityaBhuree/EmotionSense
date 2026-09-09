#!/usr/bin/env python3
"""CLI utility to test real-time WebSocket streaming telemetry with EmotionSense API.

Usage:
    python scripts/ws_stream_client.py --endpoint affect --text "I am feeling thrilled!"
    python scripts/ws_stream_client.py --endpoint speech --text "Hello world! This is a test."
    python scripts/ws_stream_client.py --endpoint speech --simulate-audio --pitch 220
"""

import argparse
import asyncio
import sys
import numpy as np
from pathlib import Path

# Add repo root to path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from src.utils.ws_client import EmotionSenseWSClient


async def main():
    parser = argparse.ArgumentParser(description="EmotionSense WebSocket Streaming Client")
    parser.add_argument("--host", default="127.0.0.1", help="API Host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="API Port (default: 8000)")
    parser.add_argument("--endpoint", choices=["affect", "speech"], default="affect", help="Target streaming endpoint")
    parser.add_argument("--text", type=str, default="This is an exciting and wonderful achievement!", help="Text message")
    parser.add_argument("--simulate-audio", action="store_true", help="Generate synthetic sine wave audio chunk")
    parser.add_argument("--pitch", type=float, default=220.0, help="Synthetic tone pitch in Hz (default: 220.0)")

    args = parser.parse_args()
    client = EmotionSenseWSClient(host=args.host, port=args.port)

    print(f"Connecting to EmotionSense WebSocket at {client.base_url} ...")

    if args.endpoint == "affect":
        print(f"Streaming text to /ws/stream-affect: \"{args.text}\"")
        try:
            resp = await client.stream_affect_message(args.text)
            print("\n[RESULT /ws/stream-affect]")
            print(f"  Dominant Emotion : {resp.get('dominant_emotion')}")
            print(f"  Confidence       : {resp.get('confidence'):.2%}")
            print(f"  Valence          : {resp.get('valence'):+.2f}")
            print(f"  Arousal          : {resp.get('arousal'):+.2f}")
            print(f"  Empathy Advice   : {resp.get('empathy_advice')}")
            print(f"  Round-Trip       : {resp.get('round_trip_ms')} ms")
        except Exception as e:
            print(f"Connection failed: {e}. (Is 'python server.py' running?)")

    elif args.endpoint == "speech":
        if args.simulate_audio:
            print(f"Generating 0.5s synthetic {args.pitch}Hz tone PCM buffer...")
            t = np.linspace(0, 0.5, 8000, endpoint=False)
            sine_pcm = (np.sin(2 * np.pi * args.pitch * t) * 32767).astype(np.int16).tobytes()
            print("Streaming audio bytes to /ws/stream-speech ...")
            try:
                resp = await client.stream_speech_audio_chunk(sine_pcm, sample_rate=16000)
                print("\n[RESULT /ws/stream-speech (Audio)]")
                print(f"  Transcript       : \"{resp.get('transcript')}\"")
                print(f"  Tokens Detected  : {len(resp.get('tokens', []))}")
                print(f"  Round-Trip       : {resp.get('round_trip_ms')} ms")
            except Exception as e:
                print(f"Connection failed: {e}. (Is 'python server.py' running?)")
        else:
            print(f"Streaming text to /ws/stream-speech: \"{args.text}\"")
            try:
                resp = await client.stream_speech_text(args.text)
                print("\n[RESULT /ws/stream-speech]")
                print(f"  Transcript       : \"{resp.get('transcript')}\"")
                print(f"  Dominant Emotion : {resp.get('dominant_emotion')}")
                print(f"  Confidence       : {resp.get('confidence'):.2%}")
                print(f"  Tokens           : {len(resp.get('tokens', []))}")
                print(f"  Round-Trip       : {resp.get('round_trip_ms')} ms")
            except Exception as e:
                print(f"Connection failed: {e}. (Is 'python server.py' running?)")


if __name__ == "__main__":
    asyncio.run(main())
