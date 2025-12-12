#!/usr/bin/env python3
"""
Demo Preparation Script
Quickly warms up models before live demo
"""

import asyncio
import sys
from warmup_models import ModelWarmer
from loguru import logger

async def prepare_demo():
    """Prepare system for demo"""
    print("🎬 PREPARING VOICE AI DEMO")
    print("=" * 40)
    print("This will warm up all AI models...")
    print("Expected time: 10-15 seconds")
    print("=" * 40)

    warmer = ModelWarmer()
    success = await warmer.warmup_all()

    print("\n" + "=" * 40)
    if success:
        print("🎉 DEMO READY!")
        print("✅ All models warmed up")
        print("✅ First user request will be FAST")
        print("🚀 You can now start your demo!")
    else:
        print("⚠️ DEMO PREPARATION INCOMPLETE")
        print("❌ Some models failed to warm up")
        print("💡 First request may still be slow")
    print("=" * 40)

    return success

if __name__ == "__main__":
    try:
        success = asyncio.run(prepare_demo())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n❌ Demo preparation cancelled")
        sys.exit(1)