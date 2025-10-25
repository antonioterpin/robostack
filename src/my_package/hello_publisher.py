#!/usr/bin/env python3
"""Hello World Publisher.

This module implements a simple publisher node that repeatedly
publishes "Hello World" messages.
"""

import time
from typing import Any, Dict

import portal


class HelloPublisher:
    """A simple hello world publisher using Portal."""

    def __init__(self, port: int = 2222, message: str = "Hello World"):
        """Initialize the publisher.

        Args:
            port: Port number for the Portal server
            message: Message to publish
        """
        self.port = port
        self.message = message
        self.server = portal.Server(port)
        self.counter = 0

        # Bind methods that clients can call
        self.server.bind('get_message', self._get_message)
        self.server.bind('get_status', self._get_status)

    def _get_message(self) -> str:
        """Get the current message with counter."""
        self.counter += 1
        return f"{self.message} #{self.counter}"

    def _get_status(self) -> Dict[str, Any]:
        """Get publisher status."""
        return {
            "active": True,
            "port": self.port,
            "message_count": self.counter,
            "uptime": time.time()
        }

    def run(self) -> None:
        """Start the publisher server."""
        print(f"🚀 Hello Publisher starting on port {self.port}")
        print(f"📢 Publishing message: '{self.message}'")

        try:
            self.server.start()
            print("✅ Publisher server started successfully!")
            print("📡 Waiting for clients to connect...")
            print("⏹️  Press Ctrl+C to stop the publisher")

            # Keep the server running indefinitely
            while True:
                time.sleep(1)

        except KeyboardInterrupt:
            print("\n⏹️  Publisher shutting down...")
        except Exception as e:
            print(f"❌ Publisher error: {e}")
            # On error, keep trying to restart
            print("🔄 Attempting to restart in 5 seconds...")
            time.sleep(5)
            self.run()  # Restart on error
        finally:
            print("👋 Publisher stopped")


def main() -> None:
    """Main function to run the publisher."""
    publisher = HelloPublisher()
    publisher.run()


if __name__ == "__main__":
    main()
