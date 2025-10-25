#!/usr/bin/env python3
"""Hello Subscriber using Portal communication framework."""

import os
import time
from typing import Optional

import portal


class HelloSubscriber:
    """A simple hello world subscriber using Portal."""

    def __init__(
            self,
            server_host: Optional[str] = None,
            server_port: int = 2222):
        """Initialize the subscriber.

        Args:
            server_host: Host address of the Portal server (defaults to env var or localhost)
            server_port: Port number of the Portal server
        """
        # Use environment variable if available, otherwise use provided host or
        # localhost
        self.server_host = server_host or os.getenv(
            'PUBLISHER_HOST', 'localhost')
        self.server_port = server_port
        self.client = portal.Client(f"{self.server_host}:{self.server_port}")
        self.message_count = 0  # Track number of messages received

    def connect(self) -> bool:
        """Connect to the publisher server."""
        try:
            print(
                "🔗 Connecting to publisher at "
                f"{self.server_host}:{self.server_port}..."
            )
            self.client.connect()
            print("✅ Connected to publisher!")
            return True
        except Exception as e:
            print(f"❌ Failed to connect: {e}")
            return False

    def run(self) -> None:
        """Start the subscriber client."""
        print("🚀 Hello Subscriber starting...")
        print("⏹️  Press Ctrl+C to stop the subscriber")

        # Keep trying to connect and receive messages indefinitely
        while True:
            try:
                if not self.connect():
                    print("💔 Could not connect to publisher. Retrying in 5 seconds...")
                    time.sleep(5)
                    continue

                print("👂 Listening for messages...")

                # Message receiving loop
                while True:
                    try:
                        # Get a new message from the publisher
                        future = self.client.get_message()
                        message = future.result()

                        print(f"📨 Received: {message}")
                        self.message_count += 1  # Increment message counter

                        # Wait before requesting next message
                        time.sleep(1)

                    except Exception as e:
                        print(f"⚠️  Error receiving message: {e}")
                        print("🔄 Connection lost, attempting to reconnect...")
                        break  # Break inner loop to reconnect

            except KeyboardInterrupt:
                print("\n⏹️  Subscriber shutting down...")
                break
            except Exception as e:
                print(f"❌ Subscriber error: {e}")
                print("🔄 Retrying in 5 seconds...")
                time.sleep(5)

        try:
            # Get final status if possible
            status = self.client.call('get_status')
            print(f"📊 Final publisher status: {status}")
        except BaseException:
            pass

        print(f"📈 Total messages received: {self.message_count}")
        print("👋 Subscriber stopped")


def main() -> None:
    """Main function to run the subscriber."""
    subscriber = HelloSubscriber()
    subscriber.run()


if __name__ == "__main__":
    main()
