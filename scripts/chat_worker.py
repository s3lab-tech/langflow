"""Google Chat Worker - Listens to Pub/Sub and triggers Langflow flows.

This worker stays running and processes incoming Google Chat messages.

Usage:
    python chat_worker.py --config config.json

    Or set environment variables:
    - GOOGLE_APPLICATION_CREDENTIALS: Path to service account JSON
    - GCP_PROJECT_ID: Google Cloud Project ID
    - PUBSUB_SUBSCRIPTION: Pub/Sub subscription name
    - LANGFLOW_API_URL: Langflow API URL (e.g., http://localhost:7860)
    - LANGFLOW_FLOW_ID: Flow ID to trigger
"""

import argparse
import json
import os
import signal
import sys
import time
from concurrent.futures import TimeoutError as FuturesTimeoutError
from pathlib import Path

import requests
from google.cloud import pubsub_v1
from google.oauth2 import service_account
from loguru import logger

# Configure logging
logger.remove()
logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level} | {message}")


class ChatWorker:
    """Worker that processes Google Chat messages from Pub/Sub."""

    def __init__(
        self,
        project_id: str,
        subscription_name: str,
        langflow_api_url: str,
        langflow_flow_id: str,
        service_account_json: str | None = None,
        bot_email: str | None = None,
    ):
        self.project_id = project_id
        self.subscription_name = subscription_name
        self.langflow_api_url = langflow_api_url.rstrip("/")
        self.langflow_flow_id = langflow_flow_id
        self.bot_email = bot_email
        self.running = True

        # Create Pub/Sub client
        if service_account_json:
            with Path(service_account_json).open() as f:
                sa_info = json.load(f)
            credentials = service_account.Credentials.from_service_account_info(sa_info)
            self.subscriber = pubsub_v1.SubscriberClient(credentials=credentials)
            if not bot_email:
                self.bot_email = sa_info.get("client_email", "")
        else:
            # Use default credentials
            self.subscriber = pubsub_v1.SubscriberClient()

        self.subscription_path = self.subscriber.subscription_path(project_id, subscription_name)

        logger.info("Worker initialized")
        logger.info(f"  Project: {project_id}")
        logger.info(f"  Subscription: {subscription_name}")
        logger.info(f"  Langflow URL: {langflow_api_url}")
        logger.info(f"  Flow ID: {langflow_flow_id}")
        logger.info(f"  Bot email filter: {self.bot_email or 'None'}")

    def _parse_chat_message(self, data: bytes) -> dict | None:
        """Parse Google Chat message from Pub/Sub data."""
        try:
            message_data = json.loads(data.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            logger.warning(f"Failed to parse message: {e}")
            return None

        event_type = message_data.get("type", "")

        # Only process MESSAGE events
        if event_type != "MESSAGE":
            logger.debug(f"Skipping event type: {event_type}")
            return None

        message = message_data.get("message", {})
        sender = message.get("sender", {})
        thread = message.get("thread", {})
        space = message_data.get("space", {})

        return {
            "text": message.get("text", ""),
            "message_name": message.get("name", ""),
            "sender_name": sender.get("displayName", "Unknown"),
            "sender_email": sender.get("email", ""),
            "sender_type": sender.get("type", ""),
            "thread_name": thread.get("name", ""),
            "thread_key": thread.get("threadKey", ""),
            "space_name": space.get("name", ""),
            "create_time": message.get("createTime", ""),
        }

    def _should_process(self, parsed: dict) -> bool:
        """Check if message should be processed."""
        # Skip bot messages
        if self.bot_email:
            sender_email = parsed.get("sender_email", "")
            if self.bot_email in sender_email:
                logger.debug(f"Skipping bot message from: {sender_email}")
                return False

        # Skip empty messages
        if not parsed.get("text", "").strip():
            logger.debug("Skipping empty message")
            return False

        return True

    def _call_langflow(self, message: dict) -> str | None:
        """Call Langflow API with the message."""
        url = f"{self.langflow_api_url}/api/v1/run/{self.langflow_flow_id}"

        payload = {
            "input_value": message["text"],
            "output_type": "chat",
            "input_type": "chat",
            "tweaks": {
                # Pass message metadata as tweaks if needed
                "thread_name": message.get("thread_name", ""),
                "sender_name": message.get("sender_name", ""),
                "space_name": message.get("space_name", ""),
            },
        }

        try:
            logger.info(f"Calling Langflow: {url}")
            response = requests.post(url, json=payload, timeout=120)
            response.raise_for_status()

            result = response.json()
            # Extract output from Langflow response
            outputs = result.get("outputs", [])
            if outputs:
                output = outputs[0].get("outputs", [])
                if output:
                    response_text = output[0].get("results", {}).get("message", {}).get("text", "")
                    if response_text:
                        return response_text

        except requests.exceptions.RequestException as e:
            logger.error(f"Langflow API error: {e}")

        return None

    def _send_reply(self, message: dict, reply_text: str):
        """Send reply back to Google Chat.

        Note: This requires the GoogleChatSender component or direct API call.
        For now, we log the reply. Implement based on your needs.
        """
        logger.info(f"Reply to {message.get('sender_name')}: {reply_text[:100]}...")
        # TODO: Implement sending reply via Google Chat API
        # You could call another Langflow flow with GoogleChatSender
        # or directly use the Google Chat API here

    def _process_message(self, pubsub_message):
        """Process a single Pub/Sub message."""
        parsed = self._parse_chat_message(pubsub_message.data)

        if parsed is None:
            return

        if not self._should_process(parsed):
            return

        logger.info(f"Processing message from {parsed['sender_name']}: {parsed['text'][:50]}...")

        # Call Langflow
        response = self._call_langflow(parsed)

        if response:
            self._send_reply(parsed, response)
        else:
            logger.warning("No response from Langflow")

    def callback(self, message):
        """Callback for streaming pull."""
        try:
            self._process_message(message)
        except Exception as e:  # noqa: BLE001
            logger.error(f"Error processing message: {e}")
        finally:
            # Always acknowledge to avoid redelivery
            message.ack()

    def run_streaming(self):
        """Run with streaming pull (recommended)."""
        logger.info("Starting streaming pull...")
        logger.info("Waiting for messages... (Ctrl+C to stop)")

        streaming_pull_future = self.subscriber.subscribe(self.subscription_path, callback=self.callback)

        # Handle graceful shutdown
        def shutdown(_signum, _frame):
            logger.info("Shutting down...")
            streaming_pull_future.cancel()
            self.running = False

        signal.signal(signal.SIGINT, shutdown)
        signal.signal(signal.SIGTERM, shutdown)

        try:
            streaming_pull_future.result()
        except FuturesTimeoutError:
            streaming_pull_future.cancel()
            streaming_pull_future.result()

    def run_polling(self):
        """Run with synchronous polling (alternative)."""
        logger.info("Starting polling mode...")
        logger.info("Waiting for messages... (Ctrl+C to stop)")

        def shutdown(_signum, _frame):
            logger.info("Shutting down...")
            self.running = False

        signal.signal(signal.SIGINT, shutdown)
        signal.signal(signal.SIGTERM, shutdown)

        while self.running:
            try:
                response = self.subscriber.pull(
                    request={
                        "subscription": self.subscription_path,
                        "max_messages": 10,
                    },
                    timeout=30,
                )

                for received_message in response.received_messages:
                    self._process_message(received_message.message)
                    self.subscriber.acknowledge(
                        request={
                            "subscription": self.subscription_path,
                            "ack_ids": [received_message.ack_id],
                        }
                    )

            except FuturesTimeoutError:
                # No messages, continue
                pass
            except Exception as e:  # noqa: BLE001
                logger.error(f"Pull error: {e}")
                time.sleep(5)


def main():
    parser = argparse.ArgumentParser(description="Google Chat Worker")
    parser.add_argument("--config", help="Path to config JSON file")
    parser.add_argument("--project-id", help="GCP Project ID")
    parser.add_argument("--subscription", help="Pub/Sub subscription name")
    parser.add_argument("--langflow-url", help="Langflow API URL")
    parser.add_argument("--flow-id", help="Langflow Flow ID")
    parser.add_argument("--service-account", help="Path to service account JSON")
    parser.add_argument("--mode", choices=["streaming", "polling"], default="streaming")

    args = parser.parse_args()

    # Load config from file or environment
    config = {}
    if args.config:
        with Path(args.config).open() as f:
            config = json.load(f)

    project_id = args.project_id or config.get("project_id") or os.getenv("GCP_PROJECT_ID")
    subscription = args.subscription or config.get("subscription") or os.getenv("PUBSUB_SUBSCRIPTION")
    langflow_url = (
        args.langflow_url or config.get("langflow_url") or os.getenv("LANGFLOW_API_URL", "http://localhost:7860")
    )
    flow_id = args.flow_id or config.get("flow_id") or os.getenv("LANGFLOW_FLOW_ID")
    service_account = (
        args.service_account or config.get("service_account") or os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    )

    if not all([project_id, subscription, flow_id]):
        parser.error("Missing required arguments: project-id, subscription, flow-id")

    worker = ChatWorker(
        project_id=project_id,
        subscription_name=subscription,
        langflow_api_url=langflow_url,
        langflow_flow_id=flow_id,
        service_account_json=service_account,
    )

    if args.mode == "streaming":
        worker.run_streaming()
    else:
        worker.run_polling()


if __name__ == "__main__":
    main()
