"""Google Chat Sender component for Langflow."""

import json
import time

import requests
from google.oauth2 import service_account
from loguru import logger

from langflow.custom import Component
from langflow.io import BoolInput, HandleInput, MessageTextInput, Output, SecretStrInput
from langflow.schema.message import Message

# Constants
REQUEST_TIMEOUT = 30
GOOGLE_CHAT_API_BASE = "https://chat.googleapis.com/v1"


class GoogleChatSender(Component):
    """Send messages to Google Chat space/thread.

    Use this component to forward customer messages to a Google Chat space
    where support operators can monitor conversations.

    Typical flow:
    Chat Input → Google Chat Sender

    Features:
    - Uses session_id directly as thread_key
    - Groups all messages from same session in one Google Chat thread
    - No Firestore lookup needed for thread_key
    """

    display_name = "Google Chat Sender"
    description = "Send messages to Google Chat. Uses session_id as thread_key."
    icon = "MessageSquare"

    # Class-level cache: thread_key -> thread_name mapping
    _thread_cache: dict[str, str] = {}

    inputs = [
        SecretStrInput(
            name="service_account_json",
            display_name="Service Account JSON",
            info="Google Cloud Service Account JSON with Chat API access",
            required=True,
        ),
        MessageTextInput(
            name="space_id",
            display_name="Space ID",
            info="Google Chat Space ID (e.g., spaces/AAAA...)",
            required=True,
        ),
        HandleInput(
            name="message_input",
            display_name="Message",
            info="Message to send to Google Chat",
            input_types=["Message", "Data"],
            required=True,
        ),
        MessageTextInput(
            name="sender_name",
            display_name="Sender Name",
            info="Display name for the sender (e.g., 'Customer' or 'AI Bot')",
            value="Customer",
        ),
        MessageTextInput(
            name="thread_name",
            display_name="Thread Name (Full)",
            info="Full thread name (e.g., spaces/xxx/threads/yyy). Override auto-lookup.",
            required=False,
            advanced=True,
        ),
        BoolInput(
            name="clear_cache",
            display_name="Clear Thread Cache",
            info="Clear the thread cache before sending",
            value=False,
            advanced=True,
        ),
    ]

    outputs = [
        Output(display_name="Message", name="message_output", method="send_message"),
    ]

    def _get_session_id(self) -> str:
        """Get session_id from message or flow context."""
        session_id = ""

        # Method 1: From message_input
        if self.message_input:
            session_id = getattr(self.message_input, "session_id", "") or ""

        # Method 2: From graph.session_id
        if not session_id and hasattr(self, "graph") and self.graph:
            session_id = getattr(self.graph, "session_id", "") or ""

        return session_id.strip() if session_id else ""

    def _get_thread_key(self) -> str:
        """Get thread_key from session_id (use session_id directly as thread_key)."""
        session_id = self._get_session_id()

        if not session_id:
            logger.warning("No session_id found, cannot create thread_key")
            return ""

        logger.info(f"Using session_id as thread_key: {session_id}")
        return session_id

    def _get_access_token(self, service_account_info: dict) -> str:
        """Get access token for Google Chat API."""
        credentials = service_account.Credentials.from_service_account_info(
            service_account_info,
            scopes=["https://www.googleapis.com/auth/chat.bot"],
        )

        # Refresh to get access token
        from google.auth.transport.requests import Request

        credentials.refresh(Request())
        return credentials.token

    def _format_message(self) -> str:
        """Format the message content."""
        if hasattr(self.message_input, "text"):
            content = self.message_input.text
        elif hasattr(self.message_input, "get_text"):
            content = self.message_input.get_text()
        else:
            content = str(self.message_input)

        # Add sender prefix
        sender = self.sender_name or "Unknown"
        return f"*{sender}:*\n{content}"

    def send_message(self) -> Message:
        """Send message to Google Chat space and pass through the original message."""
        logger.info("=== Google Chat Sender: Starting ===")
        start_time = time.time()

        # Clear cache if requested
        if getattr(self, "clear_cache", False):
            logger.info("Clearing thread cache...")
            GoogleChatSender._thread_cache.clear()
            logger.info("Cache cleared")

        # Debug: show current cache state
        logger.info(f"Current cache: {GoogleChatSender._thread_cache}")

        # Parse service account
        if not self.service_account_json or not self.service_account_json.strip():
            msg = "Service Account JSON is required"
            raise ValueError(msg)

        try:
            service_account_info = json.loads(self.service_account_json, strict=False)
        except json.JSONDecodeError as e:
            msg = f"Invalid Service Account JSON: {e}"
            raise ValueError(msg) from e

        # logger.debug(f"Service Account: {service_account_info.get('client_email')}")

        # Get access token
        # logger.debug("Getting access token...")
        access_token = self._get_access_token(service_account_info)
        # logger.debug(f"Access token obtained in {time.time() - start_time:.2f}s")

        # Build request
        space_id = self.space_id.strip()
        if not space_id.startswith("spaces/"):
            space_id = f"spaces/{space_id}"

        url = f"{GOOGLE_CHAT_API_BASE}/{space_id}/messages"

        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
        }

        # Format message
        message_text = self._format_message()
        # logger.debug(f"Message length: {len(message_text)} chars")

        payload = {
            "text": message_text,
        }

        # Add thread reference
        thread_name_input = getattr(self, "thread_name", "")
        thread_name_input = thread_name_input.strip() if thread_name_input else ""

        # Get thread_key (use session_id directly)
        thread_key = self._get_thread_key()

        # Build cache key (space + thread_key)
        cache_key = f"{space_id}:{thread_key}" if thread_key else ""

        if thread_name_input:
            # Use explicit thread name (highest priority)
            payload["thread"] = {"name": thread_name_input}
            url += "?messageReplyOption=REPLY_MESSAGE_FALLBACK_TO_NEW_THREAD"
            # logger.info(f"Using explicit thread name: {thread_name_input}")
        elif cache_key and cache_key in GoogleChatSender._thread_cache:
            # Use cached thread name (most reliable for subsequent messages)
            cached_name = GoogleChatSender._thread_cache[cache_key]
            payload["thread"] = {"name": cached_name}
            url += "?messageReplyOption=REPLY_MESSAGE_FALLBACK_TO_NEW_THREAD"
            # logger.info(f"Using cached thread name: {cached_name} (key: {thread_key})")
        elif thread_key:
            # Use thread key (first message with this key)
            payload["thread"] = {"threadKey": thread_key}
            url += "?messageReplyOption=REPLY_MESSAGE_FALLBACK_TO_NEW_THREAD"
            # logger.info(f"Using thread key (first message): {thread_key}")
        else:
            logger.info("No thread specified - creating new thread")

        # Send request
        # logger.info(f"Sending to: {space_id}")
        # logger.info(f"URL: {url}")
        # logger.info(f"Payload thread: {payload.get('thread', 'None')}")
        time.time()

        try:
            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=REQUEST_TIMEOUT,
            )
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            logger.error(f"Google Chat API error: {e}")
            if hasattr(e, "response") and e.response is not None:
                logger.error(f"Response: {e.response.text}")
            raise

        result = response.json()
        # logger.info(f"Message sent in {time.time() - send_start:.2f}s")
        # logger.info(f"Message name: {result.get('name')}")

        # Extract thread name for future replies
        thread_name_result = result.get("thread", {}).get("name", "")
        result.get("thread", {}).get("threadKey", "")
        self._thread_name = thread_name_result

        # Cache thread name for future messages with same thread_key
        if cache_key and thread_name_result:
            GoogleChatSender._thread_cache[cache_key] = thread_name_result
            logger.info(f"Cached thread: {cache_key} -> {thread_name_result}")

        # logger.info(f"Thread name: {thread_name_result}")
        # logger.info(f"Thread key used: {thread_key_used}")
        # logger.info(f"Cache size: {len(GoogleChatSender._thread_cache)}")

        logger.info(f"return: {self.message_input}")

        logger.info(f"=== Google Chat Sender: Complete in {time.time() - start_time:.2f}s ===")

        # Pass through the original message
        return self.message_input
