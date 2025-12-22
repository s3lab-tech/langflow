"""Google Drive Image Trigger component for Langflow."""

import json

from google.oauth2 import service_account
from googleapiclient.discovery import build
from loguru import logger

from langflow.custom import Component
from langflow.io import BoolInput, IntInput, MessageTextInput, Output, SecretStrInput
from langflow.schema import Data
from langflow.schema.dataframe import DataFrame


class GoogleDriveImageTrigger(Component):
    """Trigger when new images are found in a Google Drive folder."""

    display_name = "G-Drive Image Trigger"
    description = "Google Driveフォルダ内の画像を検出し、ファイルIDを出力します"
    icon = "GoogleDrive"

    # Class-level cache for tracking seen files (persists across runs in same session)
    _seen_files: set[str] = set()

    inputs = [
        MessageTextInput(
            name="trigger",
            display_name="Trigger",
            info="Connect Chat Input here to trigger the flow",
            required=False,
        ),
        SecretStrInput(
            name="service_account_json",
            display_name="Service Account JSON",
            info="Service Account JSON key content (paste the JSON)",
            required=True,
        ),
        MessageTextInput(
            name="folder_id",
            display_name="Folder ID",
            info="監視するGoogle DriveフォルダのID",
            required=True,
        ),
        BoolInput(
            name="only_new_files",
            display_name="Only New Files",
            info="前回の実行以降に追加された新しいファイルのみを検出",
            value=True,
        ),
        BoolInput(
            name="reset_cache",
            display_name="Reset Cache",
            info="キャッシュをリセットして全ファイルを再検出",
            value=False,
            advanced=True,
        ),
        IntInput(
            name="limit",
            display_name="Max Results",
            info="取得する最大ファイル数",
            value=10,
        ),
    ]

    outputs = [
        Output(display_name="File IDs", name="file_ids", method="check_for_images", type_=DataFrame),
        Output(display_name="Has New Images", name="has_new", method="has_new_images"),
    ]

    def _get_drive_service(self):
        """Create Google Drive API service."""
        if not self.service_account_json or not self.service_account_json.strip():
            msg = "Service Account JSON is required"
            raise ValueError(msg)

        try:
            service_account_info = json.loads(self.service_account_json, strict=False)
            creds = service_account.Credentials.from_service_account_info(
                service_account_info,
                scopes=["https://www.googleapis.com/auth/drive.readonly"],
            )
            logger.info(f"Authenticated with service account: {creds.service_account_email}")
            return build("drive", "v3", credentials=creds)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON: {e!s}")
            msg = f"Invalid JSON: {e!s}"
            raise ValueError(msg) from e
        except (OSError, ValueError) as e:
            logger.error(f"Authentication error: {e!s}")
            msg = f"認証エラー: {e!s}"
            raise ValueError(msg) from e

    def _fetch_images(self) -> list[dict]:
        """Fetch images from the specified folder."""
        service = self._get_drive_service()

        # Build query for images in the specified folder
        query = f"mimeType contains 'image/' and '{self.folder_id}' in parents and trashed = false"
        logger.info(f"Searching with query: {query}")
        logger.info(f"Folder ID: {self.folder_id}")
        logger.info(f"Limit: {self.limit}")

        # First, verify folder access by getting folder info
        try:
            folder_info = (
                service.files()
                .get(
                    fileId=self.folder_id,
                    fields="id, name, mimeType",
                    supportsAllDrives=True,
                )
                .execute()
            )
            logger.info(f"Folder found: {folder_info.get('name')} (Type: {folder_info.get('mimeType')})")
        except Exception as e:
            logger.error(f"Cannot access folder: {e!s}")
            logger.error("Make sure the folder is shared with the Service Account email")
            msg = f"Cannot access folder {self.folder_id}: {e!s}"
            raise ValueError(msg) from e

        results = (
            service.files()
            .list(
                q=query,
                pageSize=self.limit,
                fields="files(id, name, mimeType, createdTime, modifiedTime)",
                orderBy="createdTime desc",
                supportsAllDrives=True,
                includeItemsFromAllDrives=True,  # Required for Shared Drives
            )
            .execute()
        )

        files = results.get("files", [])
        logger.info(f"Found {len(files)} images in folder")

        if len(files) == 0:
            # Try listing ALL files in folder for debugging
            logger.info("No images found. Checking all files in folder...")
            all_files_query = f"'{self.folder_id}' in parents and trashed = false"
            all_results = (
                service.files()
                .list(
                    q=all_files_query,
                    pageSize=self.limit,
                    fields="files(id, name, mimeType)",
                    supportsAllDrives=True,
                    includeItemsFromAllDrives=True,
                )
                .execute()
            )
            all_files = all_results.get("files", [])
            logger.info(f"Total files in folder: {len(all_files)}")
            for f in all_files:
                logger.info(f"  - {f.get('name')} (Type: {f.get('mimeType')})")

        for f in files:
            logger.debug(f"  - {f.get('name')} (ID: {f.get('id')}, Type: {f.get('mimeType')})")

        return files

    def check_for_images(self) -> DataFrame:
        """Check for images and return file IDs as DataFrame."""
        logger.info("=== G-Drive Image Trigger: check_for_images ===")
        logger.info(f"Only new files: {self.only_new_files}")
        logger.info(f"Reset cache: {self.reset_cache}")
        logger.info(f"Current cache size: {len(GoogleDriveImageTrigger._seen_files)}")

        # Reset cache if requested
        if self.reset_cache:
            logger.info("Resetting cache...")
            GoogleDriveImageTrigger._seen_files.clear()

        images = self._fetch_images()

        # Filter to only new files if enabled
        if self.only_new_files:
            new_images = [img for img in images if img["id"] not in GoogleDriveImageTrigger._seen_files]
            logger.info(f"Filtered to {len(new_images)} new images (out of {len(images)} total)")
        else:
            new_images = images
            logger.info(f"Returning all {len(new_images)} images")

        # Update seen files cache
        for img in images:
            GoogleDriveImageTrigger._seen_files.add(img["id"])

        # Convert to Data objects
        data_list = []
        for img in new_images:
            logger.debug(f"Adding image: {img.get('name')} (ID: {img['id']})")
            data = Data(
                text=img["id"],  # File ID as main text for easy chaining
                data={
                    "file_id": img["id"],
                    "file_name": img.get("name", ""),
                    "mime_type": img.get("mimeType", ""),
                    "created_time": img.get("createdTime", ""),
                    "modified_time": img.get("modifiedTime", ""),
                },
            )
            data_list.append(data)

        logger.info(f"Returning DataFrame with {len(data_list)} items")
        return DataFrame(data_list)

    def has_new_images(self) -> Data:
        """Check if there are new images (boolean output)."""
        # Reset cache if requested
        if self.reset_cache:
            GoogleDriveImageTrigger._seen_files.clear()

        images = self._fetch_images()

        # Check for new files
        if self.only_new_files:
            new_images = [img for img in images if img["id"] not in GoogleDriveImageTrigger._seen_files]
            has_new = len(new_images) > 0
            count = len(new_images)
        else:
            has_new = len(images) > 0
            count = len(images)

        # Update seen files cache
        for img in images:
            GoogleDriveImageTrigger._seen_files.add(img["id"])

        return Data(
            text=str(has_new),
            data={
                "has_new_images": has_new,
                "new_image_count": count,
                "total_images_in_folder": len(images),
            },
        )
