"""Google Drive File Lister component for Langflow."""

import json

from google.oauth2 import service_account
from googleapiclient.discovery import build

from langflow.custom import Component
from langflow.io import DropdownInput, IntInput, MessageTextInput, Output, SecretStrInput
from langflow.schema import Data

# File type to MIME type mapping
FILE_TYPE_MIME_MAP = {
    "Images": "mimeType contains 'image/'",
    "Documents": "mimeType = 'application/vnd.google-apps.document'",
    "Spreadsheets": "mimeType = 'application/vnd.google-apps.spreadsheet'",
    "PDFs": "mimeType = 'application/pdf'",
    "Videos": "mimeType contains 'video/'",
    "Audio": "mimeType contains 'audio/'",
    "All Files": "mimeType != 'application/vnd.google-apps.folder'",
}


class GoogleDriveFileLister(Component):
    """List files from Google Drive by type."""

    display_name = "G-Drive File Lister"
    description = "Google Driveからファイルを検索してリストアップします"
    icon = "GoogleDrive"

    inputs = [
        SecretStrInput(
            name="service_account_json",
            display_name="Service Account JSON",
            info="Service Account JSON key content (paste the JSON)",
            required=True,
        ),
        DropdownInput(
            name="file_type",
            display_name="File Type",
            info="検索するファイルの種類を選択",
            options=["Images", "Documents", "Spreadsheets", "PDFs", "Videos", "Audio", "All Files", "Custom"],
            value="Images",
        ),
        MessageTextInput(
            name="custom_mime_type",
            display_name="Custom MIME Type",
            info="File TypeでCustomを選択した場合のMIMEタイプ (例: application/zip)",
            advanced=True,
        ),
        MessageTextInput(
            name="folder_id",
            display_name="Folder ID (Optional)",
            info="特定のフォルダ内だけ検索したい場合はIDを入力 (空欄なら全体検索)",
            advanced=True,
        ),
        IntInput(
            name="limit",
            display_name="Max Results",
            value=10,
        ),
    ]

    outputs = [
        Output(display_name="File List", name="files", method="list_files"),
    ]

    def list_files(self) -> list[Data]:
        """List files from Google Drive."""
        if not self.service_account_json or not self.service_account_json.strip():
            msg = "Service Account JSON is required"
            raise ValueError(msg)

        # Parse and authenticate
        try:
            service_account_info = json.loads(self.service_account_json, strict=False)
            creds = service_account.Credentials.from_service_account_info(
                service_account_info,
                scopes=["https://www.googleapis.com/auth/drive.readonly"],
            )
            service = build("drive", "v3", credentials=creds)
        except json.JSONDecodeError as e:
            msg = f"Invalid JSON: {e!s}"
            raise ValueError(msg) from e
        except (OSError, ValueError) as e:
            msg = f"認証エラー: {e!s}"
            raise ValueError(msg) from e

        # Build MIME type query based on selection
        file_type = getattr(self, "file_type", "Images")

        if file_type == "Custom":
            custom_mime = getattr(self, "custom_mime_type", "")
            if not custom_mime:
                msg = "Custom MIME typeを指定してください"
                raise ValueError(msg)
            mime_query = f"mimeType = '{custom_mime}'"
        else:
            mime_query = FILE_TYPE_MIME_MAP.get(file_type, FILE_TYPE_MIME_MAP["Images"])

        # Build search query (selected type, not trashed)
        query = f"{mime_query} and trashed = false"

        # Add folder filter if specified
        if self.folder_id:
            query += f" and '{self.folder_id}' in parents"

        # Execute API
        results = (
            service.files()
            .list(
                q=query,
                pageSize=self.limit,
                fields="nextPageToken, files(id, name, mimeType, webContentLink, webViewLink, description)",
                supportsAllDrives=True,
            )
            .execute()
        )

        items = results.get("files", [])

        # Convert to Langflow Data format
        data_list = []
        for item in items:
            data = Data(
                text=item.get("name"),
                data={
                    "file_id": item.get("id"),
                    "mime_type": item.get("mimeType"),
                    "download_link": item.get("webContentLink"),
                    "view_link": item.get("webViewLink"),
                    "description": item.get("description", ""),
                },
            )
            data_list.append(data)

        return data_list
