from types import SimpleNamespace
from zipfile import ZipFile

import pytest

from dbgpt_serve.utils.auth import UserRequest


class FakeUploadFile:
    def __init__(self, filename: str, content: bytes):
        self.filename = filename
        self.content_type = "text/x-python"
        self._content = content

    async def read(self) -> bytes:
        return self._content


@pytest.mark.asyncio
async def test_python_file_upload_rejects_traversal_filename(tmp_path, monkeypatch):
    from dbgpt_app.openapi.api_v1 import python_upload_api

    monkeypatch.setattr(
        python_upload_api.CFG, "SYSTEM_APP", SimpleNamespace(work_dir=str(tmp_path))
    )

    result = await python_upload_api.python_file_upload(
        FakeUploadFile("../../outside.py", b"print('escaped')"),
        UserRequest(user_id="alice"),
    )

    assert result.success is False
    assert not (tmp_path / "outside.py").exists()


@pytest.mark.asyncio
async def test_python_file_upload_rejects_absolute_filename(tmp_path, monkeypatch):
    from dbgpt_app.openapi.api_v1 import python_upload_api

    monkeypatch.setattr(
        python_upload_api.CFG, "SYSTEM_APP", SimpleNamespace(work_dir=str(tmp_path))
    )

    result = await python_upload_api.python_file_upload(
        FakeUploadFile(str(tmp_path / "outside.py"), b"print('escaped')"),
        UserRequest(user_id="alice"),
    )

    assert result.success is False
    assert not (tmp_path / "outside.py").exists()


@pytest.mark.asyncio
async def test_python_file_upload_allows_plain_filename_inside_user_dir(
    tmp_path, monkeypatch
):
    from dbgpt_app.openapi.api_v1 import python_upload_api

    monkeypatch.setattr(
        python_upload_api.CFG, "SYSTEM_APP", SimpleNamespace(work_dir=str(tmp_path))
    )

    result = await python_upload_api.python_file_upload(
        FakeUploadFile("inside.py", b"print('inside')"),
        UserRequest(user_id="alice"),
    )

    expected_path = tmp_path / "python_uploads" / "alice" / "inside.py"
    assert result.success is True
    assert result.data == str(expected_path)
    assert expected_path.read_bytes() == b"print('inside')"


@pytest.mark.asyncio
async def test_python_file_upload_rejects_symlink_escape(tmp_path, monkeypatch):
    from dbgpt_app.openapi.api_v1 import python_upload_api

    upload_dir = tmp_path / "python_uploads" / "alice"
    outside_dir = tmp_path / "outside"
    upload_dir.mkdir(parents=True)
    outside_dir.mkdir()
    try:
        (upload_dir / "linked").symlink_to(outside_dir, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlinks are not available in this environment: {exc}")

    monkeypatch.setattr(
        python_upload_api.CFG, "SYSTEM_APP", SimpleNamespace(work_dir=str(tmp_path))
    )

    result = await python_upload_api.python_file_upload(
        FakeUploadFile("linked/escaped.py", b"print('escaped')"),
        UserRequest(user_id="alice"),
    )

    assert result.success is False
    assert not (outside_dir / "escaped.py").exists()


@pytest.mark.asyncio
async def test_python_file_upload_extracts_zip_inside_user_dir(tmp_path, monkeypatch):
    from dbgpt_app.openapi.api_v1 import python_upload_api

    monkeypatch.setattr(
        python_upload_api.CFG, "SYSTEM_APP", SimpleNamespace(work_dir=str(tmp_path))
    )

    zip_path = tmp_path / "dataset.zip"
    with ZipFile(zip_path, "w") as zip_file:
        zip_file.writestr("reports/sales.csv", "store,sales\nA,10\n")
        zip_file.writestr("notes/readme.md", "# Notes\n")

    result = await python_upload_api.python_file_upload(
        FakeUploadFile("dataset.zip", zip_path.read_bytes()),
        UserRequest(user_id="alice"),
    )

    extracted_dir = tmp_path / "python_uploads" / "alice" / "dataset_extracted"
    assert result.success is True
    assert result.data == str(extracted_dir)
    assert (
        extracted_dir / "reports" / "sales.csv"
    ).read_text() == "store,sales\nA,10\n"
    assert (extracted_dir / "notes" / "readme.md").read_text() == "# Notes\n"


@pytest.mark.asyncio
async def test_python_file_upload_rejects_unsafe_zip_path(tmp_path, monkeypatch):
    from dbgpt_app.openapi.api_v1 import python_upload_api

    monkeypatch.setattr(
        python_upload_api.CFG, "SYSTEM_APP", SimpleNamespace(work_dir=str(tmp_path))
    )

    zip_path = tmp_path / "unsafe.zip"
    with ZipFile(zip_path, "w") as zip_file:
        zip_file.writestr("../outside.csv", "escaped")

    result = await python_upload_api.python_file_upload(
        FakeUploadFile("unsafe.zip", zip_path.read_bytes()),
        UserRequest(user_id="alice"),
    )

    assert result.success is False
    assert not (tmp_path / "python_uploads" / "outside.csv").exists()


@pytest.mark.asyncio
async def test_python_files_upload_preserves_folder_paths(tmp_path, monkeypatch):
    from dbgpt_app.openapi.api_v1 import python_upload_api

    monkeypatch.setattr(
        python_upload_api.CFG, "SYSTEM_APP", SimpleNamespace(work_dir=str(tmp_path))
    )

    result = await python_upload_api.python_files_upload(
        [
            FakeUploadFile("folder/sales.csv", b"store,sales\nA,10\n"),
            FakeUploadFile("folder/notes/readme.md", b"# Notes\n"),
        ],
        UserRequest(user_id="alice"),
    )

    upload_dir = tmp_path / "python_uploads" / "alice"
    assert result.success is True
    assert result.data == str(upload_dir)
    assert (upload_dir / "folder" / "sales.csv").read_text() == "store,sales\nA,10\n"
    assert (upload_dir / "folder" / "notes" / "readme.md").read_text() == "# Notes\n"
