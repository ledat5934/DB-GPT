import logging
import os
import shutil
import zipfile
from pathlib import Path
from typing import List

from fastapi import APIRouter, Depends, File, UploadFile

from dbgpt._private.config import Config
from dbgpt_app.openapi.api_view_model import Result
from dbgpt_serve.utils.auth import UserRequest, get_user_from_headers

router = APIRouter()
CFG = Config()
logger = logging.getLogger(__name__)

_MAX_ZIP_MEMBER_SIZE = 200 * 1024 * 1024
_MAX_ZIP_TOTAL_SIZE = 1024 * 1024 * 1024


def _safe_stem(filename: str) -> str:
    stem = Path(filename).stem.strip().replace(" ", "_")
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in stem)
    return safe or "uploaded_files"


def _resolve_upload_path(upload_dir: str, filename: str) -> str:
    upload_dir_path = Path(upload_dir).resolve()
    filename_path = Path(filename)
    if filename_path.is_absolute():
        raise ValueError("filename must be a relative path inside upload directory")

    file_path = (upload_dir_path / filename_path).resolve()
    try:
        file_path.relative_to(upload_dir_path)
    except ValueError as exc:
        raise ValueError("filename must stay inside upload directory") from exc
    return str(file_path)


def _get_upload_dir(user_token: UserRequest) -> str:
    user_id = user_token.user_id or "default"
    base_dir = os.getcwd()
    if (
        CFG.SYSTEM_APP
        and hasattr(CFG.SYSTEM_APP, "work_dir")
        and CFG.SYSTEM_APP.work_dir
    ):
        base_dir = CFG.SYSTEM_APP.work_dir

    upload_dir = os.path.join(base_dir, "python_uploads", user_id)
    os.makedirs(upload_dir, exist_ok=True)
    return upload_dir


def _extract_zip_file(zip_path: str, extract_dir: str) -> List[str]:
    extracted_files: List[str] = []
    extract_root = Path(extract_dir).resolve()
    total_size = 0

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        for zip_info in zip_ref.infolist():
            if zip_info.is_dir():
                continue
            if zip_info.file_size > _MAX_ZIP_MEMBER_SIZE:
                raise ValueError(f"Zip member is too large: {zip_info.filename}")
            total_size += zip_info.file_size
            if total_size > _MAX_ZIP_TOTAL_SIZE:
                raise ValueError("Zip file is too large after extraction")

            target_path = (extract_root / zip_info.filename).resolve()
            try:
                target_path.relative_to(extract_root)
            except ValueError as exc:
                raise ValueError("Zip file contains an unsafe path") from exc

        zip_ref.extractall(extract_root)

    for path in extract_root.rglob("*"):
        if path.is_file():
            extracted_files.append(str(path.resolve()))
    return extracted_files


async def _save_upload_file(upload_dir: str, file: UploadFile) -> str:
    if not file or not file.filename:
        raise ValueError("No file provided or filename is empty")

    file_path = _resolve_upload_path(upload_dir, file.filename)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    content = await file.read()
    if not content:
        raise ValueError("Uploaded file is empty")

    with open(file_path, "wb") as buffer:
        buffer.write(content)
    return os.path.abspath(file_path)


async def _save_uploads(upload_dir: str, files: List[UploadFile]) -> str:
    saved_paths: List[str] = []
    for file in files:
        saved_path = await _save_upload_file(upload_dir, file)
        saved_paths.append(saved_path)

        if file.filename and file.filename.lower().endswith(".zip"):
            extract_dir = _resolve_upload_path(
                upload_dir, f"{_safe_stem(file.filename)}_extracted"
            )
            if os.path.exists(extract_dir):
                shutil.rmtree(extract_dir)
            os.makedirs(extract_dir, exist_ok=True)
            _extract_zip_file(saved_path, extract_dir)
            return os.path.abspath(extract_dir)

    if len(saved_paths) == 1:
        return saved_paths[0]

    return os.path.abspath(upload_dir)


@router.post("/v1/python/file/upload", response_model=Result[str])
async def python_file_upload(
    file: UploadFile = File(...),
    user_token: UserRequest = Depends(get_user_from_headers),
):
    try:
        if not file or not file.filename:
            return Result.failed(msg="No file provided or filename is empty")

        user_id = user_token.user_id or "default"
        logger.info(
            f"Uploading file: {file.filename}, content_type: {file.content_type}, "
            f"user: {user_id}"
        )

        upload_dir = _get_upload_dir(user_token)
        abs_path = await _save_uploads(upload_dir, [file])
        logger.info(f"File uploaded successfully to {abs_path}")

        return Result.succ(abs_path)
    except Exception as e:
        logger.exception(f"File upload failed: {e}")
        return Result.failed(msg=f"Upload error: {str(e)}")


@router.post("/v1/python/files/upload", response_model=Result[str])
async def python_files_upload(
    files: List[UploadFile] = File(...),
    user_token: UserRequest = Depends(get_user_from_headers),
):
    try:
        if not files:
            return Result.failed(msg="No files provided")

        upload_dir = _get_upload_dir(user_token)
        result_path = await _save_uploads(upload_dir, files)
        logger.info(f"Files uploaded successfully to {result_path}")
        return Result.succ(result_path)
    except Exception as e:
        logger.exception(f"Files upload failed: {e}")
        return Result.failed(msg=f"Upload error: {str(e)}")
