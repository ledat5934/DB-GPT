import json
import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd

from ..core.agent import Agent, AgentMessage
from ..core.profile import DynConfig, ProfileConfig
from ..resource.database import DBResource
from . import excel_path
from .actions.insert_action import Excel2TableAction

logger = logging.getLogger(__name__)


def find_excel_files(directory: str) -> list[str]:
    """
    Find all .csv and .xlsx files under the specified directory and return their
    absolute paths.

    Args:
        directory: Directory path to search.

    Returns:
        A list of absolute paths for all .csv and .xlsx files, or an empty list if
        the directory does not exist.
    """
    # Check whether the directory exists
    if not os.path.isdir(directory):
        return []

    # Store the result list
    file_paths = []

    # Traverse the directory and its subdirectories
    for root, dirs, files in os.walk(directory):
        for file in files:
            # Check the file extension
            if file.lower().endswith((".csv", ".xlsx")):
                # Get the absolute file path and add it to the list
                absolute_path = os.path.abspath(os.path.join(root, file))
                file_paths.append(absolute_path)

    return file_paths


excel_files = find_excel_files(excel_path)


# TODO Implementation of Excel Agent Function, temp hidden Agent display for page
# class Excel2TableAgent(ConversableAgent):
class Excel2TableAgent:
    """Excel Scientist Agent."""

    profile: ProfileConfig = ProfileConfig(
        name=DynConfig(
            "ExcelScientistAgent",
            category="agent",
            key="dbgpt_agent_expand_excel2table_agent_profile_name",
        ),
        role=DynConfig(
            "ExcelScientist",
            category="agent",
            key="dbgpt_agent_expand_excel2table_agent_profile_role",
        ),
        goal=DynConfig(
            "Based on the Excel table header (Chinese/English) and sample "
            "data, complete 3 core tasks: "
            "1. Field name adaptive processing: If the Excel header is "
            "already in English (snake_case/camelCase), retain it directly "
            "without translation; if it is Chinese, convert it to standard "
            "English snake_case (e.g., product ID -> product_id); "
            "2. Field order strict alignment: The field order in the CREATE "
            "TABLE SQL must be exactly the same as the header order in the "
            "Excel table (to support subsequent data insertion by field order); "
            "3. Generate {{dialect}} database table creation SQL (including "
            "primary key, field type, length constraint) and a semantic table"
            " name (snake_case). "
            "Finally, provide the result in the specified JSON format (only "
            "table name and create SQL) to support subsequent automatic data "
            "insertion.",
            category="agent",
            key="dbgpt_agent_expand_excel2table_agent_profile_goal",
        ),
        constraints=DynConfig(
            [
                "Field naming: Detect English vs Chinese headers.  Keep English headers"
                " in snake_case;  translate Chinese headers to meaningful English "
                "snake_case.  Ensure uniqueness.",
                "Field order: SQL field order must strictly follow Excel header order, "
                "no rearrangement.",
                "Field completeness: All Excel headers must be included in the CREATE "
                "TABLE SQL, no omission.",
                "SQL generation: Infer field types from sample data;  specify VARCHAR "
                "length;  add PRIMARY KEY for ID fields;  apply NOT NULL "
                "where required.",
                "Table naming: Use [module]_[data_type], snake_case, ≤30 chars, avoid "
                "reserved words.",
                "History consistency: If headers were converted before, keep the same "
                "English names and order;  avoid duplicates.",
            ],
            category="agent",
            key="dbgpt_agent_expand_excel2table_agent_profile_constraints",
        ),
        desc=DynConfig(
            "Store the data in one or more Excel tables mentioned by the user"
            " respectively in the database to facilitate subsequent "
            "statistical analysis of the data therein.",
            category="agent",
            key="dbgpt_agent_expand_excel2table_agent_profile_desc",
        ),
    )

    max_retry_count: int = 1
    language: str = "en"

    def __init__(self, **kwargs):
        """Create a new DataScientistAgent instance."""
        super().__init__(**kwargs)
        self._init_actions([Excel2TableAction])

    async def thinking(
        self,
        messages: List[AgentMessage],
        sender: Optional[Agent] = None,
        prompt: Optional[str] = None,
        stream_callback: Optional[Callable[[Dict[str, Any]], Any]] = None,
    ) -> Tuple[Optional[str], Optional[str]]:
        all_file_data = []
        for excel_file in excel_files:
            filename_with_ext = os.path.basename(excel_file)
            headers, table_data = read_excel_headers_and_data(excel_file)
            mdstr = data2md(headers, table_data)
            all_file_data.append((filename_with_ext, mdstr))
        message_parts = ["Sample data from the Excel files:"]
        for i, (filename, mdstr) in enumerate(all_file_data, 1):
            message_parts.append(f"\nFile {i}: {filename}")
            message_parts.append(f"Data content:\n{mdstr}")
        prompt = "\n".join(message_parts)
        result = await super().thinking(
            messages, sender, prompt, stream_callback=stream_callback
        )
        return result

    def _init_reply_message(
        self,
        received_message: AgentMessage,
        rely_messages: Optional[List[AgentMessage]] = None,
    ) -> AgentMessage:
        reply_message = super()._init_reply_message(received_message, rely_messages)
        reply_message.context = {
            "display_type": self.actions[0].render_prompt(),
            "dialect": self.database.dialect,
        }
        return reply_message

    @property
    def database(self) -> DBResource:
        """Get the database resource."""
        dbs: List[DBResource] = DBResource.from_resource(self.resource)
        if not dbs:
            raise ValueError(
                f"Resource type {self.actions[0].resource_need} is not supported."
            )
        return dbs[0]

    async def correctness_check(
        self, message: AgentMessage
    ) -> Tuple[bool, Optional[str]]:
        """Verify whether the current execution results meet the target expectations "
        "for multiple tables."""
        action_out = message.action_report
        if action_out is None:
            return (
                False,
                f"No executable analysis SQL is generated, {message.content}.",
            )

        if not action_out.is_exe_success:
            return (
                False,
                f"Please check your answer, {action_out.content}.",
            )

        try:
            action_reply_obj = json.loads(action_out.content)
            tables = action_reply_obj.get("tables", [])
            database = action_reply_obj.get("database")

            if not action_out.resource_value or not database:
                return (
                    False,
                    "Please check your answer, the data resource information "
                    "is not found.",
                )

            if not tables or len(tables) == 0:
                return (
                    False,
                    "No table information found in the execution result.",
                )

            # Verify data for each table
            for table_info in tables:
                table_name = table_info.get("table_name")
                if not table_name:
                    return (
                        False,
                        "Missing table name in execution result.",
                    )

                # Check whether the table exists
                check_table_sql = f"""
                        SELECT COUNT(*) AS table_exists 
                        FROM sqlite_master 
                        WHERE type='table' AND name='{table_name}';
                        """
                cols, vals = await self.database.query(
                    sql=check_table_sql,
                    db=action_out.resource_value,
                )
                if not vals or vals[0][0] == 0:
                    return (
                        False,
                        f"Table {table_name} was not created successfully.",
                    )

                # Check whether data was inserted successfully
                count_sql = f"SELECT COUNT(*) AS total_records FROM {table_name};"
                columns, values = await self.database.query(
                    sql=count_sql,
                    db=action_out.resource_value,
                )

                if not values or len(values) <= 0 or values[0][0] == 0:
                    return (
                        False,
                        f"Table {table_name} exists but contains no data. Please "
                        "check the data insertion process.",
                    )

                logger.info(
                    f"Table {table_name} verification success! There are "
                    f"{values[0][0]} rows of data."
                )

            # All tables passed verification
            logger.info(f"All {len(tables)} tables verification success!")
            return True, None

        except Exception as e:
            logger.exception(f"DataScientist check exception!{str(e)}")
            return (
                False,
                f"Verification error, please re-read the historical information to "
                "fix this. "
                f"The error message is as follows: {str(e)}",
            )


def read_excel_headers_and_data(
    file_path: str, read_rows: Optional[int] = 3
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    Read an Excel file and return header information and structured data, with
    support for specifying the number of rows to read.

    Args:
        file_path: Excel file path in .xlsx format.
        read_rows: Optional number of data rows to read, excluding the header.
            Defaults to 3; use None or 0 to read all rows.

    Returns:
        A tuple of headers and data rows. Headers are read from the first row, and
        each data row is a dictionary keyed by header with empty cells converted to
        None.
    """
    # 1. Basic file validation
    if not Path(file_path).exists():
        raise FileNotFoundError(f"File does not exist: {file_path}")
    if Path(file_path).suffix.lower() != ".xlsx":
        raise ValueError(
            f"Unsupported file format: {Path(file_path).suffix}; "
            "only .xlsx is supported"
        )

    try:
        # 2. Read Excel data first, then truncate as needed
        df = pd.read_excel(
            file_path,
            sheet_name=0,  # Read the first worksheet
            engine="openpyxl",
            # Convert empty cells to empty strings first, then normalize later.
            keep_default_na=False,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to read Excel: {str(e)}")

    # 3. Header extraction and validation
    headers = list(df.columns)
    if not headers:
        raise ValueError(
            "The Excel file has no header information (the first row is empty)"
        )

    # 4. Handle read_rows by selecting the target number of data rows
    total_data_rows = len(df)  # Total data rows excluding the header
    # If all rows are requested (None/0), use all rows; otherwise use the
    # smaller of requested and available rows.
    if read_rows in (None, 0):
        target_rows = total_data_rows
    elif isinstance(read_rows, int) and read_rows > 0:
        target_rows = min(read_rows, total_data_rows)
    else:
        raise ValueError(
            f"Invalid read_rows parameter: {read_rows}; only positive integers, "
            "None, or 0 are supported"
        )

    # Select the target rows to avoid reading irrelevant rows and improve efficiency
    df_target = df.head(target_rows)

    # 5. Structure data as a list of dictionaries and convert empty strings to None
    data = []
    for _, row in df_target.iterrows():
        row_data = {
            header: (row[header] if row[header] != "" else None) for header in headers
        }
        data.append(row_data)

    return headers, data


def data2md(headers, table_data):
    md_lines = []

    md_lines.append("| " + " | ".join(headers) + " |")
    md_lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

    for row in table_data:
        values = []
        for h in headers:
            val = row.get(h, "")
            if hasattr(val, "strftime"):  # datetime
                values.append(val.strftime("%Y-%m-%d"))
            else:
                values.append(str(val))
        md_lines.append("| " + " | ".join(values) + " |")

    markdown_table = "\n".join(md_lines)
    return markdown_table
