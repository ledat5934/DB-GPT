import json

from dbgpt._private.config import Config
from dbgpt.core import (
    ChatPromptTemplate,
    HumanPromptTemplate,
    SystemPromptTemplate,
)
from dbgpt_app.scene import AppScenePromptTemplateAdapter, ChatScene
from dbgpt_app.scene.chat_data.chat_excel.excel_learning.out_parser import (
    LearningExcelOutputParser,
)

CFG = Config()

_PROMPT_SCENE_DEFINE_EN = "You are a data analysis expert. "
_DEFAULT_TEMPLATE_EN = """
You are provided with user data and asked to understand and respond according to the \
requirements below.
The data is currently in a DuckDB table, \
a sample of which is as follows:
``````json
{data_example}
``````
The table summary information is as follows:
``````json
{table_summary}
``````
The DuckDB table structure information is as follows:
{table_schema}
Analyze the meaning and function of each column of data, and provide simple and clear \
explanations of technical terms, \
with the following specific requirements:
1. Carefully read the table structure, data samples, and table summary information \
provided to you
2. Extract information such as column names, data types, data meanings, data formats, \
etc. 3. To standardize the data structure, I need to transform the original column \
names, such as converting "Age" to "age", "Completion progress" to \
"completion_progress", etc.
4. You need to provide the original column names, transformed column names, \
data types, data meanings, data formats, etc.
5. If it's a time type, please provide the time format, such as: yyyy-MM-dd HH:MM:ss.
6. Please provide some useful analysis ideas from different dimensions for the user \
(arranged in order from simple to complex analysis complexity)
7. You need to output the extracted information according to the format below, \
ensuring that the output format is correct Column name conversion rules:
1. If it's in English letters, convert them all to lowercase, and replace spaces with \
underscores
2. If it's numbers, keep them as is
3. If it's Chinese, translate the Chinese field names to English, and replace spaces \
with underscores
4. If it's in other languages, translate them to English, and replace spaces with \
underscores
5. If it's special characters, delete them directly
6. DuckDB adheres to the SQL standard, which requires that identifiers \
(column names, table names) cannot start with a number.
7. All column fields must be analyzed and converted, remember to output in JSON
Avoid phrases like ' // ... (similar analysis for other columns) ...'
8. You need to provide the original column names and the transformed new column names \
in the JSON, as well as your analysis of the meaning and function of that column. If \
it's a time type, please provide the time format, such as: \
yyyy-MM-dd HH:MM:ss
You must output JSON data, where:
The `data_analysis` property is a summary of the data content analysis, \
The `column_analysis` is a JSON array type containing the conversion and analysis \
results for each column, \
The `analysis_program` property is the analysis approach.
Please think step by step, ensure that you answer only in JSON format, and ensure it \
can be parsed by Python's json.loads() function.
Response format is as follows:
```json
    {response}
```
"""

_PROMPT_SCENE_DEFINE_ZH = _PROMPT_SCENE_DEFINE_EN
_DEFAULT_TEMPLATE_ZH = _DEFAULT_TEMPLATE_EN

_RESPONSE_FORMAT_SIMPLE_EN = {
    "data_analysis": "Data content analysis summary",
    "column_analysis": [
        {
            "old_column_name": "Original column name",
            "new_column_name": "Converted new column name",
            "column_description": "Description of field 1, explanation of professional "
            "terms (as simple and clear as possible)",
        }
    ],
    "analysis_program": ["1. Analysis plan ", "2. Analysis plan "],
}
_RESPONSE_FORMAT_SIMPLE_ZH = _RESPONSE_FORMAT_SIMPLE_EN

RESPONSE_FORMAT_SIMPLE = (
    _RESPONSE_FORMAT_SIMPLE_EN if CFG.LANGUAGE == "en" else _RESPONSE_FORMAT_SIMPLE_ZH
)


_DEFAULT_TEMPLATE = (
    _DEFAULT_TEMPLATE_EN if CFG.LANGUAGE == "en" else _DEFAULT_TEMPLATE_ZH
)

PROMPT_SCENE_DEFINE = (
    _PROMPT_SCENE_DEFINE_EN if CFG.LANGUAGE == "en" else _PROMPT_SCENE_DEFINE_ZH
)

_USER_INPUT = "Please analyze the data for you"
_USER_INPUT_ZH = _USER_INPUT

USER_INPUT = _USER_INPUT if CFG.LANGUAGE == "en" else _USER_INPUT_ZH


PROMPT_NEED_STREAM_OUT = False

# Temperature is a configuration hyperparameter that controls the randomness of
# language model output.
# A high temperature produces more unpredictable and creative results, while a low
# temperature produces more common and conservative output.
# For example, if you adjust the temperature to 0.5, the model will usually generate
# text that is more predictable and less creative than if you set the temperature to
# 1.0.
PROMPT_TEMPERATURE = 0.8

prompt = ChatPromptTemplate(
    messages=[
        SystemPromptTemplate.from_template(
            PROMPT_SCENE_DEFINE + _DEFAULT_TEMPLATE,
            response_format=json.dumps(
                RESPONSE_FORMAT_SIMPLE, ensure_ascii=False, indent=4
            ),
        ),
        HumanPromptTemplate.from_template("{user_input}"),
    ]
)

prompt_adapter = AppScenePromptTemplateAdapter(
    prompt=prompt,
    template_scene=ChatScene.ExcelLearning.value(),
    stream_out=PROMPT_NEED_STREAM_OUT,
    output_parser=LearningExcelOutputParser(),
    temperature=PROMPT_TEMPERATURE,
)
CFG.prompt_template_registry.register(prompt_adapter, is_default=True)
