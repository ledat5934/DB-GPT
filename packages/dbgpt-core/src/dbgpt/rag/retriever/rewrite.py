"""Query rewrite."""

from typing import List, Optional

from dbgpt.core import LLMClient, ModelMessage, ModelMessageRoleType, ModelRequest
from dbgpt.core.awel.flow import Parameter, ResourceCategory, register_resource
from dbgpt.util.i18n_utils import _

REWRITE_PROMPT_TEMPLATE_EN = """
Based on the given context {context}, Generate {nums} search queries related to:
{original_query}, Provide following comma-separated format: 'queries: <queries>'":
    "original query:{original_query}\n"
    "queries:"
"""

REWRITE_PROMPT_TEMPLATE_ZH = REWRITE_PROMPT_TEMPLATE_EN


@register_resource(
    _("Query Rewrite"),
    "query_rewrite",
    category=ResourceCategory.RAG,
    description=_("Query rewrite."),
    parameters=[
        Parameter.build_from(
            _("Model Name"),
            "model_name",
            str,
            description=_("The LLM model name."),
        ),
        Parameter.build_from(
            _("LLM Client"),
            "llm_client",
            LLMClient,
            description=_("The llm client."),
        ),
        Parameter.build_from(
            _("Language"),
            "language",
            str,
            description=_("The language of the query rewrite prompt."),
            optional=True,
            default="en",
        ),
    ],
)
class QueryRewrite:
    """Query rewrite.

    query reinforce, include query rewrite, query correct
    """

    def __init__(
        self,
        model_name: str,
        llm_client: LLMClient,
        language: Optional[str] = "en",
    ) -> None:
        """Create QueryRewrite with model_name, llm_client, language.

        Args:
            model_name(str): model name
            llm_client(LLMClient, optional): llm client
            language(str, optional): language
        """
        self._model_name = model_name
        self._llm_client = llm_client
        self._language = language
        self._prompt_template = (
            REWRITE_PROMPT_TEMPLATE_EN
            if language == "en"
            else REWRITE_PROMPT_TEMPLATE_ZH
        )

    async def rewrite(
        self, origin_query: str, context: Optional[str], nums: Optional[int] = 1
    ) -> List[str]:
        """Query rewrite.

        Args:
            origin_query: str original query
            context: Optional[str] context
            nums: Optional[int] rewrite nums

        Returns:
            queries: List[str]
        """
        from dbgpt.util.chat_util import run_async_tasks

        prompt = self._prompt_template.format(
            context=context, original_query=origin_query, nums=nums
        )
        messages = [ModelMessage(role=ModelMessageRoleType.SYSTEM, content=prompt)]
        request = ModelRequest(model=self._model_name, messages=messages)
        tasks = [self._llm_client.generate(request)]
        queries = await run_async_tasks(tasks=tasks, concurrency_limit=1)
        queries = [model_out.text for model_out in queries]
        queries = list(
            filter(
                lambda content: "LLMServer Generate Error" not in content,
                queries,
            )
        )
        if len(queries) == 0:
            print("llm generate no rewrite queries.")
            return queries
        new_queries = self._parse_llm_output(output=queries[0])[0:nums]
        print(f"rewrite queries: {new_queries}")
        return new_queries

    def correct(self) -> List[str] | None:
        """Query correct."""
        pass

    def _parse_llm_output(self, output: str) -> List[str]:
        """Parse llm output.

        Args:
            output: str

        Returns:
            output: List[str]
        """
        lowercase = True
        try:
            results = []
            response = output.strip()

            if response.startswith("queries:"):
                response = response[len("queries:") :]
            if response.startswith("queries："):
                response = response[len("queries：") :]

            queries = response.split(",")
            if len(queries) == 1:
                queries = response.split("，")
            if len(queries) == 1:
                queries = response.split("?")
            if len(queries) == 1:
                queries = response.split("？")
            for k in queries:
                if k.startswith("queries:"):
                    k = k[len("queries:") :]
                if k.startswith("queries："):
                    k = response[len("queries：") :]
                rk = k
                if lowercase:
                    rk = rk.lower()
                s = rk.strip()
                if s == "":
                    continue
                results.append(s)
        except Exception as e:
            print(f"parse query rewrite prompt_response error: {e}")
            return []
        return results
