"""Planner Agent."""

from typing import Any, Dict, List, Optional

from dbgpt._private.pydantic import Field

from ...resource.pack import ResourcePack
from ..agent import Agent, AgentMessage
from ..base_agent import ConversableAgent
from ..plan.plan_action import PlanAction
from ..profile import DynConfig, ProfileConfig


class PlannerAgent(ConversableAgent):
    """Planner Agent.

    Planner agent, realizing task goal planning decomposition through LLM.
    """

    agents: List[ConversableAgent] = Field(default_factory=list)

    profile: ProfileConfig = ProfileConfig(
        name=DynConfig(
            "Planner",
            category="agent",
            key="dbgpt_agent_plan_planner_agent_profile_name",
        ),
        role=DynConfig(
            "Planner",
            category="agent",
            key="dbgpt_agent_plan_planner_agent_profile_role",
        ),
        goal=DynConfig(
            "Understand each of the following intelligent agents and their "
            "capabilities, using the provided resources, solve user problems by "
            "coordinating intelligent agents. Please utilize your LLM's knowledge "
            "and understanding ability to comprehend the intent and goals of the "
            "user's problem, generating a task plan that can be completed through"
            " the collaboration of intelligent agents without user assistance.",
            category="agent",
            key="dbgpt_agent_plan_planner_agent_profile_goal",
        ),
        expand_prompt=DynConfig(
            "Available Intelligent Agents:\n {{ agents }}",
            category="agent",
            key="dbgpt_agent_plan_planner_agent_profile_expand_prompt",
        ),
        constraints=DynConfig(
            [
                "Every step of the task plan should exist to advance towards solving "
                "the user's goals. Do not generate meaningless task steps; ensure "
                "that each step has a clear goal and its content is complete.",
                "Pay attention to the dependencies and logic of each step in the task "
                "plan. For the steps that are depended upon, consider the data they "
                "depend on and whether it can be obtained based on the current goal. "
                "If it cannot be obtained, please indicate in the goal that the "
                "dependent data needs to be generated.",
                "Each step must be an independently achievable goal. Ensure that the "
                "logic and information are complete. Avoid steps with unclear "
                "objectives, like 'Analyze the retrieved issues data,' where it's "
                "unclear what specific content needs to be analyzed.",
                "Please ensure that only the intelligent agents mentioned above are "
                "used, and you may use only the necessary parts of them. Allocate "
                "them to appropriate steps strictly based on their described "
                "capabilities and limitations. Each intelligent agent can be reused.",
                "Utilize the provided resources to assist in generating the plan "
                "steps according to the actual needs of the user's goals. Do not use "
                "unnecessary resources.",
                "Each step should ideally use only one type of resource to accomplish "
                "a sub-goal. If the current goal can be broken down into multiple "
                "subtasks of the same type, you can create mutually independent "
                "parallel tasks.",
                "Data resources can be loaded and utilized by the appropriate "
                "intelligent agents without the need to consider the issues related "
                "to data loading links.",
                "Try to merge continuous steps that have sequential dependencies. If "
                "the user's goal does not require splitting, you can create a "
                "single-step task with content that is the user's goal.",
                "Carefully review the plan to ensure it comprehensively covers all "
                "information involved in the user's problem and can ultimately "
                "achieve the goal. Confirm whether each step includes the necessary "
                "resource information, such as URLs, resource names, etc.",
            ],
            category="agent",
            key="dbgpt_agent_plan_planner_agent_profile_constraints",
        ),
        desc=DynConfig(
            "You are a task planning expert! You can coordinate intelligent agents"
            " and allocate resources to achieve complex task goals.",
            category="agent",
            key="dbgpt_agent_plan_planner_agent_profile_desc",
        ),
        examples=DynConfig(
            """
user:help me build a sales report summarizing our key metrics and trends
assistants:[
    {{
        "serial_number": "1",
        "agent": "DataScientist",
        "content": "Retrieve total sales, average sales, and number of transactions grouped by "product_category"'.",
        "rely": ""
    }},
    {{
        "serial_number": "2",
        "agent": "DataScientist",
        "content": "Retrieve monthly sales and transaction number trends.",
        "rely": ""
    }},
    {{
        "serial_number": "3",
        "agent": "Reporter",
        "content": "Integrate analytical data into the format required to build sales reports.",
        "rely": "1,2"
    }}
]""",  # noqa: E501
            category="agent",
            key="dbgpt_agent_plan_planner_agent_profile_examples",
        ),
    )
    _goal_zh: str = (
        "Understand each agent and its capabilities below, use the provided "
        "resources, and coordinate agents to solve the user's problem. Use your "
        "LLM knowledge and understanding to infer the user's intent and goal, "
        "then generate a task plan that agents can complete without further user help."
    )
    _expand_prompt_zh: str = "Available agents:\n {{ agents }}"

    _constraints_zh: List[str] = [
        "Every task-plan step must advance the user's goal. Do not generate "
        "meaningless steps; ensure each step has a clear and complete objective.",
        "Pay attention to dependencies and logic between steps. Dependent steps "
        "must consider whether required data can be produced by earlier goals; "
        "if not, explicitly state that the dependency data must be generated.",
        "Each step should be an independently completable goal with complete "
        "logic and information. Avoid vague goals such as 'Analyze the retrieved "
        "issues data' when the specific analysis target is unclear.",
        "Use only the agents mentioned above, and only the parts needed. Assign "
        "steps strictly according to each agent's described capabilities and "
        "limitations. Agents may be reused.",
        "Use the provided resources only when they are needed to achieve the "
        "user's goal; do not use unnecessary resources.",
        "Prefer one resource type per step. If the current goal can be split "
        "into multiple same-type subtasks, create independent parallel tasks.",
        "Data resources can be loaded and used by suitable agents; do not worry "
        "about data-loading links.",
        "Merge consecutive same-type steps with sequential dependencies when "
        "possible. If the user's goal does not need decomposition, create a "
        "single-step task using the user's goal as the step content.",
        "Carefully check the plan to ensure it covers all information in the "
        "user's problem and can ultimately achieve the goal. Confirm each step "
        "includes any required resource information, such as URLs or resource names.",
    ]
    _desc_zh: str = (
        "You are a task-planning expert who can coordinate agents and allocate "
        "resources to achieve complex goals."
    )

    def __init__(self, **kwargs):
        """Create a new PlannerAgent instance."""
        super().__init__(**kwargs)
        self._init_actions([PlanAction])

    def _init_reply_message(
        self,
        received_message: AgentMessage,
        rely_messages: Optional[List[AgentMessage]] = None,
    ) -> AgentMessage:
        reply_message = super()._init_reply_message(received_message)
        reply_message.context = {
            "agents": "\n".join([f"- {item.role}:{item.desc}" for item in self.agents]),
        }
        return reply_message

    def bind_agents(self, agents: List[ConversableAgent]) -> ConversableAgent:
        """Bind the agents to the planner agent."""
        self.agents = agents
        resources = []
        for agent in self.agents:
            if agent.resource:
                resources.append(agent.resource)
        self.resource = ResourcePack(resources)
        return self

    def prepare_act_param(
        self,
        received_message: Optional[AgentMessage],
        sender: Agent,
        rely_messages: Optional[List[AgentMessage]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Prepare the parameters for the act method."""
        return {
            "context": self.not_null_agent_context,
            "plans_memory": self.memory.plans_memory,
        }
