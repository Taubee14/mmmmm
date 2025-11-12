import logging
import os
from dataclasses import dataclass
from typing import List, Optional

from api_client import ActionAPIClient

logger = logging.getLogger(__name__)

DEFAULT_MODEL = (
    os.getenv("PLANNER_MODEL_NAME")
    or os.getenv("DASHSCOPE_MODEL_NAME")
    or "qwen-plus"
)
DEFAULT_TEMPERATURE = float(os.getenv("PLANNER_TEMPERATURE", "0.1"))

PLANNER_SYSTEM_PROMPT = """\
You are Task Planner, an operations expert who decides whether a user's request needs a multi-step plan.
General principles:
- Produce a plan only when it unlocks clear execution steps or reduces ambiguity.
- Skip planning when the task is a single short action, a trivial data lookup, or can be answered directly.
- Exactly 3-6 concise steps when planning is needed; each step must move the user closer to completion.
- Use the user's language when drafting step titles and descriptions.
- When the request lacks required details, include a step to gather missing information before proceeding.

Return valid JSON with this schema:
{
  "requires_plan": true | false,
  "complexity": "simple" | "moderate" | "complex",
  "reason": "<why you chose to plan or not>",
  "steps": [
    {
      "title": "<imperative, <= 60 characters>",
      "description": "<one or two sentences summarizing the step>"
    }
  ]
}

Rules:
- The steps array MUST be empty when requires_plan is false.
- The steps array MUST contain exactly two items when requires_plan is true.
- Do not invent technical prerequisites that are not implied by the task.
- Never mention that you are an AI model; focus on the work plan.
- Keep descriptions between 8 and 35 words, highlight concrete deliverables or checkpoints.

Examples:
Input: "Run jobs on Katana"
Output: requires_plan true; steps may include "Create job folder", "Prepare submission script", "Submit job via qsub", "Monitor queue status".

Input: "Open the calculator app"
Output: requires_plan false (single direct action).

Input: "Draft an outreach email to ACME about feature X"
Output: requires_plan true; first step should gather missing details if they are absent.
"""

PROMPT_TEMPLATE = """\
# Task
{task}
{context_block}

# Instructions
Determine whether the task needs decomposition. If it does, propose exactly 2 high-level steps following the JSON schema.\
"""


@dataclass
class PlannerStep:
    title: str
    description: str

    def as_dict(self) -> dict:
        return {
            "title": self.title,
            "description": self.description,
        }


@dataclass
class PlannerResult:
    requires_plan: bool
    reason: str
    steps: List[PlannerStep]
    complexity: str
    raw: dict


class PlannerError(Exception):
    pass


class TaskPlanner:
    def __init__(
        self,
        client: ActionAPIClient,
        *,
        model: str = DEFAULT_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
        system_prompt: str = PLANNER_SYSTEM_PROMPT,
    ) -> None:
        self._client = client
        self._model = model
        self._temperature = temperature
        self._system_prompt = system_prompt

    @classmethod
    def from_env(cls) -> "TaskPlanner":
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise PlannerError("DASHSCOPE_API_KEY is not configured for planner")

        base_url = os.getenv("DASHSCOPE_BASE_URL")
        client = ActionAPIClient(api_key=api_key, base_url=base_url)
        return cls(client=client)

    def plan(
        self,
        task: str,
        *,
        context: Optional[str] = None,
    ) -> PlannerResult:
        if not task or not task.strip():
            raise PlannerError("Task text is required for planning")

        prompt = self._build_prompt(task, context)

        try:
            response = self._client.generate_actions(
                prompt=prompt,
                system_prompt=self._system_prompt,
                temperature=self._temperature,
                max_tokens=800,
                model=self._model,
            )
        except Exception as exc:  # pragma: no cover - network call
            logger.warning("Planner call failed: %s", exc)
            raise PlannerError(f"Planner call failed: {exc}") from exc

        if not isinstance(response, dict):
            raise PlannerError("Planner response is not a JSON object")

        requires_plan_flag = bool(response.get("requires_plan"))
        steps_payload = response.get("steps") or []
        reason = str(response.get("reason") or "").strip()
        complexity = str(response.get("complexity") or "moderate").lower()

        steps: List[PlannerStep] = []
        if isinstance(steps_payload, list):
            for item in steps_payload:
                if not isinstance(item, dict):
                    continue
                title = str(item.get("title") or "").strip()
                description = str(item.get("description") or "").strip()

                if not title and not description:
                    continue

                if not title:
                    title = description[:60] or "Step"
                if not description:
                    description = title

                steps.append(PlannerStep(title=title, description=description))

        if len(steps) > 2:
            steps = steps[:2]

        if not requires_plan_flag:
            steps = []

        has_steps = len(steps) > 0
        requires_plan = requires_plan_flag or has_steps

        if requires_plan and not has_steps:
            raise PlannerError("Planner marked requires_plan but returned no steps")

        if requires_plan and len(steps) != 2:
            raise PlannerError("Planner returned an invalid number of steps; expected exactly 2")

        return PlannerResult(
            requires_plan=requires_plan,
            reason=reason,
            steps=steps,
            complexity=complexity if complexity in {"simple", "moderate", "complex"} else "moderate",
            raw=response,
        )

    def _build_prompt(self, task: str, context: Optional[str]) -> str:
        context_block = ""
        if context:
            context_block = f"\n# Additional Context\n{context.strip()}"

        return PROMPT_TEMPLATE.format(
            task=task.strip(),
            context_block=context_block,
        )
