# -*- coding: utf-8 -*-
import os
import datetime
import requests
import json
from PIL import Image
from typing import Optional, Any, AsyncGenerator, Union, Sequence

from agentbricks.utils.grounding_utils import draw_point, encode_image
from pathlib import Path
import time  # 添加 time 模块导入
import io
from agentbricks.components.sandbox_center.sandboxes.cloud_computer_wy import (
    CloudComputer,
)
from agentbricks.components.sandbox_center.utils.utils import (
    get_image_size_from_url,
)
from agentbricks.components.sandbox_center.sandboxes.cloud_phone_wy import (
    CloudPhone,
)
import asyncio

from uuid import uuid4

# AgentScope imports

from agentscope_runtime.engine.agents.base_agent import Agent
from agentscope_runtime.engine.schemas.agent_schemas import (
    Content,
    Message,
)
from agentbricks.schemas.agent import DataContent
from agentscope_runtime.engine.schemas.context import Context
from agentbricks.agents.gui_agent_app_v2 import GuiAgent
from agentbricks.utils.logger_util import logger

from planner import PlannerError, TaskPlanner

TYPING_DELAY_MS = 12
TYPING_GROUP_SIZE = 50
HUMAN_HELP_ACTION = "human_help"
gui_agent = GuiAgent()
PLANNER_OUTPUT_FILE = os.path.join(
    os.path.dirname(__file__),
    "logs",
    "planner_output.txt",
)
MODEL_OUTPUT_FILE = os.path.join(
    os.path.dirname(__file__),
    "logs",
    "model_output.txt",
)


class E2BPCController:
    """Adapt the E2B sandbox to provide an interface similar to the cloud computer instance manager"""

    def __init__(self, sandbox):
        if sandbox is None:
            raise ValueError("E2B sandbox instance is required")
        self.sandbox = sandbox
        self.device = getattr(sandbox, "device", None)
        if self.device is None:
            raise ValueError("E2B sandbox is missing device reference")

    @staticmethod
    def _success() -> tuple[str, str]:
        return "success", ""

    async def _run(self, func, *args, **kwargs):
        return await asyncio.to_thread(func, *args, **kwargs)

    async def open_app(self, name: str):
        def _do():
            # launch_app 支持同步调用
            if hasattr(self.sandbox, "launch_app"):
                return self.sandbox.launch_app(name, ope_type="pca")
            raise NotImplementedError("launch_app not supported in E2B sandbox")

        await asyncio.to_thread(_do)
        return self._success()

    async def tap(self, x: int, y: int, count: int = 1):
        def _do():
            self.device.move_mouse(x, y)
            if count == 1:
                self.device.left_click()
            elif count == 2:
                self.device.double_click()
            else:
                for _ in range(count):
                    self.device.left_click()

        await asyncio.to_thread(_do)
        return self._success()

    async def right_tap(self, x: int, y: int, count: int = 1):
        def _do():
            self.device.move_mouse(x, y)
            for _ in range(max(1, count)):
                self.device.right_click()

        await asyncio.to_thread(_do)
        return self._success()

    async def hotkey(self, key_list: Sequence[str]):
        def _do():
            keys = list(key_list)
            if len(keys) == 1:
                self.device.press(keys[0])
            else:
                self.device.press(keys)

        await asyncio.to_thread(_do)
        return self._success()

    async def press_key(self, key: str):
        await self._run(self.device.press, key)
        return self._success()

    async def tap_type_enter(self, x: int, y: int, text: str):
        await self.tap(x, y, 1)
        await self.type_text(text)
        await self.press_key("enter")
        return self._success()

    async def type_text(self, text: str):
        await self._run(self.device.write, text, 50, 12)
        return self._success()

    async def drag(self, x1: int, y1: int, x2: int, y2: int):
        def _do():
            self.device.move_mouse(x1, y1)
            self.device.mouse_press("left")
            self.device.move_mouse(x2, y2)
            self.device.mouse_release("left")

        await asyncio.to_thread(_do)
        return self._success()

    async def replace(self, x: int, y: int, text: str):
        await self.tap(x, y, 1)
        await self.hotkey(["ctrl", "a"])
        await self.type_text(text)
        return self._success()

    async def append(self, x: int, y: int, text: str):
        await self.tap(x, y, 1)
        await self.type_text(text)
        return self._success()

    async def mouse_move(self, x: int, y: int):
        await self._run(self.device.move_mouse, x, y)
        return self._success()

    async def middle_click(self, x: int, y: int, count: int = 1):
        def _do():
            self.device.move_mouse(x, y)
            for _ in range(max(1, count)):
                self.device.middle_click()

        await asyncio.to_thread(_do)
        return self._success()

    async def type_with_clear_enter(
        self,
        text: str,
        clear: bool,
        enter: bool,
    ):
        if clear:
            await self.hotkey(["ctrl", "a"])
        if text:
            await self.type_text(text)
        if enter:
            await self.press_key("enter")
        return self._success()

    async def scroll_pos(self, x: int, y: int, pixels: int):
        def _do():
            self.device.move_mouse(x, y)
            direction = "down" if pixels >= 0 else "up"
            amount = max(1, abs(pixels) // 120 or 1)
            self.device.scroll(direction, amount)

        await asyncio.to_thread(_do)
        return self._success()

    async def scroll(self, pixels: int):
        if pixels == 0:
            return self._success()

        def _do():
            direction = "down" if pixels >= 0 else "up"
            amount = max(1, abs(pixels) // 120 or 1)
            self.device.scroll(direction, amount)

        await asyncio.to_thread(_do)
        return self._success()

    async def type_with_clear_enter_pos(
        self,
        text: str,
        x: int,
        y: int,
        clear: bool,
        enter: bool,
    ):
        await self.tap(x, y, 1)
        await self.type_with_clear_enter(text, clear, enter)
        return self._success()
# 资源池配置 - 从环境变量或默认值获取
PHONE_INSTANCE_IDS = (
    os.getenv("PHONE_INSTANCE_IDS", "").split(",")
    if os.getenv("PHONE_INSTANCE_IDS")
    else []
)
DESKTOP_IDS = (
    os.getenv("DESKTOP_IDS", "").split(",") if os.getenv("DESKTOP_IDS") else []
)


class ComputerUseAgent(Agent):
    def __init__(
        self,
        name: str = "ComputerUseAgent",
        agent_config: Optional[dict] = None,
    ):
        super().__init__(name=name, agent_config=agent_config)

        # Extract parameters from agent_config
        config = agent_config or {}
        equipment = config.get("equipment")
        # output_dir = config.get("output_dir", ".")
        mode = config.get("mode", "pc_use")
        sandbox_type = config.get("sandbox_type", "pc_wuyin")
        status_callback = config.get("status_callback")
        pc_use_add_info = config.get("pc_use_add_info", "")
        max_steps = config.get("max_steps", 20)
        chat_id = config.get("chat_id", "")
        user_id = config.get("user_id", "")
        e2e_info = config.get("e2e_info", [])
        extra_params = config.get("extra_params", "")
        state_manager = config.get("state_manager")  # 新增：获取状态管理器

        # Save initialization parameters for copy method
        self._attr = {
            "name": name,
            "agent_config": self.agent_config,
        }

        # Initialize computer use specific attributes
        self.chat_instruction = None
        self.latest_screenshot = None  # Most recent PNG of the screen
        self.image_counter = 0  # Current screenshot number
        # Store state manager for dynamic equipment access
        self.state_manager = state_manager
        self.mode = mode
        self.sandbox_type = sandbox_type

        # Equipment / sandbox references
        self._e2b_sandbox = None
        self.equipment = None

        # 初始化设备
        self._initialize_equipment(equipment)

        # Setup output directory based on chat_id and timestamp
        time_now = datetime.datetime.now()
        # 在Docker容器中使用相对路径，确保目录在容器内正确创建
        self.tmp_dir = os.path.join(
            "output",
            user_id,
            chat_id,
            time_now.strftime("%Y%m%d_%H%M%S"),
        )

        # 确保基础output目录存在
        try:
            base_output_dir = "output"
            if not os.path.exists(base_output_dir):
                os.makedirs(base_output_dir, exist_ok=True)
                print(f"Created base output directory: {base_output_dir}")
        except Exception as e:
            print(f"Warning: Failed to create base output directory: {e}")

        # Configuration
        self.status_callback = status_callback
        self.max_steps = max_steps
        self.user_id = user_id
        self.chat_id = chat_id
        self.e2e_info = e2e_info
        self.extra_params = extra_params

        # Setup sandbox reference
        if self.sandbox is None and hasattr(self, "_e2b_sandbox"):
            if self._e2b_sandbox and hasattr(self._e2b_sandbox, "device"):
                self.sandbox = self._e2b_sandbox.device

        print(f"e2e_info: {e2e_info}")

        # Mode-specific setup
        if mode == "pc_use":
            self.session_id = ""
            self.add_info = pc_use_add_info
        elif mode == "phone_use":
            if self.sandbox_type == "phone_wuyin":
                self.session_id = ""
                self.add_info = pc_use_add_info
        else:
            logger.error("Invalid mode")
            raise ValueError(
                f"Invalid mode: {mode}, must be one of: [pc_use, phone_use]",
            )

        # Control flags
        self._is_cancelled = False
        self._interrupted = False
        self._plan_forced_completion = False
        # Background wait task management
        self._wait_task = None

        planner_enabled_cfg = config.get("enable_planner")
        if planner_enabled_cfg is None:
            planner_enabled_cfg = (
                os.getenv("ENABLE_TASK_PLANNER", "true").lower()
                not in {"0", "false", "no"}
            )
        if not planner_enabled_cfg:
            logger.warning("Planner is mandatory; overriding disabled configuration.")
        self._planner_enabled = True
        self._planner_instance: Optional[TaskPlanner] = None
        self._planner_unavailable = False

    def _get_task_planner(self) -> TaskPlanner:
        if not self._planner_enabled:
            raise PlannerError("Task planner has been disabled but is required.")

        if self._planner_instance is None and not self._planner_unavailable:
            try:
                self._planner_instance = TaskPlanner.from_env()
            except PlannerError as err:
                logger.warning("Planner unavailable: %s", err)
                self._planner_unavailable = True
                raise
            except Exception as err:  # pragma: no cover - defensive
                logger.warning("Planner initialization failed: %s", err)
                self._planner_unavailable = True
                raise PlannerError(f"Planner initialization failed: {err}") from err

        if self._planner_instance is None:
            raise PlannerError("Task planner is unavailable after initialization attempts.")

        return self._planner_instance

    def _log_plan_payload(self, plan_payload: dict) -> None:
        """Log planner output to console for quick inspection."""
        task = plan_payload.get("task", "")
        reason = plan_payload.get("reason", "")
        source = plan_payload.get("source", "")
        steps = plan_payload.get("steps") or []
        logger.info(
            "Planner output | task='%s' | reason='%s' | source='%s' | step_count=%s",
            task,
            reason,
            source,
            len(steps),
        )
        for idx, step in enumerate(steps, start=1):
            title = (step.get("title") or "").strip()
            description = (step.get("description") or "").strip()
            logger.info("  Step %s: %s — %s", idx, title, description)
        self._write_plan_to_file(plan_payload)

    def _write_plan_to_file(self, plan_payload: dict) -> None:
        """Append planner output to a local text file for debugging."""
        try:
            os.makedirs(os.path.dirname(PLANNER_OUTPUT_FILE), exist_ok=True)
            raw_timestamp = plan_payload.get("timestamp") or time.time()
            try:
                timestamp = float(raw_timestamp)
            except (TypeError, ValueError):
                timestamp = time.time()
            formatted_time = datetime.datetime.fromtimestamp(timestamp).strftime(
                "%Y-%m-%d %H:%M:%S",
            )
            lines = [
                "=== Planner Output ===",
                f"time: {formatted_time}",
                f"task: {plan_payload.get('task', '')}",
                f"reason: {plan_payload.get('reason', '')}",
                f"source: {plan_payload.get('source', '')}",
                f"complexity: {plan_payload.get('complexity', '')}",
            ]
            steps = plan_payload.get("steps") or []
            for idx, step in enumerate(steps, start=1):
                title = (step.get("title") or "").strip()
                description = (step.get("description") or "").strip()
                lines.append(f"Step {idx}: {title}")
                if description and description != title:
                    lines.append(f"  {description}")
            lines.append("")
            with open(PLANNER_OUTPUT_FILE, "a", encoding="utf-8") as handle:
                handle.write("\n".join(lines) + "\n")
        except Exception as err:  # pragma: no cover - defensive
            logger.warning("Failed to write planner output file: %s", err)

    def _write_model_output(
        self,
        status: str,
        payload: dict,
        *,
        step: Optional[int] = None,
    ) -> None:
        """Persist model responses (e.g., analysis results or errors) to disk."""
        try:
            os.makedirs(os.path.dirname(MODEL_OUTPUT_FILE), exist_ok=True)
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            lines = [
                "=== Model Output ===",
                f"time: {timestamp}",
                f"status: {status}",
                f"mode: {payload.get('mode', '')}",
            ]
            if step is not None:
                lines.append(f"step: {step}")
            if "message" in payload:
                lines.append(f"message: {payload['message']}")
            raw_response = payload.get("response")
            if raw_response is not None:
                try:
                    response_text = json.dumps(
                        raw_response,
                        ensure_ascii=False,
                        indent=2,
                    )
                except TypeError:
                    response_text = str(raw_response)
                lines.append("response:")
                lines.append(response_text)
            context = payload.get("context")
            if context is not None:
                try:
                    context_text = json.dumps(
                        context,
                        ensure_ascii=False,
                        indent=2,
                    )
                except TypeError:
                    context_text = str(context)
                lines.append("context:")
                lines.append(context_text)
            lines.append("")
            with open(MODEL_OUTPUT_FILE, "a", encoding="utf-8") as handle:
                handle.write("\n".join(lines))
        except Exception as err:  # pragma: no cover - defensive
            logger.warning("Failed to write model output file: %s", err)

    def _build_fallback_plan(self, instruction: str, reason: str, source: str) -> dict:
        normalized_instruction = (instruction or "").strip() or "Execute task"
        title = normalized_instruction[:60]
        return {
            "type": "plan",
            "requires_plan": True,
            "task": instruction,
            "reason": reason,
            "complexity": "simple",
            "steps": [
                {
                    "title": title,
                    "description": normalized_instruction,
                },
            ],
            "timestamp": time.time(),
            "uuid": str(uuid4()),
            "source": source,
        }

    async def _maybe_create_plan(self, instruction: str) -> dict:
        try:
            planner = self._get_task_planner()
        except PlannerError as err:
            logger.error(
                "Planner unavailable for task '%s': %s. Falling back to single-step plan.",
                instruction,
                err,
            )
            plan_payload = self._build_fallback_plan(
                instruction,
                reason=f"Planner unavailable: {err}",
                source="task_planner_unavailable_fallback",
            )
            self._log_plan_payload(plan_payload)
            return plan_payload

        try:
            result = await asyncio.to_thread(planner.plan, instruction)
        except PlannerError as err:
            logger.error(
                "Planner error for task '%s': %s. Falling back to single-step plan.",
                instruction,
                err,
            )
            plan_payload = self._build_fallback_plan(
                instruction,
                reason=f"Planner error: {err}",
                source="task_planner_error_fallback",
            )
            self._log_plan_payload(plan_payload)
            return plan_payload
        except Exception as err:  # pragma: no cover - defensive
            logger.error(
                "Unexpected planner error for task '%s': %s. Falling back to single-step plan.",
                instruction,
                err,
            )
            plan_payload = self._build_fallback_plan(
                instruction,
                reason=f"Planner unexpected error: {err}",
                source="task_planner_exception_fallback",
            )
            self._log_plan_payload(plan_payload)
            return plan_payload

        if not result:
            logger.error(
                "Planner returned no result for task '%s'. Falling back to single-step plan.",
                instruction,
            )
            plan_payload = self._build_fallback_plan(
                instruction,
                reason="Planner returned no result.",
                source="task_planner_empty_result_fallback",
            )
            self._log_plan_payload(plan_payload)
            return plan_payload

        steps = [step.as_dict() for step in result.steps] if result.steps else []
        if not steps:
            fallback_reason = result.reason or "Planner returned no steps; generated fallback."
            logger.info(
                "Planner produced no steps for task '%s'; forcing single-step plan.",
                instruction,
            )
            plan_payload = self._build_fallback_plan(
                instruction,
                reason=fallback_reason,
                source="task_planner_forced_single_step",
            )
            self._log_plan_payload(plan_payload)
            return plan_payload

        plan_payload = {
            "type": "plan",
            "requires_plan": True,
            "task": instruction,
            "reason": result.reason,
            "complexity": result.complexity,
            "steps": steps,
            "timestamp": time.time(),
            "uuid": str(uuid4()),
            "source": "task_planner_v1",
        }
        self._log_plan_payload(plan_payload)
        return plan_payload

    def _reset_runtime_flags(self):
        """Reset runtime control flags so the next subtask can start"""
        self._is_cancelled = False
        self._interrupted = False
        self._plan_forced_completion = False
        if self._wait_task and not self._wait_task.done():
            self._wait_task.cancel()
        self._wait_task = None

    async def _run_instruction(
        self,
        instruction: str,
        intro_text: Optional[str] = None,
    ) -> AsyncGenerator[Union[Message, Content], None]:
        """Execute a single instruction using the existing workflow"""
        self._reset_runtime_flags()

        if intro_text:
            yield DataContent(
                data={
                    "step": "",
                    "stage": "start",
                    "type": "text",
                    "text": intro_text,
                    "timestamp": time.time(),
                    "uuid": str(uuid4()),
                },
            )

        if self.state_manager:
            await self.state_manager.clear_stop_signal(
                self.user_id,
                self.chat_id,
            )

        async for result in self._execute_computer_use_task(instruction):
            yield result

    async def _ensure_equipment(self):
        """Ensure the device is available; fetch it from the state manager if needed"""
        if self.sandbox_type == "e2b_desktop":
            # E2B 设备通过本地缓存获取
            if not isinstance(self.equipment, E2BPCController):
                sandbox = (
                    self.state_manager.get_e2b_sandbox(
                        self.user_id,
                        self.chat_id,
                    )
                    if self.state_manager
                    else None
                )
                if sandbox:
                    self._setup_e2b_equipment(sandbox)
                    logger.info("✅ Successfully recovered the E2B device instance")
                    return True
                await self._handle_missing_equipment()
            return self.equipment is not None

        if self.equipment is None and self.state_manager is not None:
            try:
                logger.info(
                    f"Attempting to fetch device info from the state manager, chat ID: {self.chat_id}",
                )

                equipment_info = await self.state_manager.get_equipment_info(
                    self.user_id,
                    self.chat_id,
                )

                if equipment_info:
                    self.equipment = await self._initialize_device_from_info(
                        equipment_info,
                    )
                    self._setup_sandbox_reference()
                    logger.info(
                        f"✅ Successfully rebuilt device object: {equipment_info['equipment_type']}",
                    )
                    return True
                else:
                    await self._handle_missing_equipment()

            except Exception as e:
                logger.error(f"Failed to fetch device: {str(e)}")
                raise Exception(f"Unable to acquire device: {str(e)}")

        return self.equipment is not None

    async def _initialize_device_from_info(self, equipment_info):
        """Initialize the device object using the device info"""
        equipment_type = equipment_info["equipment_type"]
        instance_info = equipment_info["instance_manager_info"]

        if equipment_type == "pc_wuyin":
            return await self._create_device(
                CloudComputer,
                instance_info["desktop_id"],
            )
        elif equipment_type == "phone_wuyin":
            return await self._create_device(
                CloudPhone,
                instance_info["instance_id"],
            )
        else:
            raise Exception(f"Unsupported device type: {equipment_type}")

    async def _create_device(self, device_class, device_id):
        """Create the device instance and automatically handle event-loop issues"""
        try:
            if device_class == CloudComputer:
                device = CloudComputer(desktop_id=device_id)
            else:  # CloudPhone
                device = CloudPhone(instance_id=device_id)
            await device.initialize()
            return device
        except RuntimeError as e:
            if "There is no current event loop" in str(
                e,
            ) or "got Future" in str(e):
                logger.warning(f"Detected event loop issue; initializing via thread pool: {e}")
                return await asyncio.to_thread(
                    lambda: self._sync_init_device(device_class, device_id),
                )
            raise
        except Exception as e:
            logger.error(f"{device_class.__name__} initialization failed: {str(e)}")
            raise Exception(f"{device_class.__name__} initialization failed: {str(e)}")

    def _sync_init_device(self, device_class, device_id):
        """Synchronously initialize the device in a new event loop"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            if device_class == CloudComputer:
                device = CloudComputer(desktop_id=device_id)
            else:
                device = CloudPhone(instance_id=device_id)
            loop.run_until_complete(device.initialize())
            return device
        finally:
            loop.close()

    def _initialize_equipment(self, equipment: Any):
        """Initialize the device reference based on the sandbox type"""
        if self.sandbox_type == "e2b_desktop":
            self._setup_e2b_equipment(equipment)
        else:
            self.equipment = equipment
            self._setup_sandbox_reference()

    def _setup_e2b_equipment(self, equipment: Any):
        """Build the E2B control adapter"""
        if equipment is None:
            self.equipment = None
            self._e2b_sandbox = None
            return

        if isinstance(equipment, E2BPCController):
            adapter = equipment
            sandbox = equipment.sandbox
        else:
            sandbox = equipment
            adapter = E2BPCController(sandbox)

        self._e2b_sandbox = sandbox
        self.equipment = adapter
        if hasattr(sandbox, "device") and sandbox.device:
            self.sandbox = sandbox.device

    def _setup_sandbox_reference(self):
        """Set sandbox reference"""
        if self.sandbox_type == "e2b_desktop":
            if self._e2b_sandbox and hasattr(self._e2b_sandbox, "device"):
                self.sandbox = self._e2b_sandbox.device
            return
        if hasattr(self.equipment, "device") and self.equipment.device:
            self.sandbox = self.equipment.device

    async def _handle_missing_equipment(self):
        """Handle missing device information"""
        chat_state = await self.state_manager.get_chat_state(
            self.user_id,
            self.chat_id,
        )
        storage_status = chat_state.get("equipment_storage_status")

        if storage_status == "e2b_desktop":
            raise Exception("E2B device not ready; reactivate the environment and retry")
        elif storage_status == "stored_in_redis":
            raise Exception("Device info expired; reinitialize the device")
        else:
            raise Exception("Device not initialized; call /cua/init to initialize it first")

    async def _check_stop_signal(self):
        """Check the stop signal in Redis (simplified to avoid event-loop conflicts)"""
        try:
            # 如果没有状态管理器，直接返回False
            if not self.state_manager:
                return False

            # 直接使用现有的状态管理器检查停止信号，而不是创建新的
            if hasattr(self.state_manager, "check_stop_signal"):
                stop_requested = await self.state_manager.check_stop_signal(
                    self.user_id,
                    self.chat_id,
                )

                if stop_requested:
                    # 清除停止信号
                    await self.state_manager.clear_stop_signal(
                        self.user_id,
                        self.chat_id,
                    )
                    # 发送停止确认状态
                    await self.state_manager.update_status(
                        self.user_id,
                        self.chat_id,
                        {
                            "status": "stopped",
                            "message": "Agent has received stop signal"
                            " and is terminating",
                            "type": "SYSTEM",
                        },
                    )
                    logger.info("Stop signal detected; setting cancellation flag")

                return stop_requested
            else:
                # 如果状态管理器不支持停止信号检查，返回False
                return False

        except Exception as e:
            logger.error(f"Error while checking stop signal: {e}")
            return False

    def copy(self) -> "ComputerUseAgent":
        """Create a copy of the agent"""
        return ComputerUseAgent(**self._attr)

    async def run_async(
        self,
        context: Context,
        **kwargs: Any,
    ) -> AsyncGenerator[Union[Message, Content], None]:
        """
        AgentScope async run method that yields streaming responses
        """
        # 确保设备可用
        try:
            await self._ensure_equipment()
        except Exception as e:
            error_msg = f"Device unavailable: {str(e)}"
            # logger.error(error_msg)
            logger.error(error_msg)
            yield DataContent(
                data={
                    "step": "",
                    "stage": "error",
                    "type": "text",
                    "text": error_msg,
                },
            )
            return

        request = context.request
        if not request or not request.input:
            error_msg = "No input found in request."
            # logger.error(error_msg)
            print(error_msg)
            yield DataContent(
                data={
                    "step": "",
                    "stage": "error",
                    "type": "text",
                    "text": f"Error: {error_msg}",
                },
            )
            return

        # Extract instruction from the first input message
        first_message = request.input[0]
        if hasattr(first_message, "content") and isinstance(
            first_message.content,
            list,
        ):
            # Find text content in the message
            instruction = None
            for content_item in first_message.content:
                if (
                    hasattr(content_item, "type")
                    and content_item.type == "text"
                ):
                    instruction = content_item.text
                    break
        else:
            instruction = (
                str(first_message.content)
                if hasattr(first_message, "content")
                else str(first_message)
            )

        if not instruction:
            error_msg = "No instruction text found in input message."
            # logger.error(error_msg)
            logger.error(error_msg)
            yield DataContent(
                data={
                    "step": "",
                    "stage": "error",
                    "type": "text",
                    "text": f"Error: {error_msg}",
                },
            )
            return

        # Update chat_id from session if available
        session_id = (
            request.session_id if hasattr(request, "session_id") else None
        )
        if session_id:
            self.chat_id = session_id

        plan_payload = await self._maybe_create_plan(instruction)
        plan_steps = []
        pending_task_summary = None
        if plan_payload:
            yield DataContent(data=plan_payload)
            plan_steps = plan_payload.get("steps") or []

        if plan_steps:
            total_steps = len(plan_steps)
            pending_task_summary = None
            completed_steps = 0
            for idx, step in enumerate(plan_steps, start=1):
                sub_instruction = (
                    (step.get("description") or "").strip()
                    or (step.get("title") or "").strip()
                )
                if not sub_instruction:
                    logger.warning(
                        "Planner step %s missing actionable text, skipping",
                        idx,
                    )
                    continue

                title = step.get("title") or f"Subtask {idx}"
                description = step.get("description") or ""
                intro_text = f"🧭 Subtask {idx}/{total_steps}: {title}"
                if description:
                    intro_text += f" — {description}"

                pending_task_summary = None
                async for message in self._run_instruction(
                    sub_instruction,
                    intro_text=intro_text,
                ):
                    if getattr(message, 'data', {}).get('stage') == 'all_completed':
                        pending_task_summary = message
                        continue
                    yield message

                plan_forced_completion = self._plan_forced_completion
                if plan_forced_completion:
                    self._plan_forced_completion = False
                    if idx < total_steps:
                        # Early stop on intermediate step; treat as normal completion and continue.
                        self._is_cancelled = False
                        plan_forced_completion = False

                if self._is_cancelled:
                    if plan_forced_completion:
                        completed_steps += 1
                        yield DataContent(
                            data={
                                "type": "text",
                                "stage": "plan_step_completed",
                                "text": f"✅ Subtask {idx}/{total_steps} completed",
                                "timestamp": time.time(),
                                "uuid": str(uuid4()),
                            },
                        )
                        break
                    logger.info(
                        "Execution cancelled during planner step %s/%s",
                        idx,
                        total_steps,
                    )
                    break

                completed_steps += 1
                yield DataContent(
                    data={
                        "type": "text",
                        "stage": "plan_step_completed",
                        "text": f"✅ Subtask {idx}/{total_steps} completed",
                        "timestamp": time.time(),
                        "uuid": str(uuid4()),
                    },
                )
            if completed_steps >= total_steps:
                yield DataContent(
                    data={
                        "type": "text",
                        "stage": "plan_completed",
                        "text": "🎯 All subtasks completed.",
                        "timestamp": time.time(),
                        "uuid": str(uuid4()),
                    },
                )
            else:
                yield DataContent(
                    data={
                        "type": "text",
                        "stage": "plan_cancelled",
                        "text": "⏹️ Task ended early; remaining subtasks were skipped.",
                        "timestamp": time.time(),
                        "uuid": str(uuid4()),
                    },
                )
            if pending_task_summary:
                yield pending_task_summary
            return

        async for message in self._run_instruction(
            instruction,
            intro_text=f"🤖 Starting task: {instruction}",
        ):
            yield message

    async def _execute_computer_use_task(
        self,
        instruction: str,
    ) -> AsyncGenerator[Union[Message, Content], None]:
        """
        Execute computer use task with streaming responses
        """
        logger.info("Running task...")
        try:
            # cjj疑似改
            while not self._is_cancelled:
                self.chat_instruction = instruction
                logger.info(f"USER: {instruction}")
                if self.mode in ["pc_use", "phone_use"]:
                    self.session_id = ""

                should_continue = True
                step_count = 0
                while should_continue and step_count < self.max_steps:
                    # 检查取消标志
                    if self._is_cancelled:
                        break

                    # 检查Redis中的停止信号
                    if self.state_manager and await self._check_stop_signal():
                        logger.info("Stop signal received from Redis, terminating task")
                        self._is_cancelled = True
                        break

                    step_count += 1

                    # Yield step start message
                    yield DataContent(
                        data={
                            "step": f"{step_count}",
                            "stage": "output",
                            "type": "text",
                            "text": f"🔄 Step {step_count}",
                        },
                    )
                    step_info = {
                        "step": step_count,
                        "auxiliary_info": {},
                        "observation": "",
                        "action_parsed": "",
                        "action_executed": "",
                        "timestamp": time.time(),
                        "uuid": str(uuid4()),
                    }
                    # 添加设备ID信息
                    equipment_id = "Unknown"
                    equipment_controller = None
                    if self.sandbox_type == "e2b_desktop":
                        equipment_id = "e2b_desktop"
                        equipment_controller = self.equipment
                    elif hasattr(self.equipment, "instance_manager"):
                        if hasattr(
                            self.equipment.instance_manager,
                            "desktop_id",
                        ):
                            equipment_id = (
                                self.equipment.instance_manager.desktop_id
                            )
                        elif hasattr(
                            self.equipment.instance_manager,
                            "instance_id",
                        ):
                            equipment_id = (
                                self.equipment.instance_manager.instance_id
                            )
                        equipment_controller = self.equipment.instance_manager
                    else:
                        equipment_controller = self.equipment
                    step_info["equipment_id"] = equipment_id

                    # 发送推理开始状态
                    step_info["analyzing"] = True
                    logger.info("Starting reasoning")
                    time_s_agent = time.time()

                    try:
                        # Yield analysis start message
                        yield DataContent(
                            data={
                                "step": f"{step_count}",
                                "stage": "output",
                                "type": "text",
                                "text": "🔍 Analyzing screenshot",
                            },
                        )

                        # Process analyse_screenshot as async generator
                        screenshot_analysis = None
                        auxiliary_info = None
                        mode_response = None

                        async for data_content in self.analyse_screenshot(
                            step_count,
                        ):
                            # Yield status updates
                            if (
                                data_content.data.get("type")
                                == "analysis_result"
                            ):
                                # Extract final results
                                screenshot_analysis = data_content.data.get(
                                    "text",
                                )
                                auxiliary_info = data_content.data.get(
                                    "auxiliary_info",
                                )
                                mode_response = data_content.data.get(
                                    "mode_response",
                                )
                                yield DataContent(
                                    data={
                                        "step": f"{step_count}",
                                        "stage": "draw",
                                        "type": "analysis_result",
                                        "text": f"{screenshot_analysis}",
                                        "auxiliary_info": auxiliary_info,
                                    },
                                )
                            else:
                                yield data_content

                    except Exception as analyse_error:
                        error_msg = f"Analysis failed: {str(analyse_error)}"
                        logger.error(error_msg)
                        yield DataContent(
                            data={
                                "step": f"{step_count}",
                                "stage": "error",
                                "type": "text",
                                "text": f"Error: {error_msg}",
                            },
                        )
                        raise analyse_error

                    logger.info(
                        "screenshot analysis "
                        f"cost time{time.time() - time_s_agent}",
                    )

                    # 推理完成，更新完整信息
                    step_info["observation"] = screenshot_analysis
                    step_info["analyzing"] = False
                    if auxiliary_info:
                        step_info["auxiliary_info"].update(auxiliary_info)

                    # Yield action execution message
                    yield DataContent(
                        data={
                            "step": f"{step_count}",
                            "stage": "output",
                            "type": "text",
                            "text": "⚡ Executing action",
                        },
                    )

                    if self.status_callback:
                        self.emit_status("STEP", step_info)

                    # 使用try-catch包围设备操作，防止异步问题
                    try:
                        if self.mode == "pc_use":
                            action_result = await self._execute_pc_action(
                                mode_response,
                                equipment_controller,
                                step_count,
                            )
                            # 处理人工干预
                            if action_result.get("human_intervention"):
                                yield DataContent(
                                    data=action_result["human_intervention"],
                                )

                                # 如果需要等待人工干预，开始等待
                                if action_result["result"] == "wait_for_human":
                                    wait_time = action_result.get(
                                        "wait_time",
                                        60,
                                    )
                                    self._wait_task = asyncio.create_task(
                                        self._do_wait_for_human_help(
                                            wait_time,
                                        ),
                                    )

                                    try:
                                        await self._wait_task
                                    except asyncio.CancelledError:
                                        logger.info(
                                            "PC wait task was cancelled",
                                        )

                            if action_result["result"] == "stop":
                                should_continue = False
                                yield DataContent(
                                    data={
                                        "step": f"{step_count}",
                                        "stage": "completed",
                                        "type": "text",
                                        "text": "Step completed!",
                                    },
                                )
                                self._plan_forced_completion = True
                                self._is_cancelled = True
                            if "Answer" in action_result["result"]:
                                should_continue = False
                                yield DataContent(
                                    data={
                                        "step": f"{step_count}",
                                        "stage": "completed",
                                        "type": "text",
                                        "text": action_result["result"],
                                    },
                                )
                                self._plan_forced_completion = True
                                self._is_cancelled = True
                        elif self.mode == "phone_use":
                            action_result = await self._execute_phone_action(
                                mode_response,
                                equipment_controller,
                                auxiliary_info,
                                step_count,
                            )
                            # 处理人工干预
                            if action_result.get("human_intervention"):
                                yield DataContent(
                                    data=action_result["human_intervention"],
                                )

                                # 如果需要等待人工干预，开始等待
                                if action_result["result"] == "wait_for_human":
                                    wait_time = action_result.get(
                                        "wait_time",
                                        60,
                                    )
                                    self._wait_task = asyncio.create_task(
                                        self._do_wait_for_human_help(
                                            wait_time,
                                        ),
                                    )

                                    try:
                                        await self._wait_task
                                    except asyncio.CancelledError:
                                        print(
                                            "Phone wait task was cancelled by "
                                            "user intervention",
                                        )

                            if action_result["result"] == "stop":
                                should_continue = False
                                yield DataContent(
                                    data={
                                        "step": f"{step_count}",
                                        "stage": "completed",
                                        "type": "text",
                                        "text": "Step completed!",
                                    },
                                )
                                self._plan_forced_completion = True
                                self._is_cancelled = True

                    except Exception as action_error:
                        error_msg = f"Error while executing action: {str(action_error)}"
                        logger.error(error_msg)
                        yield DataContent(
                            data={
                                "step": f"{step_count}",
                                "stage": "error",
                                "type": "text",
                                "text": f"{error_msg}",
                            },
                        )
                        continue

                if not should_continue:
                    yield DataContent(
                        data={
                            "step": "",
                            "stage": "all_completed",
                            "type": "text",
                            "text": f"Task complete! Executed {step_count} steps",
                        },
                    )
                    break
                elif step_count >= self.max_steps:
                    yield DataContent(
                        data={
                            "step": "",
                            "stage": "limit_completed",
                            "type": "text",
                            "text": f"Reached maximum step limit ({self.max_steps}); task stopped",
                        },
                    )
                    break
                elif self._is_cancelled:
                    logger.info("✅ Task canceled")
                    yield DataContent(
                        data={
                            "step": "",
                            "stage": "canceled",
                            "type": "text",
                            "text": "⏹️ Task cancelled",
                        },
                    )
                    break

        except Exception as e:
            error_msg = str(e)
            # 检查是否为GUI service request failed的错误
            if (
                "Error querying" in error_msg
                and "GUI service request failed" in error_msg
            ):
                # 尝试提取请求ID
                import re

                request_id_match = re.search(
                    r'"request_id":"([^"]+)"',
                    error_msg,
                )
                if request_id_match:
                    request_id = request_id_match.group(1)
                    formatted_error = (
                        f"Internal agent invocation failed, request ID: {request_id}"
                    )
                else:
                    formatted_error = "Internal agent invocation failed"
            else:
                formatted_error = f"Error while executing task: {error_msg}"

            logger.error(f"Error while executing task: {error_msg}")
            yield DataContent(
                data={
                    "step": "",
                    "stage": "error",
                    "type": "text",
                    "text": formatted_error,
                },
            )
        finally:
            self.stop(notify=False)
            # 异步删除临时文件夹
            if os.path.exists(self.tmp_dir):
                import shutil

                async def cleanup_temp_dir():
                    try:
                        await asyncio.to_thread(shutil.rmtree, self.tmp_dir)
                    except Exception as e:
                        logger.info(
                            f"Failed to delete {self.tmp_dir}. Reason: {e}",
                        )

                # 在后台执行清理，不阻塞任务结束
                asyncio.create_task(cleanup_temp_dir())

            logger.info("Agent run loop exited.")

    def stop(
        self,
        *,
        notify: bool = True,
        status: str = "running",
        message: str = "Stop request received, waiting for current step to complete...",
    ):
        print("Agent stopped by user request.")
        self._is_cancelled = True
        if notify:
            self.emit_status(
                "SYSTEM",
                {
                    "message": message,
                    "status": status,
                },
            )

    def interrupt_wait(self):
        """
        由前端调用，用于中断当前的等待状态
        """
        self._interrupted = True
        # 取消后台等待任务
        if self._wait_task and not self._wait_task.done():
            self._wait_task.cancel()
            logger.info("Background wait task cancelled")
        logger.info("Agent wait stopped by user request.")
        # 发送状态更新到前端
        self.emit_status(
            "SYSTEM",
            {
                "message": "Stop wait request received,"
                " waiting for current step to complete...",
                "status": "running",
            },
        )

    def emit_status(self, status_type: str, data: dict):
        """Emit status update"""
        status_data = {
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "type": status_type,
            "status": data.get("status", "running"),
            "data": data,
        }

        logger.info(
            f"emit_status - chat_id: {self.chat_id}, type: {status_type}",
        )

        if self.status_callback:
            try:
                if asyncio.iscoroutinefunction(self.status_callback):
                    self._run_async_callback(self.chat_id, status_data)
                else:
                    result = self.status_callback(self.chat_id, status_data)
                    logger.info(f"Synchronous callback result: {result}")
            except Exception as e:
                logger.error(f"Error in status callback: {e}")
        else:
            logger.warning("No status callback available")

    async def annotate_image(
        self,
        point: list,
        anno_type: str,
        is_save: bool = False,
    ):
        """Asynchronous image annotation"""

        def _create_annotation():
            from PIL import Image, ImageDraw

            if anno_type == "point":
                annotated_img = draw_point(
                    Image.open(self.latest_screenshot),
                    point,
                )
                return annotated_img
            elif anno_type == "box":
                coor = point
                # 四个点的坐标
                points = [
                    (coor[0], coor[1]),
                    (coor[0], coor[3]),
                    (coor[2], coor[3]),
                    (coor[2], coor[1]),
                ]
                image = Image.open(self.latest_screenshot)
                draw = ImageDraw.Draw(image)
                # 绘制四边形
                draw.polygon(
                    points,
                    outline="red",
                    fill=None,
                    width=3,
                )
                return image  # 返回图像对象，不是draw对象

            elif anno_type == "arrow":
                import math

                image = Image.open(self.latest_screenshot)
                draw = ImageDraw.Draw(image)
                [x1, y1, x2, y2] = point
                arrow_size = 10
                color = "red"
                draw.line((x1, y1, x2, y2), fill=color, width=2)

                # 计算箭头的方向向量
                angle = math.atan2(y2 - y1, x2 - x1)

                # 计算箭头的两个顶点
                arrow_x1 = x2 - arrow_size * math.cos(angle - math.pi / 6)
                arrow_y1 = y2 - arrow_size * math.sin(angle - math.pi / 6)
                arrow_x2 = x2 - arrow_size * math.cos(angle + math.pi / 6)
                arrow_y2 = y2 - arrow_size * math.sin(angle + math.pi / 6)

                # 绘制箭头（三角形）
                draw.polygon(
                    [(x2, y2), (arrow_x1, arrow_y1), (arrow_x2, arrow_y2)],
                    fill=color,
                )
                return image  # 返回图像对象，不是draw对象

            # 如果anno_type不匹配，返回原始图像
            return Image.open(self.latest_screenshot)

        annotated_img = await asyncio.to_thread(_create_annotation)

        screenshot_filename = os.path.basename(self.latest_screenshot)
        p = Path(screenshot_filename)  # 保留前面的目录
        oss_screenshot_filename = f"{p.stem}_{uuid4().hex}{p.suffix}"

        img_path = None
        oss_url = None

        if is_save:
            try:
                img_path = await self.save_image(
                    annotated_img,
                    f"{screenshot_filename[:-4]}_annotated",
                )
                logger.info(f"[DEBUG] Annotated image saved to: {img_path}")
            except Exception as e:
                logger.error(f"Failed to save annotated image: {e}")

        # 只有在成功保存图片时才上传
        if img_path:
            try:
                # 异步上传到oss
                async def _upload_to_oss():
                    return await self.equipment.upload_file_and_sign(
                        img_path,
                        oss_screenshot_filename,
                    )

                oss_url = await _upload_to_oss()
                logger.info(
                    f"[DEBUG] Annotated image uploaded to OSS: {oss_url}",
                )
            except Exception as e:
                logger.info(
                    f"Failed to upload annotated image to OSS: {e}",
                )
                oss_url = None
        else:
            logger.info("[DEBUG] No image path, skipping OSS upload")

        return encode_image(annotated_img), oss_url

    def _run_async_callback(self, chat_id, status_data):
        """Simplified async callback handling"""
        if not self.status_callback:
            return

        try:
            import asyncio

            try:
                loop = asyncio.get_running_loop()
                task = loop.create_task(
                    self.status_callback(chat_id, status_data),
                )
                logger.info(f"Created async task for status callback: {task}")
            except RuntimeError:
                logger.warning(
                    "No running event loop, calling callback synchronously",
                )
                try:
                    asyncio.run(self.status_callback(chat_id, status_data))
                except Exception as sync_error:
                    logger.error(
                        f"Sync callback execution failed: {sync_error}",
                    )
                    try:
                        self.status_callback(chat_id, status_data)
                    except Exception as fallback_error:
                        logger.error(
                            f"Fallback callback also failed: {fallback_error}",
                        )
        except Exception as e:
            logger.error(f"Error running async callback: {e}")

    async def save_image(self, image, prefix="image"):
        """Async screenshot saving"""

        def _save_sync():
            if not os.path.exists(self.tmp_dir):
                os.makedirs(self.tmp_dir)

            # 使用随机数命名文件，避免重复
            random_suffix = uuid4().hex[:8]
            filename = f"{prefix}_{random_suffix}.png"
            filepath = os.path.join(self.tmp_dir, filename)

            if isinstance(image, Image.Image):
                image.save(filepath)
            else:
                with open(filepath, "wb") as f:
                    f.write(image)
            return filepath

        return await asyncio.to_thread(_save_sync)

    async def take_screenshot(self, prefix="screenshot"):
        """Unified screenshot method that chooses the proper approach for each device type"""
        if self.sandbox_type == "pc_wuyin":
            return await self._screenshot_pc(prefix)
        elif self.sandbox_type == "phone_wuyin":
            return await self._screenshot_phone(prefix)
        else:
            return await self._screenshot_default(prefix)

    async def _screenshot_pc(self, prefix):
        """Desktop screenshot handling"""
        try:
            filepath, filename = self._prepare_screenshot_path(prefix)
            await self.equipment.get_screenshot_base64_save_local(
                filename.replace(".png", ""),
                filepath,
            )

            oss_url = await self._upload_to_oss(filepath)
            self.latest_screenshot = filepath

            image_data = await self._read_image_file(filepath)
            return image_data, oss_url, filename
        except Exception as e:
            raise Exception(f"Desktop screenshot failed: {str(e)}")

    async def _screenshot_phone(self, prefix):
        """Phone screenshot handling"""
        try:
            filepath, filename = self._prepare_screenshot_path(prefix)
            oss_url = await self.equipment.get_screenshot_oss_phone()

            # Download image to local disk
            await self._download_image(oss_url, filepath)

            # 重新上传到OSS获取新URL
            new_oss_url = await self._upload_to_oss(filepath)
            self.latest_screenshot = filepath

            image_data = await self._read_image_file(filepath)
            return image_data, new_oss_url, filename
        except Exception as e:
            raise Exception(f"Phone screenshot failed: {str(e)}")

    async def _screenshot_default(self, prefix):
        """Fallback screenshot handling (E2B, etc.)"""
        raw_bytes = await asyncio.to_thread(self.sandbox.screenshot)
        image, compressed_bytes = await asyncio.to_thread(
            self._prepare_e2b_image,
            raw_bytes,
        )
        filename = await self.save_image(image, prefix)
        self.latest_screenshot = filename

        image_data = await self._read_image_file(filename)
        base64_data = encode_image(compressed_bytes)
        oss_url = f"data:image/jpeg;base64,{base64_data}"
        return image_data, oss_url, os.path.basename(filename)

    def _prepare_e2b_image(self, raw_bytes: bytes):
        """Compress E2B screenshots, limit size, and return the processed image plus bytes"""
        with Image.open(io.BytesIO(raw_bytes)) as img:
            image = img.convert("RGB")

        max_width = int(os.getenv("E2B_SCREENSHOT_MAX_WIDTH", "1024"))
        max_height = int(os.getenv("E2B_SCREENSHOT_MAX_HEIGHT", "768"))
        jpeg_quality = int(os.getenv("E2B_SCREENSHOT_JPEG_QUALITY", "70"))
        image.thumbnail((max_width, max_height), Image.LANCZOS)

        buffer = io.BytesIO()
        image.save(
            buffer,
            format="JPEG",
            quality=jpeg_quality,
            optimize=True,
            progressive=True,
        )
        compressed_bytes = buffer.getvalue()

        return image, compressed_bytes

    def _prepare_screenshot_path(self, prefix):
        """Prepare screenshot file path"""
        os.makedirs(self.tmp_dir, exist_ok=True)
        random_suffix = uuid4().hex[:8]
        filename = f"{prefix}_{random_suffix}.png"
        filepath = os.path.join(self.tmp_dir, filename)
        return filepath, filename

    async def _upload_to_oss(self, filepath):
        """Upload file to OSS"""
        p = Path(filepath)
        oss_filepath = f"{p.stem}_{uuid4().hex}{p.suffix}"
        return await self.equipment.upload_file_and_sign(
            filepath,
            oss_filepath,
        )

    async def _download_image(self, url, filepath):
        """Download image to local disk"""

        def download():
            response = requests.get(url, stream=True)
            if response.status_code == 200:
                with open(filepath, "wb") as f:
                    for chunk in response.iter_content(1024):
                        f.write(chunk)
            else:
                raise Exception(f"Failed to download image from {url}")

        await asyncio.to_thread(download)

    async def _read_image_file(self, filepath):
        """Read image file"""

        def read_file():
            with open(filepath, "rb") as f:
                return f.read()

        return await asyncio.to_thread(read_file)

    # def use_upload_local_file_oss(self, file_path, file_name):
    #     with open(file_path, "rb") as file:
    #         return self.equipment.upload_local_file_oss(file, file_name)

    def _handle_action_error(self, error, action_type="action"):
        """Generic handler for action execution errors"""
        error_msg = f"Error in {action_type}: {str(error)}"
        logger.error(error_msg)
        import traceback

        traceback.print_exc()
        return {"result": "error", "error": str(error)}

    async def _handle_human_intervention(self, task, step_count):
        """Generic handler for human intervention"""
        try:
            human_intervention_info = await self._wait_for_human_help(
                task,
                step_count,
            )
            return {
                "result": "wait_for_human",
                "human_intervention": human_intervention_info,
                "wait_time": int(os.getenv("HUMAN_WAIT_TIME", "60")),
            }
        except Exception as e:
            logger.error(f"Human intervention handling failed: {str(e)}")
            error_intervention_info = {
                "step": f"{step_count}" if step_count else "",
                "stage": "human_help",
                "type": "human_intervention",
                "text": f"Human intervention handling failed: {str(e)}",
                "task_description": task if "task" in locals() else "Unknown task",
                "wait_time": 0,
                "timestamp": time.time(),
                "uuid": str(uuid4()),
            }
            return {
                "result": "continue",
                "human_intervention": error_intervention_info,
            }

    async def _wait_for_human_help(self, task, step_count=None):
        """
        异步等待人类帮助完成任务的通用方法
        立即返回人工干预信息，然后开始等待
        """
        time_to_sleep = int(os.getenv("HUMAN_WAIT_TIME", "60"))  # 转换为整数
        logger.info(
            "HUMAN_HELP: The system will wait "
            f"for {time_to_sleep} "
            f"seconds for human to do the task: {task}",
        )

        # 立即返回人工干预信息，让前端马上显示
        human_intervention_info = {
            "step": f"{step_count}" if step_count else "",
            "stage": "human_help",
            "type": "human_intervention",
            "text": f"Human assistance required: {task}",
            "task_description": task,
            "wait_time": time_to_sleep,
            "timestamp": time.time(),
            "uuid": str(uuid4()),
        }

        return human_intervention_info

    async def _do_wait_for_human_help(self, time_to_sleep):
        """
        实际执行等待逻辑的方法 - 简化版本，避免事件循环问题
        """
        # 重置中断标志
        self._interrupted = False
        start_time = time.time()
        sleep_interval = min(5, time_to_sleep)  # 每次最多等待5秒

        # 简化的等待循环，减少复杂性
        while (
            (time.time() - start_time) < time_to_sleep
            and not self._interrupted
            and not self._is_cancelled
        ):
            await asyncio.sleep(sleep_interval)

            # 简单检查停止信号，不创建新的连接
            try:
                if self.state_manager and hasattr(
                    self.state_manager,
                    "check_stop_signal",
                ):
                    if await self.state_manager.check_stop_signal(
                        self.user_id,
                        self.chat_id,
                    ):
                        logger.info("Stop signal received while waiting for human intervention")
                        self._is_cancelled = True
                        break
            except Exception as e:
                logger.error(f"Error while checking stop signal: {e}")
                # 继续等待，不因为这个错误而终止

        waited_time = time.time() - start_time

        if self._interrupted:
            logger.info("Human help wait was interrupted by user.")
            self._interrupted = False  # 重置标志
        elif self._is_cancelled:
            logger.info("Human help wait was cancelled.")
        else:
            logger.info(
                f"Human help wait completed after {waited_time:.1f}s",
            )

    async def analyse_screenshot(self, step_count: int = None):
        auxiliary_info = {}
        try:
            # 发送截图阶段状态
            yield DataContent(
                data={
                    "step": f"{step_count}",
                    "stage": "screenshot",
                    "type": "analysis_stage",
                    "text": "capturing",
                    "timestamp": time.time(),
                    "uuid": str(uuid4()),
                },
            )

            if self.sandbox_type == "pc_wuyin":
                time_s = time.time()
                logger.info(
                    f"Running analyse_screenshot_instance_id:"
                    f" {self.equipment.instance_manager.desktop_id}",
                )
                screenshot_img, screenshot_oss, screenshot_filename = (
                    await self.take_screenshot("screenshot")
                )
                logger.info(f"screenshot cost time{time.time() - time_s}")
            elif self.sandbox_type == "phone_wuyin":
                time_s = time.time()
                logger.info(
                    f"Running analyse_screenshot_android_instance_name:"
                    f" {self.equipment.instance_manager.instance_id}",
                )
                screenshot_img, screenshot_oss, screenshot_filename = (
                    await self.take_screenshot("screenshot")
                )
                logger.info(f"screenshot cost time{time.time() - time_s}")
                width, height = await get_image_size_from_url(
                    screenshot_oss,
                )
                auxiliary_info["width"] = width
                auxiliary_info["height"] = height
            else:
                screenshot_img, screenshot_oss, screenshot_filename = (
                    await self.take_screenshot("screenshot")
                )

            # 发送AI分析阶段状态
            yield DataContent(
                data={
                    "step": f"{step_count}",
                    "stage": "ai_analysis",
                    "type": "analysis_stage",
                    "text": "analyzing",
                    "timestamp": time.time(),
                    "uuid": str(uuid4()),
                },
            )
        except Exception as e:
            yield DataContent(
                data={
                    "step": f"{step_count}",
                    "stage": "error",
                    "type": "SYSTEM",
                    "text": "Error taking screenshot: %s" % e,
                },
            )
            logger.error(f"Error taking screenshot: {e}")
            return

        if self.mode == "pc_use":
            try:
                # app v2
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "data",
                                "data": {
                                    "messages": [
                                        {"image": screenshot_oss},
                                        {"instruction": self.chat_instruction},
                                        {"add_info": self.add_info},
                                        {"session_id": self.session_id},
                                        {"a11y": []},
                                        {"use_a11y": 0},
                                        {"use_reflection": False},
                                        {"task_is_complex": False},
                                        {"thought_language": "english"},
                                    ],
                                },
                            },
                        ],
                    },
                ]
                model_name = "pre-gui_owl_7b"
                if isinstance(self.e2e_info, list):
                    messages[0]["content"][0]["data"]["messages"].extend(
                        self.e2e_info,
                    )
                    for item in self.e2e_info:
                        if isinstance(item, dict) and "model_name" in item:
                            model_name = item["model_name"]
                            break
                elif (
                    self.e2e_info
                ):  # 如果e2e_info存在但不是列表，将其作为单个元素添加
                    messages[0]["content"][0]["data"]["messages"].append(
                        self.e2e_info,
                    )
                    model_name = self.e2e_info["model_name"]

                # 添加新的param_list字典
                param_dict = {
                    "param_list": [
                        {"add_info": self.add_info},
                        {"a11y": ""},
                        {"use_a11y": -1},
                        {"enable_reflector": True},
                        {"enable_notetaker": True},
                        {"worker_model": model_name},
                        {"manager_model": model_name},
                        {"reflector_model": model_name},
                        {"notetaker_model": model_name},
                    ],
                }

                # 将param_dict添加到messages中
                messages[0]["content"][0]["data"]["messages"].append(
                    param_dict,
                )
                mode_response = await gui_agent.run(messages, "pc_use")
                logger.info(f"PC model response: {mode_response}")
                # 发送图像处理阶段状态
                yield DataContent(
                    data={
                        "step": f"{step_count}",
                        "stage": "image_processing",
                        "type": "analysis_stage",
                        "message": "processing",
                        "timestamp": time.time(),
                        "uuid": str(uuid4()),
                    },
                )

                # 添加短暂延迟，确保前端能够处理image_processing状态
                await asyncio.sleep(0.2)

                action = mode_response.get("action", "")
                action_params = mode_response.get("action_params", {})

                self.session_id = mode_response.get("session_id", "")
                auxiliary_info["request_id"] = mode_response.get(
                    "request_id",
                    "",
                )
                auxiliary_info["session_id"] = mode_response.get(
                    "session_id",
                    "",
                )

                # 为click类型的动作生成标注图片
                if "position" in action_params:
                    try:
                        point_x = action_params["position"][0]
                        point_y = action_params["position"][1]
                        _, img_path = await self.annotate_image(
                            [point_x, point_y],
                            anno_type="point",
                            is_save=True,
                        )
                        auxiliary_info["annotated_img_path"] = img_path
                    except Exception as e:
                        logger.error(
                            f"Error generating annotated image: {e}",
                        )
                elif (
                    "position1" in action_params
                    and "position2" in action_params
                ):
                    _, img_path = await self.annotate_image(
                        [
                            action_params["position1"][0],
                            action_params["position1"][1],
                            action_params["position2"][0],
                            action_params["position2"][1],
                        ],
                        anno_type="box",
                        is_save=True,
                    )
                    auxiliary_info["annotated_img_path"] = img_path
                else:
                    # by cjj 所有操作都保留截图
                    auxiliary_info[
                        "annotated_img_path"
                    ] = f"data:image/jpeg;base64,{screenshot_oss}"

                result_data = {
                    "thought": mode_response.get("thought", ""),
                    "action": action,
                    # "action_params": action_params,
                    "explanation": mode_response.get("explanation", ""),
                    "annotated_img_path": auxiliary_info.get(
                        "annotated_img_path",
                        "",
                    ),
                }
                self._write_model_output(
                    "success",
                    {
                        "mode": self.mode,
                        "response": mode_response,
                        "context": {
                            "thought": result_data["thought"],
                            "explanation": result_data["explanation"],
                            "action": action,
                            "action_params": action_params,
                        },
                    },
                    step=step_count,
                )
                result = json.dumps(result_data, ensure_ascii=False)

            except Exception as e:
                self._write_model_output(
                    "error",
                    {
                        "mode": self.mode,
                        "message": f"Error querying PC use model: {e}",
                    },
                    step=step_count,
                )
                logger.error(f"Error querying PC use model: {e}")
                raise RuntimeError(f"Error querying PC use model: {e}")
        elif self.mode == "phone_use":
            try:
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "data",
                                "data": {
                                    "messages": [
                                        {"image": screenshot_oss},
                                        {"instruction": self.chat_instruction},
                                        {"add_info": self.add_info},
                                        {"session_id": self.session_id},
                                        {"thought_language": "english"},
                                    ],
                                },
                            },
                        ],
                    },
                ]
                if isinstance(self.e2e_info, list):
                    messages[0]["content"][0]["data"]["messages"].extend(
                        self.e2e_info,
                    )
                elif (
                    self.e2e_info
                ):  # 如果e2e_info存在但不是列表，将其作为单个元素添加
                    messages[0]["content"][0]["data"]["messages"].append(
                        self.e2e_info,
                    )

                # 添加新的param_list字典
                param_dict = {
                    "param_list": [
                        {"add_info": self.add_info},
                    ],
                }

                # 将param_dict添加到messages中
                messages[0]["content"][0]["data"]["messages"].append(
                    param_dict,
                )

                mode_response = await gui_agent.run(messages, "phone_use")
                action = mode_response.get("operation", "")
                logger.info(f"Phone model response: {mode_response} - {action}")
                # 发送图像处理阶段状态
                yield DataContent(
                    data={
                        "step": f"{step_count}",
                        "stage": "image_processing",
                        "type": "analysis_stage",
                        "message": "processing",
                        "timestamp": time.time(),
                        "uuid": str(uuid4()),
                    },
                )

                # 添加短暂延迟，确保前端能够处理image_processing状态
                await asyncio.sleep(0.2)

                if "click" in action.lower():
                    # 为click类型的动作生成标注图片
                    try:
                        print(f"Received action: {action}")
                        coordinate = (
                            action.split("(")[-1].split(")")[0].split(",")
                        )
                        x1, y1, x2, y2 = (
                            int(coordinate[0]),
                            int(coordinate[1]),
                            int(coordinate[2]),
                            int(
                                coordinate[3],
                            ),
                        )
                        x, y = int((x1 + x2) / 2), int((y1 + y2) / 2)
                        # 从auxiliary_info获取屏幕尺寸
                        width = auxiliary_info.get("width", 1080)
                        height = auxiliary_info.get("height", 1920)
                        point_x = int(x / 1000 * width)
                        point_y = int(y / 1000 * height)
                        _, img_path = await self.annotate_image(
                            [point_x, point_y],
                            anno_type="point",
                            is_save=True,
                        )
                        auxiliary_info["annotated_img_path"] = img_path

                    except Exception as e:
                        logger.info(
                            f"Error generating annotated image: {e}",
                        )
                else:
                    action = action
                    # 所有动作都要保存图片，但是只有要标记的才oss by cjj
                    auxiliary_info["annotated_img_path"] = screenshot_oss

                result_data = {
                    "thought": mode_response.get("thought", ""),
                    "action": action,
                    "explanation": mode_response.get("explanation", ""),
                    "annotated_img_path": auxiliary_info.get(
                        "annotated_img_path",
                        "",
                    ),
                }
                # 如果action包含括号，需要拆分
                if (
                    action
                    and isinstance(action, str)
                    and "(" in action
                    and ")" in action
                ):
                    # 提取括号前的部分作为action
                    action_part = action.split("(", 1)[0].strip()
                    # 提取括号及内部内容作为action_params

                    result_data["action"] = action_part
                    # result_data["action_params"] = params_part
                self._write_model_output(
                    "success",
                    {
                        "mode": self.mode,
                        "response": mode_response,
                        "context": {
                            "thought": result_data.get("thought", ""),
                            "explanation": result_data.get("explanation", ""),
                            "action": result_data.get("action", ""),
                        },
                    },
                    step=step_count,
                )
                result = json.dumps(result_data, ensure_ascii=False)
                self.session_id = mode_response.get("session_id", "")
                auxiliary_info["request_id"] = mode_response.get(
                    "request_id",
                    "",
                )
                auxiliary_info["session_id"] = mode_response.get(
                    "session_id",
                    "",
                )
            except Exception as e:
                self._write_model_output(
                    "error",
                    {
                        "mode": self.mode,
                        "message": f"Error querying Phone use model: {e}",
                    },
                    step=step_count,
                )
                yield DataContent(
                    data={
                        "step": f"{step_count}",
                        "stage": "error",
                        "type": "SYSTEM",
                        "text": "Error querying Phone use model %s" % e,
                    },
                )
                logger.error(f"Error querying Phone use model: {e}")

                # 发送分析阶段失败状态，确保前端不会卡在AI分析阶段
                yield DataContent(
                    data={
                        "step": f"{step_count}",
                        "stage": "error",
                        "type": "analysis_stage",
                        "text": "Analysis failed",
                        "timestamp": time.time(),
                        "uuid": str(uuid4()),
                    },
                )
                logger.error(f"Error querying Phone use model: {e}")
                raise RuntimeError(f"Error querying Phone use model: {e}")
        else:
            logger.error(
                f"Invalid mode: {self.mode},"
                "must be one of: pc_use，phone_use",
            )
            raise ValueError(
                f"Invalid mode: {self.mode},"
                "must be one of: pc_use，phone_use",
            )

        # 发送完成状态之前添加短暂延迟，确保前端能够处理image_processing状态
        await asyncio.sleep(0.1)

        # 发送完成状态
        yield DataContent(
            data={
                "step": f"{step_count}",
                "stage": "completed",
                "type": "analysis_stage",
                "text": "completed",
                "timestamp": time.time(),
                "uuid": str(uuid4()),
            },
        )
        # Yield final result
        yield DataContent(
            data={
                "step": f"{step_count}",
                "stage": "completed",
                "type": "analysis_result",
                "text": result,
                "auxiliary_info": auxiliary_info,
                "mode_response": mode_response,
            },
        )

    async def _execute_pc_action(
        self,
        mode_response,
        equipment,
        step_count=None,
    ):
        """Execute PC actions based on mode response"""
        try:
            logger.info("Executing PC action")
            action_type = mode_response.get("action", "")
            action_parameter = mode_response.get("action_params", {})
            if action_type == "stop":
                print("stop")
                return {"result": "stop"}
            elif action_type == "open app":
                name = action_parameter["name"]
                if name == "File Explorer":
                    name = "File Explorer"
                await equipment.open_app(name)
            elif action_type == "wait":
                wait_time = action_parameter.get("time", 5)
                await asyncio.sleep(wait_time)
            elif action_type == "click":
                x = action_parameter["position"][0]
                y = action_parameter["position"][1]
                count = action_parameter["count"]
                await equipment.tap(x, y, count=count)
            elif action_type == "right click":
                x = action_parameter["position"][0]
                y = action_parameter["position"][1]
                count = action_parameter["count"]
                await equipment.right_tap(x, y, count=count)
            elif action_type == "hotkey":
                keylist = action_parameter["key_list"]
                await equipment.hotkey(keylist)
            elif action_type == "presskey":
                key = action_parameter["key"]
                await equipment.press_key(key)
            elif action_type == "click_type":
                x = action_parameter["position"][0]
                y = action_parameter["position"][1]
                text = action_parameter["text"]
                await equipment.tap_type_enter(x, y, text)
            elif action_type == "drag":
                x1 = action_parameter["position1"][0]
                y1 = action_parameter["position1"][1]
                x2 = action_parameter["position2"][0]
                y2 = action_parameter["position2"][1]
                await equipment.drag(x1, y1, x2, y2)
            elif action_type == "replace":
                x = action_parameter["position"][0]
                y = action_parameter["position"][1]
                text = action_parameter["text"]
                await equipment.replace(x, y, text)
            elif action_type == "append":
                x = action_parameter["position"][0]
                y = action_parameter["position"][1]
                text = action_parameter["text"]
                await equipment.append(x, y, text)
            elif action_type == "tell":
                answer_dict = action_parameter["answer"]
                print(answer_dict)
            elif action_type == "mouse_move":
                x = action_parameter["position"][0]
                y = action_parameter["position"][1]
                await equipment.mouse_move(x, y)
            elif action_type == "middle_click":
                x = action_parameter["position"][0]
                y = action_parameter["position"][1]
                await equipment.middle_click(x, y)
            elif action_type == "type_with_clear_enter":
                clear = action_parameter["clear"]
                enter = action_parameter["enter"]
                text = action_parameter["text"]
                await equipment.type_with_clear_enter(text, clear, enter)
            elif action_type == "call_user":
                task = mode_response.get("explanation")
                return await self._handle_human_intervention(task, step_count)
            elif action_type == "scroll":
                if "position" in action_parameter:  # -E
                    x = action_parameter["position"][0]
                    y = action_parameter["position"][1]
                    pixels = action_parameter["pixels"]
                    await equipment.scroll_pos(x, y, pixels)
                else:  # e2e
                    pixels = action_parameter["pixels"]
                    await equipment.scroll(pixels)
            elif action_type == "type_with_clear_enter_pos":  # New
                clear = action_parameter["clear"]
                enter = action_parameter["enter"]
                text = action_parameter["text"]
                x = action_parameter["position"][0]
                y = action_parameter["position"][1]
                await equipment.type_with_clear_enter_pos(
                    text,
                    x,
                    y,
                    clear,
                    enter,
                )
            else:
                logger.warning(f"Unknown action_type '{action_type}'")
                print(f"Warning: Unknown action_type '{action_type}'")

            return {"result": "continue"}

        except Exception as e:
            return self._handle_action_error(e, "_execute_pc_action")

    async def _execute_phone_action(
        self,
        mode_response,
        equipment,
        auxiliary_info,
        step_count=None,
    ):
        """Execute phone actions based on mode response"""
        try:
            action = mode_response.get("operation")
            operation_str_list = action.split("$")
            screen_size = 1
            width, height = auxiliary_info["width"], auxiliary_info["height"]
            for id_, operation in enumerate(operation_str_list):
                if "Select" in operation:
                    task = mode_response.get("explanation")
                    return await self._handle_human_intervention(
                        task,
                        step_count,
                    )
                elif "Click" in operation:
                    coordinate = (
                        operation.split("(")[-1].split(")")[0].split(",")
                    )
                    x1, y1, x2, y2 = (
                        int(coordinate[0]),
                        int(coordinate[1]),
                        int(coordinate[2]),
                        int(coordinate[3]),
                    )
                    await equipment.tab__(x1, y1, x2, y2, width, height)
                elif "Swipe down" in operation:
                    x1, y1 = int(width * screen_size / 2), int(
                        height * screen_size / 3,
                    )
                    x2, y2 = int(width * screen_size / 2), int(
                        2 * height * screen_size / 3,
                    )
                    await equipment.slide(x1, y1, x2, y2)
                elif "Swipe up" in operation:
                    x1, y1 = int(width * screen_size / 2), int(
                        2 * height * screen_size / 3,
                    )
                    x2, y2 = int(width * screen_size / 2), int(
                        height * screen_size / 3,
                    )
                    await equipment.slide(x1, y1, x2, y2)
                elif "Swipe" in operation:
                    coordinate = (
                        operation.split("(")[-1].split(")")[0].split(",")
                    )
                    x1, y1, x2, y2 = (
                        int(int(coordinate[0]) / 1000 * width),
                        int(int(coordinate[1]) / 1000 * height),
                        int(int(coordinate[2]) / 1000 * width),
                        int(int(coordinate[3]) / 1000 * height),
                    )
                    a_x1, a_x2 = int(x1 * screen_size + x2 * 0), int(
                        x1 * 0 + x2 * screen_size,
                    )
                    a_y1, a_y2 = int(y1 * screen_size + y2 * 0), int(
                        y1 * 0 + y2 * screen_size,
                    )
                    await equipment.slide(a_x1, a_y1, a_x2, a_y2)
                elif "Type" in operation:
                    parameter = operation.split("(")[-1].split(")")[0]
                    await equipment.type(parameter)
                elif "Back" in operation:
                    await equipment.back()
                elif "Home" in operation:
                    await equipment.home()
                elif "Done" in operation:
                    return {"result": "stop"}
                elif "Answer" in operation:
                    return {"result": operation}
                elif "Wait" in operation:
                    task = mode_response.get("explanation")
                    return await self._handle_human_intervention(
                        task,
                        step_count,
                    )
                else:
                    print(f"Warning: Unknown phone operation '{operation}'")

            return {"result": "continue"}

        except Exception as e:
            return self._handle_action_error(e, "_execute_phone_action")
