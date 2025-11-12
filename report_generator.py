"""
Utility helpers to build human readable task reports after an agent run.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from agentbricks.utils.logger_util import logger
from api_client import ActionAPIClient


def _format_timestamp(ts: Optional[float]) -> str:
    """Convert timestamp (seconds) to human friendly string."""
    if not ts:
        return "-"
    dt = _dt.datetime.fromtimestamp(ts)
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def _sanitize_filename(value: str) -> str:
    """Return safe string for filenames."""
    safe = re.sub(r"[^\w.-]", "_", value, flags=re.ASCII)
    return safe or "report"


def _extract_instruction(events: List[Dict[str, Any]]) -> str:
    for event in events:
        payload = event.get("payload", {})
        stage = payload.get("stage")
        text = payload.get("text") or payload.get("message")
        if stage == "start" and text:
            return text.replace("🤖 开始执行任务:", "").strip()
    return ""


def _format_step(payload: Dict[str, Any]) -> Optional[str]:
    stage = payload.get("stage")
    message = payload.get("message")
    text = payload.get("text")
    if not any([stage, message, text]):
        return None

    parts = []
    if stage:
        parts.append(f"[{stage}]")
    if payload.get("type") == "analysis_result":
        try:
            result = json.loads(text)
            action = result.get("action")
            explanation = result.get("explanation")
            if action:
                parts.append(f"动作: {action}")
            if explanation:
                parts.append(f"说明: {explanation}")
            return " ".join(parts).strip()
        except Exception:
            pass

    content = message or text
    if content:
        parts.append(str(content).strip())
    return " ".join(parts).strip() if parts else None


def _extract_final(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if payload.get("type") != "analysis_result":
        return None
    text = payload.get("text")
    if not text:
        return None
    try:
        parsed = json.loads(text)
        return {
            "thought": parsed.get("thought", ""),
            "action": parsed.get("action", ""),
            "explanation": parsed.get("explanation", ""),
            "annotated_img_path": parsed.get("annotated_img_path", ""),
        }
    except Exception:
        # 无法解析成JSON时，至少返回原始文本
        return {"raw": text}


def _generate_llm_markdown(payload: Dict[str, Any]) -> Optional[str]:
    api_key = (
        os.getenv("DASHSCOPE_API_KEY")
        or os.getenv("OPENAI_API_KEY")
    )
    if not api_key:
        logger.warning("跳过 LLM 报告生成：未找到 API Key")
        return None

    base_url = os.getenv(
        "ACTION_API_BASE_URL",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    try:
        client = ActionAPIClient(api_key=api_key, base_url=base_url)
    except Exception as init_error:
        logger.error(f"初始化 LLM 客户端失败: {init_error}")
        return None

    system_prompt = (
        "You are a system operation log analyst. Convert the agent’s logs into a clear, step-by-step operation "
        "report for non-technical users.\n"
        "Requirements:\n"
        "1. Start with a title: “### Operation Record: [Task Name] ([Date])”.\n"
        "2. Each action must be a numbered description prefixed with “**Step X:**”.\n"
        "3. Summaries should be natural English describing what was observed, why the system acted, and how it tried "
        "to perform the task.\n"
        "4. Replace technical details—like JSON fields or raw coordinates—with intuitive descriptions (e.g., “clicked "
        "the Start menu,” “located the Edge icon on the taskbar”).\n"
        "5. End with “**Result:**” summarizing the outcome.\n"
        "6. Keep the tone professional, concise, and easy to visualize."
    )

    user_prompt = (
        "You will receive the raw log content below. Please transform it into the human-readable document described "
        "in the requirements:\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )

    try:
        response = client.chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=1500,
        )
        content = response.choices[0].message.content
        return content.strip() if content else None
    except Exception as llm_error:
        logger.error(f"调用 LLM 生成报告失败: {llm_error}")
        return None


async def generate_human_report(
    state_manager,
    user_id: str,
    chat_id: str,
    task_id: Optional[str],
    sandbox_type: Optional[str],
) -> Optional[str]:
    """
    Build a Markdown report for the latest agent run.

    Returns the file path if report generated, otherwise None.
    """
    try:
        events = await state_manager.get_stream_seq(
            user_id,
            chat_id,
            from_sequence=1,
            task_id=task_id,
        )
    except Exception as exc:  # pragma: no cover - defensive
        logger.error(f"读取流式数据失败，无法生成报告: {exc}")
        return None

    if not events:
        logger.info("没有可用的流式事件，跳过报告生成")
        return None

    processed: List[Dict[str, Any]] = []
    for entry in events:
        raw = entry.get("data", {}) or {}
        payload = raw.get("data") if isinstance(raw, dict) else {}
        processed.append(
            {
                "timestamp": entry.get("timestamp"),
                "sequence_number": entry.get("sequence_number"),
                "payload": payload if isinstance(payload, dict) else {},
            },
        )

    start_time = processed[0]["timestamp"]
    end_time = processed[-1]["timestamp"]
    duration = "-"
    if start_time and end_time:
        duration = f"{max(end_time - start_time, 0):.1f} 秒"

    instruction = _extract_instruction(processed)
    final_info: Dict[str, Any] = {}
    steps: List[str] = []
    events_for_llm: List[Dict[str, Any]] = []

    for event in processed:
        payload = event["payload"]
        events_for_llm.append(
            {
                "timestamp": _format_timestamp(event.get("timestamp")),
                "sequence_number": event.get("sequence_number"),
                "stage": payload.get("stage"),
                "type": payload.get("type"),
                "text": payload.get("text"),
                "message": payload.get("message"),
            },
        )
        final_hit = _extract_final(payload)
        if final_hit:
            final_info = final_hit
        step_line = _format_step(payload)
        if step_line:
            ts_label = _format_timestamp(event["timestamp"])
            steps.append(f"- {ts_label} {step_line}")

    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    filename = (
        f"{_sanitize_filename(chat_id)}_"
        f"{_sanitize_filename(task_id or 'task')}_"
        f"{_dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    )
    report_path = reports_dir / filename

    limit_reached = any(
        evt["payload"].get("stage") == "limit_completed" for evt in processed
    )
    error_messages = [
        evt["payload"].get("text", "")
        for evt in processed
        if evt["payload"].get("stage") == "error"
    ]

    overview = (
        f"`{chat_id}` 会话的 Computer Use Agent 执行记录"
        f"（沙盒: `{sandbox_type or '⚠️ 待确认'}`）"
    )

    def _build_sentence(event: Dict[str, Any]) -> str:
        payload = event.get("payload", {})
        stage = payload.get("stage") or "⚠️ 待确认"
        text = payload.get("text") or payload.get("message") or "⚠️ 无描述"
        timestamp = _format_timestamp(event.get("timestamp"))

        guidance_map = {
            "start": "先保证 `/cua/init` 已成功完成，再提交具体指令。",
            "output": "这是代理正在执行操作的反馈，如需复现，可保持相同输入重新调用 `/cua/run`。",
            "analysis_stage": "模型处于分析阶段，耐心等待，不要重复提交请求。",
            "screenshot": "系统正在抓取屏幕快照，可稍后在报告或日志中查看结果。",
            "ai_analysis": "AI 正解析截图，如长时间无输出，请检查心跳是否仍在发送。",
            "image_processing": "截图标注进行中，保持网络和 Redis 连接稳定。",
            "draw": "已生成标注图，可从 `annotated_img_path` 下载查看。",
            "completed": "该步骤完成，若要继续自动化，可追加下一条自然语言指令。",
            "limit_completed": "触发步数上限，重新调用时可提升 `config.max_steps`。",
            "error": "步骤执行失败，需根据错误信息检查环境或重新初始化。",
        }
        guidance = guidance_map.get(
            stage,
            "如需复现该阶段，请使用相同参数再次调用并观察返回的流式事件。",
        )
        return (
            f"{timestamp} —— 阶段 `{stage}`：{text}。"
            f"操作建议：{guidance}"
        )

    narrative_steps = [
        _build_sentence(evt)
        for evt in processed
        if evt.get("payload")
        and evt["payload"].get("stage")
        and evt["payload"].get("stage") not in {"analysis_stage", "heartbeat"}
    ]
    if not narrative_steps:
        narrative_steps = ["⚠️ 待确认：报告未捕获到可描述的步骤，请检查日志。"]

    final_summary = "⚠️ 待确认：任务结束状态未捕获"
    if final_info:
        final_summary_parts = []
        if final_info.get("thought"):
            final_summary_parts.append(f"思考：{final_info['thought']}")
        if final_info.get("action"):
            final_summary_parts.append(f"动作：{final_info['action']}")
        if final_info.get("explanation"):
            final_summary_parts.append(f"说明：{final_info['explanation']}")
        if final_info.get("annotated_img_path"):
            final_summary_parts.append(
                f"截图：{final_info['annotated_img_path']}"
            )
        if final_summary_parts:
            final_summary = "；".join(final_summary_parts)
        elif final_info.get("raw"):
            final_summary = final_info["raw"]
    elif limit_reached:
        final_summary = "达到最大步数限制，任务自动停止。"

    common_issues = []
    max_steps_notice = (
        "默认 `config.max_steps` 为 20，达到该值后会收到“limit_completed”并终止。"
    )
    common_issues.append(f"- {max_steps_notice}")
    common_issues.append(
        "- `/cua/run` 若缺少 `input` 将返回 HTTP 400，需提供至少一条消息。"
    )
    common_issues.append(
        "- `/cua/init` 需要完整的 `user_id` 与 `chat_id`，否则会返回 HTTP 400。"
    )
    if limit_reached and max_steps_notice not in common_issues[0]:
        common_issues.append(
            "- 已触发步数上限，请根据需要调整 `config.max_steps`。"
        )
    for err in error_messages:
        common_issues.append(f"- 任务出现错误：{err}")

    sample_user = user_id or "demo_user"
    sample_chat = chat_id or "demo_chat"
    sample_sandbox = sandbox_type or "e2b_desktop"
    sample_commands = [
        "curl http://127.0.0.1:8002/cua/init \\",
        "  -H 'Content-Type: application/json' \\",
        f"  -d '{{\"config\":{{\"mode\":\"pc_use\",\"sandbox_type\":\"{sample_sandbox}\",\"user_id\":\"{sample_user}\",\"chat_id\":\"{sample_chat}\"}},\"user_id\":\"{sample_user}\"}}'",
        "",
        f"curl 'http://127.0.0.1:8002/cua/operation_status?user_id={sample_user}&chat_id={sample_chat}'",
        "",
        "curl -N http://127.0.0.1:8002/cua/run \\",
        "  -H 'Content-Type: application/json' \\",
        f"  -d '{{\"input\":[{{\"role\":\"user\",\"content\":[{{\"type\":\"text\",\"text\":\"请打开 Edge 浏览器并查询阿里巴巴股价\"}}]}}],\"config\":{{\"mode\":\"pc_use\",\"sandbox_type\":\"{sample_sandbox}\",\"user_id\":\"{sample_user}\",\"chat_id\":\"{sample_chat}\"}}}}'",
        "",
        f"curl -o report.md 'http://127.0.0.1:8002/cua/report?user_id={sample_user}&chat_id={sample_chat}'",
    ]

    llm_payload = {
        "user_id": user_id,
        "chat_id": chat_id,
        "task_id": task_id,
        "sandbox_type": sandbox_type,
        "duration": duration,
        "instruction": instruction or "⚠️ 待确认",
        "overview": overview,
        "final_summary": final_summary,
        "limit_reached": limit_reached,
        "errors": error_messages,
        "events": events_for_llm,
        "sample_commands": sample_commands,
    }

    llm_markdown = _generate_llm_markdown(llm_payload)
    if llm_markdown:
        try:
            report_path.write_text(llm_markdown, encoding="utf-8")
            logger.info(f"已生成任务报告: {report_path}")
            return str(report_path)
        except OSError as exc:
            logger.error(f"写入报告文件失败: {exc}")
            return None

    lines = [
        "# 项目/文件概览（一句话）",
        "",
        overview,
        "",
        "## 关键功能点与调用流程图",
        "",
        "1. 通过 `/cua/init` 初始化环境并绑定用户/会话。",
        "2. 调用 `/cua/run` 推送指令，代理按以下顺序反馈执行细节：",
    ]
    for idx, sentence in enumerate(narrative_steps[:10], start=1):
        lines.append(f"   {idx}. {sentence}")
    lines.append(f"3. 当前任务总结：{final_summary}")
    lines.append("")
    lines.append("## 核心 API")
    lines.append("")
    lines.extend(
        [
            "### `/cua/init`",
            "",
            "- 用途：启动环境初始化任务。",
            "- 关键入参：",
            "",
            "| 字段 | 类型 | 说明 |",
            "|------|------|------|",
            "| `config.chat_id` | `str` | 会话标识 |",
            "| `config.user_id` | `str` | 用户标识 |",
            "| `config.mode` | `str` | 支持 `pc_use` / `phone_use` |",
            "| `config.sandbox_type` | `str` | 如 `e2b_desktop`、`pc_wuyin` |",
            "",
            "- 返回：包含 `operation_id` 的 JSON。",
            "- 异常：缺少 `user_id` 或 `chat_id` 会返回 HTTP 400。",
            "",
            "### `/cua/run`",
            "",
            "- 用途：执行指令并流式返回进度。",
            "- 关键入参：",
            "",
            "| 字段 | 类型 | 说明 |",
            "|------|------|------|",
            "| `input` | `List[Message]` | 消息数组，至少一条 |",
            "| `config` | `AgentConfig` | 包含 `mode`、`sandbox_type`、`max_steps` 等 |",
            "| `sequence_number` | `int` | 可选，用于断线续传 |",
            "",
            "- 返回：SSE 流（`StreamingResponse`）。",
            "- 异常：无输入、会话不合法会返回 HTTP 400/500。",
            "",
            "### `/cua/report`",
            "",
            "- 用途：下载最新生成的 Markdown 报告。",
            "- 入参：`user_id`、`chat_id`。",
            "- 返回：`text/markdown` 文件；若报告缺失返回 404。",
        ],
    )
    lines.append("")
    lines.append("## 常见问题与坑")
    lines.append("")
    lines.extend(common_issues[: max(3, len(common_issues))])
    lines.append("")
    lines.append("## 最小可运行示例（可直接复制）")
    lines.append("")
    lines.extend(
        [
            "```bash",
            "# 1. 初始化环境",
            f"curl http://127.0.0.1:8002/cua/init \\",
            "  -H 'Content-Type: application/json' \\",
            f"  -d '{{\"config\":{{\"mode\":\"pc_use\",\"sandbox_type\":\"{sample_sandbox}\",\"user_id\":\"{sample_user}\",\"chat_id\":\"{sample_chat}\"}},\"user_id\":\"{sample_user}\"}}'",
            "",
            "# 2. 轮询状态",
            f"curl 'http://127.0.0.1:8002/cua/operation_status?user_id={sample_user}&chat_id={sample_chat}'",
            "",
            "# 3. 发送指令（将文本替换为你的需求）",
            "curl -N http://127.0.0.1:8002/cua/run \\",
            "  -H 'Content-Type: application/json' \\",
            f"  -d '{{\"input\":[{{\"role\":\"user\",\"content\":[{{\"type\":\"text\",\"text\":\"请打开 Edge 浏览器\"}}]}}],\"config\":{{\"mode\":\"pc_use\",\"sandbox_type\":\"{sample_sandbox}\",\"user_id\":\"{sample_user}\",\"chat_id\":\"{sample_chat}\"}}}}'",
            "",
            "# 4. 下载报告",
            f"curl -o report.md 'http://127.0.0.1:8002/cua/report?user_id={sample_user}&chat_id={sample_chat}'",
            "```",
        ],
    )

    try:
        report_path.write_text("\n".join(lines), encoding="utf-8")
    except OSError as exc:  # pragma: no cover - defensive
        logger.error(f"写入报告文件失败: {exc}")
        return None

    logger.info(f"已生成任务报告: {report_path}")
    return str(report_path)
