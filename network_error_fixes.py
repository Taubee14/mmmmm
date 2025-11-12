# -*- coding: utf-8 -*-
"""
网络错误修复方案
解决Computer Use Agent后端流式处理中的网络错误问题
"""

import asyncio
import time
import json
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class StreamResponseManager:
    """流式响应管理器，解决网络连接问题"""

    def __init__(self, heartbeat_interval: int = 30, max_idle_time: int = 300):
        self.heartbeat_interval = heartbeat_interval  # 心跳间隔（秒）
        self.max_idle_time = max_idle_time  # 最大空闲时间（秒）
        self.last_data_time = time.time()
        self.last_heartbeat_time = time.time()

    async def send_heartbeat(self):
        """发送心跳数据"""
        heartbeat_data = {
            "object": "heartbeat",
            "type": "heartbeat",
            "timestamp": time.time(),
            "status": "alive",
        }
        return f"data: {json.dumps(heartbeat_data, ensure_ascii=False)}\n\n"

    def should_send_heartbeat(self) -> bool:
        """检查是否需要发送心跳"""
        current_time = time.time()
        return (
            current_time - self.last_heartbeat_time
        ) >= self.heartbeat_interval

    def update_heartbeat_time(self):
        """更新心跳时间"""
        self.last_heartbeat_time = time.time()

    def update_data_time(self):
        """更新数据发送时间"""
        self.last_data_time = time.time()

    def is_connection_stale(self) -> bool:
        """检查连接是否过期"""
        current_time = time.time()
        return (current_time - self.last_data_time) > self.max_idle_time


async def safe_serialize(obj: Any, timeout: float = 5.0) -> Dict:
    """安全的序列化，带超时保护"""
    try:
        # 使用asyncio.wait_for来限制序列化时间
        return await asyncio.wait_for(
            asyncio.to_thread(_serialize_sync, obj),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        logger.warning(f"序列化超时，对象类型: {type(obj)}")
        return {
            "error": "序列化超时",
            "object_type": str(type(obj)),
            "timestamp": time.time(),
        }
    except Exception as e:
        logger.error(f"序列化失败: {e}")
        return {
            "error": f"序列化失败: {str(e)}",
            "object_type": str(type(obj)),
            "timestamp": time.time(),
        }


def _serialize_sync(obj: Any) -> Dict:
    """同步序列化函数"""
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return {"data": obj, "type": type(obj).__name__}

    if isinstance(obj, dict):
        return {k: _serialize_sync(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple, set)):
        return [_serialize_sync(v) for v in obj]

    if hasattr(obj, "__dict__"):
        return {k: _serialize_sync(v) for k, v in obj.__dict__.items()}

    if hasattr(obj, "__slots__"):
        return {
            name: _serialize_sync(getattr(obj, name))
            for name in obj.__slots__
            if hasattr(obj, name)
        }

    return {"repr": repr(obj), "type": type(obj).__name__}


async def safe_redis_operation(
    operation_func, *args, timeout: float = 10.0, **kwargs
):
    """安全的Redis操作，带重试和超时"""
    max_retries = 3
    retry_delay = 1.0

    for attempt in range(max_retries):
        try:
            return await asyncio.wait_for(
                operation_func(*args, **kwargs),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            logger.warning(f"Redis操作超时，尝试 {attempt + 1}/{max_retries}")
            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delay * (attempt + 1))
            else:
                raise
        except Exception as e:
            logger.error(
                f"Redis操作失败，尝试 {attempt + 1}/{max_retries}: {e}"
            )
            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delay * (attempt + 1))
            else:
                raise


def create_improved_agent_stream(
    state_manager, user_id: str, chat_id: str, request
):
    """创建改进的流式响应生成器"""

    async def improved_agent_stream():
        """改进的Agent流式响应生成器"""
        task_id = None
        stream_manager = StreamResponseManager()

        try:
            # 初始化逻辑...
            context = MockContext(request)

            # 清空旧的流式数据
            await safe_redis_operation(
                state_manager.clear_stream_data,
                user_id,
                chat_id,
            )

            # 发送开始消息
            start_data = {
                "object": "content",
                "type": "status",
                "status": "started",
                "message": "任务开始执行",
                "timestamp": time.time(),
            }

            sequence_number = await safe_redis_operation(
                state_manager.store_stream_data,
                user_id,
                chat_id,
                start_data,
                task_id,
            )
            start_data["sequence_number"] = sequence_number
            stream_manager.update_data_time()

            yield f"data: {json.dumps(start_data, ensure_ascii=False)}\n\n"

            # 设置任务状态
            await state_manager.update_chat_state(
                user_id,
                chat_id,
                {
                    "is_running": True,
                    "current_task": f"Agent API Task from input: {len(request.input)} messages",
                },
            )

            # 获取聊天状态
            chat_state = await state_manager.get_chat_state(user_id, chat_id)
            task_id = chat_state.get("task_id")

            # 创建Agent配置
            agent_config = {
                "equipment": chat_state.get("equipment"),
                "output_dir": ".",
                "sandbox_type": chat_state.get("sandbox_type"),
                "status_callback": None,
                "mode": (
                    "phone_use"
                    if chat_state.get("sandbox_type") == "phone_wuyin"
                    else "pc_use"
                ),
                "pc_use_add_info": request.config.pc_use_addon_info,
                "max_steps": request.config.max_steps,
                "chat_id": chat_id,
                "user_id": user_id,
                "e2e_info": request.config.e2e_info,
                "extra_params": request.config.extra_params,
                "state_manager": state_manager,
            }

            # 创建Agent实例
            from computer_use_agent import ComputerUseAgent
            import socket
            import os

            agent = ComputerUseAgent(
                name="ComputerUseAgent",
                agent_config=agent_config,
            )

            # 存储agent引用
            import weakref

            if not hasattr(state_manager, "running_agents"):
                state_manager.running_agents = weakref.WeakValueDictionary()

            state_manager.running_agents[f"{user_id}:{chat_id}"] = agent

            # 获取机器标识
            machine_id = os.getenv("MACHINE_ID", socket.gethostname())

            # 更新Redis状态，包含机器信息
            await state_manager.update_chat_state(
                user_id,
                chat_id,
                {
                    "agent_running": True,
                    "agent_id": id(agent),
                    "agent_machine_id": machine_id,
                    "agent_pid": os.getpid(),
                },
            )

            # 启动控制信号监听任务
            async def listen_for_control_signals_task():
                """监听Redis控制信号的后台任务"""
                channel = f"control:{user_id}:{chat_id}"
                pubsub = state_manager.redis_client.pubsub()

                try:
                    await pubsub.subscribe(channel)
                    logger.info(f"开始监听控制信号: {channel}")

                    async for message in pubsub.listen():
                        if message['type'] == 'message':
                            try:
                                signal_data = json.loads(message['data'])
                                action = signal_data.get('action')

                                logger.info(f"收到控制信号: {action} from {channel}")

                                if action == 'stop':
                                    # 设置停止标志
                                    if hasattr(agent, 'should_stop'):
                                        agent.should_stop = True
                                    # 更新Redis状态
                                    await state_manager.update_chat_state(
                                        user_id, chat_id,
                                        {"is_running": False, "stop_requested": True}
                                    )
                                    break  # 停止监听
                                elif action == 'interrupt_wait':
                                    # 调用中断等待方法
                                    if hasattr(agent, 'interrupt_wait'):
                                        agent.interrupt_wait()

                            except Exception as e:
                                logger.error(f"处理控制信号时出错: {e}")

                except Exception as e:
                    logger.error(f"监听控制信号失败: {e}")
                finally:
                    try:
                        await pubsub.unsubscribe(channel)
                        await pubsub.close()
                    except Exception as e:
                        logger.error(f"关闭pub/sub连接失败: {e}")

            # 启动控制信号监听任务
            control_signal_task = asyncio.create_task(listen_for_control_signals_task())

            logger.info(f"开始Agent执行，用户: {user_id}, 对话: {chat_id}")

            # 执行Agent任务并处理流式输出
            async_iterator = None
            try:
                async_iterator = agent.run_async(context)

                async for result in async_iterator:
                    # 在每次循环开始时检查停止信号
                    try:
                        chat_state = await state_manager.get_chat_state(user_id, chat_id)
                        if isinstance(chat_state, dict):
                            should_stop = not chat_state.get("is_running", True) or chat_state.get("stop_requested", False)
                        else:
                            should_stop = not getattr(chat_state, "is_running", True) or getattr(chat_state, "stop_requested", False)

                        if should_stop or (hasattr(agent, 'should_stop') and agent.should_stop):
                            logger.info(f"检测到停止信号，结束Agent执行: {user_id}:{chat_id}")
                            break
                    except Exception as stop_check_error:
                        logger.error(f"检查停止信号时出错: {stop_check_error}")

                    try:
                        # 检查是否需要发送心跳
                        if stream_manager.should_send_heartbeat():
                            heartbeat = await stream_manager.send_heartbeat()
                            stream_manager.update_heartbeat_time()
                            yield heartbeat

                        # 安全序列化Agent输出
                        if hasattr(result, "model_dump"):
                            result_dict = result.model_dump()
                        else:
                            result_dict = await safe_serialize(result)

                        # 存储到Redis（带重试）
                        sequence_number = await safe_redis_operation(
                            state_manager.store_stream_data,
                            user_id,
                            chat_id,
                            result_dict,
                            task_id,
                        )

                        result_dict["sequence_number"] = sequence_number
                        stream_manager.update_data_time()

                        # 生成JSON响应
                        json_str = json.dumps(result_dict, ensure_ascii=False)
                        yield f"data: {json_str}\n\n"

                    except Exception as serialize_error:
                        logger.error(f"处理输出时出错: {serialize_error}")

                        # 创建错误响应
                        error_data = {
                            "object": "error",
                            "type": "serialization_error",
                            "error": f"序列化输出时出错: {str(serialize_error)}",
                            "timestamp": time.time(),
                        }

                        try:
                            sequence_number = await safe_redis_operation(
                                state_manager.store_stream_data,
                                user_id,
                                chat_id,
                                error_data,
                                task_id,
                            )
                            error_data["sequence_number"] = sequence_number
                            stream_manager.update_data_time()

                            yield f"data: {json.dumps(error_data, ensure_ascii=False)}\n\n"
                        except Exception as redis_error:
                            logger.error(
                                f"存储错误信息到Redis失败: {redis_error}"
                            )
                            # 降级：直接发送错误信息，不存储到Redis
                            error_data["sequence_number"] = None
                            yield f"data: {json.dumps(error_data, ensure_ascii=False)}\n\n"

                        continue

            except Exception as iteration_error:
                logger.error(f"Agent执行时出错: {iteration_error}")

                error_data = {
                    "object": "error",
                    "type": "iteration_error",
                    "error": f"Agent执行时出错: {str(iteration_error)}",
                    "timestamp": time.time(),
                }

                try:
                    sequence_number = await safe_redis_operation(
                        state_manager.store_stream_data,
                        user_id,
                        chat_id,
                        error_data,
                        task_id,
                    )
                    error_data["sequence_number"] = sequence_number

                    yield f"data: {json.dumps(error_data, ensure_ascii=False)}\n\n"
                except Exception as redis_error:
                    logger.error(f"存储执行错误到Redis失败: {redis_error}")
                    error_data["sequence_number"] = None
                    yield f"data: {json.dumps(error_data, ensure_ascii=False)}\n\n"

            finally:
                # 清理async_iterator
                if async_iterator and hasattr(async_iterator, "aclose"):
                    try:
                        await async_iterator.aclose()
                    except Exception as close_error:
                        logger.error(f"关闭异步迭代器时出错: {close_error}")

            # 发送完成消息
            completion_data = {
                "object": "content",
                "type": "completion",
                "status": "completed",
                "message": "任务执行完成",
                "timestamp": time.time(),
            }

            try:
                sequence_number = await safe_redis_operation(
                    state_manager.store_stream_data,
                    user_id,
                    chat_id,
                    completion_data,
                    task_id,
                )
                completion_data["sequence_number"] = sequence_number

                yield f"data: {json.dumps(completion_data, ensure_ascii=False)}\n\n"
            except Exception as redis_error:
                logger.error(f"存储完成状态失败: {redis_error}")
                completion_data["sequence_number"] = None
                yield f"data: {json.dumps(completion_data, ensure_ascii=False)}\n\n"

            logger.info(f"Agent执行完成，用户: {user_id}, 对话: {chat_id}")

        except Exception as e:
            logger.error(
                f"Agent stream execution failed for user {user_id}, chat {chat_id}: {e}"
            )

            # 发送全局错误
            final_error = {
                "object": "error",
                "type": "agent_error",
                "error": f"任务执行失败: {str(e)}",
                "timestamp": time.time(),
                "sequence_number": None,
            }

            yield f"data: {json.dumps(final_error, ensure_ascii=False)}\n\n"

        finally:
            # 清理控制信号监听任务
            try:
                if 'control_signal_task' in locals() and not control_signal_task.done():
                    control_signal_task.cancel()
                    try:
                        await control_signal_task
                    except asyncio.CancelledError:
                        logger.info(f"控制信号监听任务已取消: {user_id}:{chat_id}")
                    except Exception as task_error:
                        logger.error(f"取消控制信号监听任务时出错: {task_error}")
            except Exception as signal_cleanup_error:
                logger.error(f"清理控制信号监听任务失败: {signal_cleanup_error}")

            # 清理状态
            try:
                # 清理agent引用
                composite_key = f"{user_id}:{chat_id}"
                if (
                    hasattr(state_manager, "running_agents")
                    and composite_key in state_manager.running_agents
                ):
                    del state_manager.running_agents[composite_key]

                await state_manager.update_chat_state(
                    user_id,
                    chat_id,
                    {
                        "is_running": False,
                        "current_task": None,
                        "agent_running": False,
                        "agent_id": None,
                        "agent_machine_id": None,
                        "agent_pid": None,
                        "stop_requested": False,
                    },
                )
            except Exception as cleanup_error:
                logger.error(f"清理状态时出错: {cleanup_error}")

    return improved_agent_stream


class MockContext:
    """模拟AgentScope Context"""

    def __init__(self, request):
        self.request = request


# 网络配置优化建议
IMPROVED_STREAM_HEADERS = {
    "Cache-Control": "no-cache, no-store, must-revalidate",
    "Connection": "keep-alive",
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Credentials": "true",
    "X-Accel-Buffering": "no",  # 禁用nginx缓冲
    "Content-Type": "text/event-stream; charset=utf-8",
    "Transfer-Encoding": "chunked",
    "Keep-Alive": "timeout=300, max=1000",  # 设置keep-alive参数
}


def create_stream_response_with_error_handling(stream_generator):
    """创建带错误处理的流式响应"""
    from fastapi.responses import StreamingResponse

    async def error_wrapped_stream():
        try:
            async for chunk in stream_generator():
                yield chunk
        except Exception as e:
            logger.error(f"流式响应生成器出错: {e}")
            # 发送错误信息给客户端
            error_chunk = {
                "object": "error",
                "type": "stream_error",
                "error": f"流式响应出错: {str(e)}",
                "timestamp": time.time(),
            }
            yield f"data: {json.dumps(error_chunk, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        error_wrapped_stream(),
        media_type="text/event-stream",
        headers=IMPROVED_STREAM_HEADERS,
    )
