"""
VLM Tool Call Agent — agentic VLM framework with Docker Jupyter notebook tool.

The agent loop:
1. Send user message (with optional images) to the VLM
2. If the model calls ``execute_code``, run the code in the Docker kernel
3. Feed results (text + images) back to the model
4. Repeat until the model calls ``finish`` or max iterations reached
"""

import datetime
import json
import os
import traceback
from typing import Any, Dict, List, Optional

from openai import OpenAI

from swe_vision.config import (
    ATTACH_IMAGES_TO_LLM,
    DEFAULT_MODEL,
    MAX_ITERATIONS,
    REASONING_BACKEND,
    SYSTEM_PROMPT,
    TOOLS,
    logger,
)
from swe_vision.file_manager import NotebookFileManager
from swe_vision.image_utils import make_base64_image_content_part, make_image_content_part
from swe_vision.kernel import JupyterNotebookKernel
from swe_vision.trajectory import TrajectoryRecorder


class VLMToolCallAgent:
    """
    An agentic VLM framework that uses OpenAI's function calling to
    give a vision-language model access to a stateful Jupyter notebook
    running inside a Docker container.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        system_prompt: str = SYSTEM_PROMPT,
        max_iterations: int = MAX_ITERATIONS,
        verbose: bool = True,
        save_trajectory: Optional[str] = None,
        reasoning: bool = True,
    ):
        self.model = model
        self.max_iterations = max_iterations
        self.system_prompt = system_prompt
        self.verbose = verbose
        self.reasoning = reasoning

        self._save_trajectory_dir = save_trajectory

        client_kwargs = {}
        if api_key:
            client_kwargs["api_key"] = api_key
        if base_url:
            client_kwargs["base_url"] = base_url
        elif os.environ.get("OPENAI_BASE_URL"):
            client_kwargs["base_url"] = os.environ["OPENAI_BASE_URL"]

        self.client = OpenAI(**client_kwargs)

        print(f"Using model: {self.model}")
        print(f"Using API key: {api_key}")
        print(f"Using base URL: {base_url}")

        self.kernel: Optional[JupyterNotebookKernel] = None
        self.file_manager = NotebookFileManager()

        self.messages: List[Dict[str, Any]] = []

        self.trajectory: Optional[TrajectoryRecorder] = None

    async def _ensure_kernel(self):
        if self.kernel is None:
            self.kernel = JupyterNotebookKernel()
        if not self.kernel._started:
            await self.kernel.start()
            self.file_manager.setup_work_dir(
                host_work_dir=self.kernel.host_work_dir,
                container_work_dir=self.kernel.container_work_dir,
                kernel=self.kernel,
            )

    def _log(self, msg: str, *args, level: str = "info"):
        getattr(logger, level)(msg, *args)
        if self.verbose:
            formatted = msg % args if args else msg
            print(f"  [{level.upper()}] {formatted}", flush=True)

    def _build_user_message(
        self,
        query: str,
        image_paths: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        content = []

        file_hints = []
        if image_paths:
            basenames = [os.path.basename(os.path.abspath(p)) for p in image_paths]
            has_collision = len(basenames) != len(set(basenames))

            for idx, img_path in enumerate(image_paths):
                img_path = os.path.abspath(img_path)
                if not os.path.exists(img_path):
                    self._log("Warning: image not found: %s", img_path, level="warning")
                    continue
                if ATTACH_IMAGES_TO_LLM:
                    content.append(make_image_content_part(img_path))
                dest_name = None
                if has_collision or len(image_paths) > 1:
                    base = os.path.basename(img_path)
                    name, ext = os.path.splitext(base)
                    dest_name = f"{idx}_{name}{ext}"
                container_path = self.file_manager.copy_file_to_workdir(
                    img_path, dest_name=dest_name,
                )
                file_hints.append(container_path)

        text = query
        if file_hints:
            paths_str = ", ".join(f"`{p}`" for p in file_hints)
            text += f"\n\n[Uploaded file(s) available at: {paths_str}]"
        content.insert(0, {"type": "text", "text": text})

        return {"role": "user", "content": content}

    def _call_llm(self) -> Any:
        kwargs = dict(
            model=self.model,
            messages=self.messages,
            tools=TOOLS,
            tool_choice="auto",
        )
        reasoning_backend = REASONING_BACKEND
        if reasoning_backend == "auto":
            model_lower = (self.model or "").lower()
            reasoning_backend = "qwen3" if "qwen3" in model_lower else "openai"

        if self.reasoning:
            if reasoning_backend == "qwen3":
                kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": True}}
            else:
                kwargs["extra_body"] = {"reasoning": {"enabled": True, "effort": "xhigh"}}
                kwargs["reasoning_effort"] = "xhigh"
        else:
            if reasoning_backend == "qwen3":
                kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
            else:
                kwargs["extra_body"] = {"reasoning": {"enabled": False, "effort": "minimal"}}

        response = self.client.chat.completions.create(**kwargs)
        return response

    async def _handle_execute_code(self, code: str) -> Dict[str, Any]:
        await self._ensure_kernel()

        self._log("Executing code in Docker Jupyter notebook:\n%s",
                   code[:200] + ("..." if len(code) > 200 else ""))

        result = await self.kernel.execute(code)

        text = result["text_output"]
        if result["status"] == "error":
            text = f"[Execution Error]\n{text}"

        image_parts = []
        for img_b64 in result["images"]:
            image_parts.append(make_base64_image_content_part(img_b64))

        return {
            "text_output": text,
            "image_parts": image_parts,
            "base64_images": result["images"],
        }

    def _init_trajectory(self, query: str, image_paths: Optional[List[str]]) -> TrajectoryRecorder:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        if self._save_trajectory_dir:
            save_dir = f"{self._save_trajectory_dir}_{ts}"
        else:
            save_dir = os.path.join("trajectories", f"run_{ts}")
        recorder = TrajectoryRecorder(save_dir)
        recorder.set_metadata(
            model=self.model,
            start_time=TrajectoryRecorder._now_iso(),
            query=query,
            image_paths=image_paths or [],
            max_iterations=self.max_iterations,
            system_prompt=self.system_prompt,
        )
        return recorder

    async def run(
        self,
        query: str,
        image_paths: Optional[List[str]] = None,
    ) -> str:
        """
        Run the agentic loop for a single user query.

        Returns the final answer string.
        """
        self.trajectory = self._init_trajectory(query, image_paths)

        self.messages = [
            {"role": "system", "content": self.system_prompt},
        ]

        user_msg = self._build_user_message(query, image_paths)
        self.messages.append(user_msg)

        self.trajectory.record_user_step(query, image_paths)

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"User Query: {query}")
            if image_paths:
                print(f"Images: {image_paths}")
            print(f"{'='*60}\n")

        final_answer = None
        try:
            final_answer = await self._run_loop()
        finally:
            if final_answer is not None:
                self.trajectory.record_finish(final_answer)
            self.trajectory.save()
            self.trajectory.save_messages_raw(self.messages)

        return final_answer

    async def _run_loop(self) -> str:
        """Core agentic loop."""
        for iteration in range(1, self.max_iterations + 1):
            if self.verbose:
                print(f"\n--- Iteration {iteration}/{self.max_iterations} ---")

            MAX_RETRIES = 10
            for retry in range(MAX_RETRIES):
                try:
                    response = self._call_llm()
                    break
                except Exception as e:
                    self._log("OpenAI API error: %s, retry %d/%d", str(e), retry, MAX_RETRIES, level="error")

            if retry == MAX_RETRIES - 1:
                return f"[Error] Failed to call LLM: {e}"

            choice = response.choices[0]
            message = choice.message

            if hasattr(message, "to_dict"):
                assistant_msg = message.to_dict()
            elif hasattr(message, "model_dump"):
                assistant_msg = message.model_dump()
            else:
                assistant_msg = {"role": "assistant", "content": message.content}
            assistant_msg.setdefault("role", "assistant")
            self.messages.append(assistant_msg)

            tool_call_dicts = assistant_msg.get("tool_calls")
            reasoning_details = assistant_msg.get("reasoning_details")

            self.trajectory.record_assistant_step(
                message.content, tool_call_dicts, reasoning_details=reasoning_details,
            )

            try:
                if message.reasoning and self.verbose:
                    summary = message.reasoning if isinstance(message.reasoning, str) else ""
                    preview = summary[:300] + ("..." if len(summary) > 300 else "")
                    print(f"\n[Reasoning] {preview}")
            except Exception:
                try:
                    summary = message.reasoning_content[:300]
                except Exception:
                    pass

            if message.content:
                if self.verbose:
                    print(f"\n[Assistant] {message.content[:500]}")

            if not message.tool_calls:
                if choice.finish_reason == "stop":
                    self._log("Model stopped without calling finish tool.")
                    return message.content or "[No response]"
                continue

            for tool_call in message.tool_calls:
                fn_name = tool_call.function.name
                try:
                    fn_args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError as e:
                    self._log("Failed to parse tool arguments: %s", e, level="error")
                    err_text = f"[Error] Invalid JSON arguments: {e}"
                    self.messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": err_text,
                    })
                    self.trajectory.record_tool_step(
                        tool_call_id=tool_call.id,
                        tool_name=fn_name,
                        code=None,
                        text_output=err_text,
                    )
                    continue

                if fn_name == "finish":
                    answer = fn_args.get("answer", "")
                    if self.verbose:
                        print(f"\n{'='*60}")
                        print(f"[FINISH] Final Answer:")
                        print(answer)
                        print(f"{'='*60}\n")
                    return answer

                elif fn_name == "execute_code":
                    code = fn_args.get("code", "")
                    text_output = ""
                    image_parts: List[Dict[str, Any]] = []
                    base64_images: List[str] = []
                    try:
                        exec_result = await self._handle_execute_code(code)
                        text_output = exec_result["text_output"]
                        image_parts = exec_result["image_parts"]
                        base64_images = exec_result["base64_images"]
                    except Exception as e:
                        tb = traceback.format_exc()
                        self._log("Code execution failed: %s", e, level="error")
                        text_output = f"[Execution Error] {e}\n{tb}"

                    if image_parts:
                        tool_content: Any = [
                            {"type": "text", "text": text_output},
                        ] + image_parts
                    else:
                        tool_content = text_output

                    self.messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": tool_content,
                    })

                    self.trajectory.record_tool_step(
                        tool_call_id=tool_call.id,
                        tool_name=fn_name,
                        code=code,
                        text_output=text_output,
                        base64_images=base64_images,
                    )

                    if self.verbose:
                        print(f"\n[Code Output] {text_output[:500]}")
                        if image_parts:
                            print(f"  [{len(image_parts)} image(s) returned to model in tool message]")

                else:
                    self._log("Unknown tool: %s", fn_name, level="warning")
                    err_text = f"[Error] Unknown tool: {fn_name}"
                    self.messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": err_text,
                    })
                    self.trajectory.record_tool_step(
                        tool_call_id=tool_call.id,
                        tool_name=fn_name,
                        code=None,
                        text_output=err_text,
                    )

        self._log("Max iterations reached (%d)", self.max_iterations, level="warning")
        return "[Error] Max iterations reached without a final answer."

    async def run_interactive(self, image_paths: Optional[List[str]] = None):
        """
        Run in interactive mode — the user can keep asking questions
        and the kernel state is preserved.
        """
        print("\n" + "="*60)
        print("VLM Tool Call Agent - Interactive Mode (Docker Runtime)")
        print("Type 'quit' or 'exit' to stop.")
        print("Type 'image:<path>' to add an image to the next query.")
        print("="*60 + "\n")

        session_images = list(image_paths or [])

        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break

            if not user_input:
                continue
            if user_input.lower() in ("quit", "exit"):
                print("Goodbye!")
                break

            if user_input.lower().startswith("image:"):
                img_path = user_input[6:].strip()
                if os.path.exists(img_path):
                    session_images.append(img_path)
                    print(f"  Added image: {img_path}")
                else:
                    print(f"  Image not found: {img_path}")
                continue

            answer = await self.run(user_input, session_images if session_images else None)
            print(f"\nAnswer: {answer}\n")

            session_images = []

    async def cleanup(self):
        """Shut down the Docker kernel and clean up resources."""
        if self.kernel:
            await self.kernel.shutdown()
