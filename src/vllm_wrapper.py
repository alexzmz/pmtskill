"""vLLM Wrapper for AndroidWorld's T3A Agent."""

from typing import Any, Optional
from vllm import LLM, SamplingParams


class VLLMWrapper:
    """vLLM wrapper compatible with T3A's expected interface."""

    def __init__(
        self,
        model_path: str,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        max_model_len: Optional[int] = None,
        temperature: float = 0.7,
        top_p: float = 0.95,
        max_tokens: int = 512,
    ):
        self.model_path = model_path
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens

        # Initialize vLLM engine
        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            trust_remote_code=True,  # Required for Qwen models
        )

        self.sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )

    def predict(self, text_prompt: str) -> tuple[str, Optional[bool], Any]:
        """Generate text response from prompt.

        Returns:
            Tuple of (text_output, is_safe, raw_response)
            - text_output: The generated response string
            - is_safe: Always None (vLLM doesn't have safety filtering)
            - raw_response: The vLLM output object (non-None to avoid error)
        """
        try:
            # Generate using vLLM
            outputs = self.llm.generate([text_prompt], self.sampling_params)

            # Extract generated text
            generated_text = outputs[0].outputs[0].text.strip()

            # raw_response 必须非 None（T3A 会检查）
            # 直接返回 outputs 对象
            return (generated_text, None, outputs)

        except Exception as e:
            # 错误时也要确保 raw_response 非 None
            error_msg = f"vLLM inference failed: {str(e)}"
            # 返回一个虚拟的 raw_response 避免 T3A 报错
            return (error_msg, False, {"error": str(e)})

    def predict_mm(
        self, text_prompt: str, images: Optional[list] = None
    ) -> tuple[str, Optional[bool], Any]:
        """Multimodal prediction fallback (not used by T3A)."""
        return self.predict(text_prompt)
