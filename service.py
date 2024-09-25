import time
import uuid
from typing import AsyncGenerator, Literal

import bentoml
import pydantic
from annotated_types import Ge, Le
from bentoml.models import HuggingFaceModel
from typing_extensions import Annotated


MODEL_ID = "Qwen/Qwen2-7B-Instruct-AWQ"
VLLM_ENGINE_CONFIG = {"max_model_len": 2048, "quantization": "awq"}


class Message(pydantic.BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


@bentoml.service(
    name="qwen2",
    traffic={"timeout": 300},
    resources={"gpu": 1, "gpu_type": "nvidia-l4"},
    monitoring={
        "enabled": True,
        "type": "default",
    }
)
class VLLM:
    llm = HuggingFaceModel(MODEL_ID)

    def __init__(self) -> None:
        from vllm import AsyncEngineArgs, AsyncLLMEngine

        args = AsyncEngineArgs(model=self.llm, **VLLM_ENGINE_CONFIG)
        self.engine = AsyncLLMEngine.from_engine_args(args)

    @bentoml.api(route="/api/chat")
    async def chat(
        self,
        messages: list[Message] = [
            Message(content="what is the meaning of life?", role="user")
        ],
        model: str = MODEL_ID,
        max_tokens: Annotated[
            int,
            Ge(128),
            Le(VLLM_ENGINE_CONFIG["max_model_len"]),
        ] = VLLM_ENGINE_CONFIG["max_model_len"],
    ) -> AsyncGenerator[str, None]:
        from vllm import SamplingParams

        tokenizer = await self.engine.get_tokenizer()

        try:
            SAMPLING_PARAM = SamplingParams(max_tokens=max_tokens)

            prompt = tokenizer.apply_chat_template(
                messages,  # type: ignore
                tokenize=False,
                add_generation_prompt=True,
            )

            with bentoml.monitor("llm_chat") as mon:
                start_time = time.time()

                n_in_tokens = len(tokenizer.encode(prompt))  # type: ignore
                mon.log(n_in_tokens, name="input_tokens", role="feature", data_type="numerical")
                mon.log(max_tokens, name="max_tokens", role="feature", data_type="numerical")

                stream = await self.engine.add_request(uuid.uuid4().hex, prompt, SAMPLING_PARAM)  # type: ignore

                cursor = 0
                n_out_tokens = 0
                is_ft = True
                async for request_output in stream:
                    if is_ft:
                        mon.log(time.time() - start_time, name="TTFT", role="prediction", data_type="numerical")
                        is_ft = False
                    text = request_output.outputs[0].text
                    n_out_tokens = len(request_output.outputs[0].token_ids)
                    yield text[cursor:]
                    cursor = len(text)

                mon.log(time.time() - start_time, name="total_time", role="prediction", data_type="numerical")
                mon.log(n_out_tokens, name="output_tokens", role="prediction", data_type="numerical")
        except Exception as e:
            yield f"Error in chat API: {e}"
