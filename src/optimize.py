import json
import os
import time
from pathlib import Path

import dspy

from src import config, data, estimate, prompts


def main():  # pragma: no cover - needs the server
    # about 3 short calls to the student model; the seconds figure is a guess until M4 measures real call times
    estimate.estimate(
        [{"model": config.SERVED_NAME, "calls": 3, "seconds": 30, "basis": "assumed"}]
    )
    lm = dspy.LM(
        "openai/"
        + config.SERVED_NAME,  # "openai/" = speak the OpenAI HTTP format, which vLLM also speaks
        api_base=os.environ["MODAL_URL"] + "/v1",
        api_key="unused",  # litellm refuses an empty key; vLLM ignores it and Modal's proxy tokens in headers do the checking
        headers={
            "Modal-Key": os.environ["MODAL_ID"],
            "Modal-Secret": os.environ["MODAL_SECRET"],
        },
        model_type="chat",
        temperature=config.TEMPERATURE,
        max_tokens=config.MAX_NEW_TOKENS,
        cache=False,
    )
    dspy.configure(lm=lm, adapter=dspy.ChatAdapter(use_json_adapter_fallback=False))

    # a DSPy module: turns the signature into a prompt, calls the LM, and parses the reply into fields
    predict = dspy.Predict(dspy.Signature(prompts.SIGNATURE, prompts.ONE_LINER))

    splits = json.loads(Path(config.SPLITS_PATH).read_text())
    _, tasks = data.load_examples()
    # a task is one IFEval prompt plus the rule its answer must follow; this one is from train, so test stays untouched until M8
    task = next(t for t in tasks if t["key"] == splits["train"][0])

    start = time.time()
    replies = []
    for _ in range(2):
        try:
            replies.append(predict(prompt=task["prompt"]).response)
        except dspy.utils.exceptions.AdapterParseError as error:
            # the model answered without DSPy's [[ ## response ## ]] marker
            replies.append("PARSE ERROR, raw reply: " + error.lm_response)
    seconds = time.time() - start

    print("prompt:", task["prompt"][:100])
    for number, reply in enumerate(replies, 1):
        print(f"reply {number}: {reply[:200]!r}")
    print("replies differ:", replies[0] != replies[1])

    print("\nthe messages DSPy sent for the last call:")
    for message in lm.history[-1]["messages"]:
        print(f"[{message['role']}]\n{message['content']}\n")

    output_tokens = sum(h["usage"]["completion_tokens"] for h in lm.history)
    print(
        f"actual: {len(lm.history)} calls, {seconds:.0f} s "
        f"(about ${seconds * config.T4_PER_SECOND:.2f}), {output_tokens} output tokens"
    )


if __name__ == "__main__":  # pragma: no cover
    main()
