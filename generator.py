import argparse
import os

import pandas as pd
from llama_cpp import Llama

from templates import PROMPT_STRATEGY

parser = argparse.ArgumentParser()
parser.add_argument("-g", "--gpu_devices", help=" : CUDA_VISIBLE_DEVICES", default="0")
parser.add_argument(
    "-m",
    "--model",
    help=" : Model to evaluate",
)
parser.add_argument("-ml", "--model_len", help=" : Maximum Model Length", default=4096, type=int)
parser.add_argument("-s", "--stop", help=" : Stop tokens to add", default=None, type=str)
args = parser.parse_args()

print(f"Args - {args}")

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_devices
gpu_counts = len(args.gpu_devices.split(","))

llm = Llama(
    model_path=args.model,
    n_gpu_layers=-1,
    n_ctx=args.model_len,
    verbose=False,
)

temperature = 0
max_tokens = args.model_len
stop = ["<|endoftext|>", "[INST]", "[/INST]", "<|im_end|>", "<|end|>", "<|eot_id|>", "<end_of_turn>", "<eos>"]
if args.stop != None:
    stop.extend(args.stop.split(","))

df_questions = pd.read_json("questions.jsonl", orient="records", encoding="utf-8-sig", lines=True)

if not os.path.exists("./generated/" + args.model):
    os.makedirs("./generated/" + args.model)

for strategy_name, prompts in PROMPT_STRATEGY.items():

    def single_turn_question(question):
        output = llm.create_chat_completion(
            messages=prompts + [{"role": "user", "content": question[0]}],
            temperature=temperature,
            stop=stop,
            max_tokens=max_tokens,
        )
        return output["choices"][0]["message"]["content"].strip()

    single_turn_outputs = df_questions["questions"].map(single_turn_question)

    def double_turn_question(question, single_turn_output):
        output = llm.create_chat_completion(
            messages=prompts
            + [
                {"role": "user", "content": question[0]},
                {"role": "assistant", "content": single_turn_output},
                {"role": "user", "content": question[1]},
            ],
            temperature=temperature,
            stop=stop,
            max_tokens=max_tokens,
        )
        return output["choices"][0]["message"]["content"].strip()

    multi_turn_outputs = df_questions[["questions", "id"]].apply(
        lambda x: double_turn_question(x["questions"], single_turn_outputs[x["id"] - 1]),
        axis=1,
    )

    df_output = pd.DataFrame(
        {
            "id": df_questions["id"],
            "category": df_questions["category"],
            "questions": df_questions["questions"],
            "outputs": list(zip(single_turn_outputs, multi_turn_outputs)),
            "references": df_questions["references"],
        }
    )
    df_output.to_json(
        "./generated/" + os.path.join(args.model, f"{strategy_name}.jsonl"),
        orient="records",
        lines=True,
        force_ascii=False,
    )
