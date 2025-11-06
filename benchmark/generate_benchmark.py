import os
import sys
import time
import requests
import pandas as pd
from biochatter.llm_connect import GeminiConversation


def get(biotools_id: str) -> list:
    response = requests.get(f"https://bio.tools/api/tool/{biotools_id}/?format=json").json()
    return response["description"], response["topic"], response["function"]


def get_terms(function):
    operations = function["operation"]
    inputs = function["input"]
    outputs = function["output"]
    return operations, inputs, outputs


def throttled_query(conversation, query):
    response, _, _ = conversation.query(query)
    time.sleep(8)
    return response


tool_id = "wtv"
desc, topics, functions = get(tool_id)

qa = []

conversation = GeminiConversation(
    model_name="gemini-2.0-flash",
    prompts={},
)

conversation.set_api_key(api_key="")

instructions = """
You are assisting in creating question-answer pairs for an MCP benchmark.
You will be tasked to generate content describing a software package based on topics, operations and input and output data descriptions including supported formats.
"""

conversation.append_system_message(instructions)

qa.append(
    {
        "question": "What topics is this package associated with? -" + desc,
        "answer": ", ".join([x["uri"] for x in topics]),
    }
)

for func in functions:
    ops, ins, outs = get_terms(func)
    for i, op in enumerate(ops):
        query = f"Generate a short description based on the term labels for the functionality of the package {op}."
        content, _, _ = conversation.query(query)

        query = f"Generate a question a developer could pose if they wanted to annotate the package with a tag for this operation: {op}, but in a very general way (like: what is the main function of this package? Or: What else beside the main function can I do with this package?). Keep it general and don't include the operation directly in the question. Pose the question in a way that it makes sense as the {i}th question out of {len(ops)} without explicitly including the number of operations etc. in the question"

        question = throttled_query(conversation, query)

        result = f"{question} - {content}"
        print(result, os.linesep)

        qa.append({"question": result, "answer": op["uri"]})

    for i, inp in enumerate(ins):
        query = (
            f"Generate a description based on the term labels for the input data specifications for the package {inp}."
        )
        content, _, _ = conversation.query(query)

        query = f"Generate a question a developer could pose if they wanted to annotate the package with a tag for this input data: {inp}, but in a very general way (like: Which type of data does this package mainly handle?). Keep it general and don't include the data directly in the question. Pose the question in a way that it makes sense as the {i}th question out of {len(ins)} without explicitly including the number of inputs handled etc. in the question"

        question = throttled_query(conversation, query)

        result = f"{question} - {content}"
        print(result, os.linesep)

        answer = inp["data"]["uri"] + ", " + ", ".join([x["uri"] for x in inp["format"]])

        qa.append({"question": result, "answer": answer})

    for i, out in enumerate(outs):
        query = (
            f"Generate a description based on the term labels for the input data specifications for the package {out}."
        )
        content, _, _ = conversation.query(query)

        query = f"Generate a question a developer could pose if they wanted to annotate the package with a tag for the output data of the package ({out}). Keep it general and don't include the data directly in the question. Pose the question in a way that it makes sense as the {i}th question out of {len(outs)} without explicitly including the number of inputs handled etc. in the question"

        question = throttled_query(conversation, query)

        result = f"{question} - {content}"
        print(result, os.linesep)

        answer = out["data"]["uri"] + ", " + ", ".join([x["uri"] for x in out["format"]])

        qa.append({"question": result, "answer": answer})

outpath = os.path.join("benchmark", tool_id)
os.mkdir(outpath)
pd.DataFrame(qa).to_csv(os.path.join(outpath, "qa.tsv"), sep="\t", index=False)
