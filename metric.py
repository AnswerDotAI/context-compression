import os
from claudette import models, Chat
import numpy as np
from evaluate import load
import regex as re


class Metric:
    def __init__(self, **kwargs):
        self._load_metric(**kwargs)

    def _load_metric(self, **kwargs):
        raise NotImplementedError("This method should be overridden by subclasses.")

    def compute(self, prompts, predictions, references):
        raise NotImplementedError("This method should be overridden by subclasses.")


class Rouge(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _load_metric(self, **kwargs):
        self.metric = load("rouge", keep_in_memory=True)

    def compute(self, prompts, predictions, references):
        return self.metric.compute(predictions=predictions, references=references)


class Bleurt(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _load_metric(self, **kwargs):
        self.metric = load("bleurt", keep_in_memory=True)

    def compute(self, prompts, predictions, references):
        return np.mean(
            self.metric.compute(predictions=predictions, references=references)[
                "scores"
            ]
        )


class BertScore(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _load_metric(self, **kwargs):
        self.metric = load("bertscore", keep_in_memory=True)

    def compute(self, prompts, predictions, references):
        result = self.metric.compute(
            predictions=predictions, references=references, lang="en"
        )
        return {
            "precision": np.mean(result["precision"]),
            "recall": np.mean(result["recall"]),
            "f1": np.mean(result["f1"]),
        }


class Accuracy(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _load_metric(self, **kwargs):
        from sklearn.metrics import accuracy_score

        self.metric = accuracy_score

    def compute(self, prompts, predictions, references):
        return self.metric(references, predictions)


REFERENCE_TEMPLATE = """You are shown ground-truth answer(s) and asked to judge the quality of an LLM-generated answer.
Assign it a score from 1-5 where 1 is the worst and 5 is the best based on how similar it is to the ground-truth(s).
Do NOT explain your choice. Simply return a number from 1-5.

====GROUND TRUTHS====
{labels}

====ANSWER====
{prediction}"""

PREFILL = "The score (1-5) is:"


class LLMRouge(Metric):
    def __init__(self, **kwargs) -> None:
        assert (
            "ANTHROPIC_API_KEY" in os.environ
        ), "Please set the ANTHROPIC_API_KEY environment variable."
        super().__init__(**kwargs)

    def _load_metric(self, **kwargs):
        name = kwargs.get("name", "haiku")
        matching_names = [m for m in models if name in m]
        assert len(matching_names) > 0, f"Model name {name} not found in {models}"
        assert (
            len(matching_names) == 1
        ), f"Model name {name} found x{len(matching_names)} in {models}"
        self.chat = Chat(
            matching_names[0], sp="""You are a helpful and concise assistant."""
        )

    def parse_int(self, text):
        return int(re.search(r"\d+", text).group())

    def compute(self, prompts, predictions, labels):
        scores = []
        for p, ls in zip(predictions, labels):
            prompt = REFERENCE_TEMPLATE.format(labels="\n---\n".join(ls), prediction=p)
            score = (
                self.chat(prompt, prefill=PREFILL)
                .content[0]
                .text[len(PREFILL) :]
                .strip()
            )
            score = self.parse_int(score)
            scores.append(score)
        return {"llm_rouge": sum(scores) / len(scores)}


LLM_JUDGE_TEMPLATE = """You are shown a prompt and asked to assess the quality of an LLM-generated answer on the following dimensions:

===CRITERIA===
{criteria}

Respond with "criteria: score" for each criteria with a newline for each criteria.
Assign a score from 1-5 where 1 is the worst and 5 is the best based on how well the answer meets the criteria.

====PROMPT====
{prompt}

====ANSWER====
{prediction}"""


CRITERIA = {
    "helpful": "The answer executes the action requested by the prompt without extraneous detail.",
    "coherent": "The answer is logically structured and coherent (ignore the prompt).",
    "faithful": "The answer is faithful to the prompt and does not contain false information.",
}


class LLMJudge(LLMRouge):
    def __init__(self, **kwargs) -> None:
        assert (
            "ANTHROPIC_API_KEY" in os.environ
        ), "Please set the ANTHROPIC_API_KEY environment variable."
        super().__init__(**kwargs)

    def compute(self, prompts, predictions, labels):
        scores = []

        criteria = list(sorted([k for k in CRITERIA]))
        criteria_def = "\n".join([f"{k}: {CRITERIA[k]}" for k in criteria])
        prefill = f"\n\n====SCORES for {', '.join(criteria)}====\n\n{criteria[0]}:"

        for pred, prompt in zip(predictions, prompts):
            prompt = LLM_JUDGE_TEMPLATE.format(
                criteria=criteria_def, prompt=prompt, prediction=pred
            )
            score = (
                self.chat(prompt, prefill=prefill)
                .content[0]
                .text[len(prefill) - len(criteria[0]) - 1 :]
                .strip()
            )
            score_dict = dict(
                [s.strip().split(":") for s in score.split("\n") if len(s.strip()) > 0]
            )
            score_dict = {k: self.parse_int(v) for k, v in score_dict.items()}
            scores.append(score_dict)

        return {k: np.mean([s[k] for s in scores]) for k in criteria}


METRIC_MAPPING = {
    "accuracy": Accuracy,
    "bertscore": BertScore,
    "bleurt": Bleurt,
    "llm-rouge": LLMRouge,
    "llm-as-a-judge": LLMJudge,
    "rouge": Rouge,
}


class AutoMetric:
    def __init__(self):
        raise EnvironmentError(
            "This class is designed to be instantiated only through the from_name method"
        )

    def from_name(metric_name, **kwargs):
        if metric_name not in METRIC_MAPPING:
            raise ValueError(f"Invalid metric name: {metric_name}")
        return METRIC_MAPPING[metric_name](**kwargs)


if __name__ == "__main__":
    metric = AutoMetric.from_name("llm-as-a-judge")
    predictions = [
        "The answer to 2x2 is 4.",
        "The answer to 2x2 is 5.",
    ]
    labels = [["4"], ["4"]]
    prompts = [
        "What is 2x2?",
        "What is 2x2?",
    ]
    print(metric.compute(prompts=prompts, predictions=predictions, labels=None))
