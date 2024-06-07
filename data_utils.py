import os
import string
from abc import ABC, abstractmethod
from datasets import load_dataset, Dataset, DatasetDict
from tqdm import tqdm
import json


class Benchmark(ABC):
    def __init__(self, name: str):
        self.name = name
        self.dataset = self.download()

    @abstractmethod
    def download(self) -> Dataset | DatasetDict:
        pass

    def get_train(self) -> Dataset:
        return self.dataset["train"]

    def get_test(self) -> Dataset:
        return self.dataset["test"]

    def get_validation(self) -> Dataset:
        return self.dataset["validation"]

    def id(self, ex: dict) -> str:
        return ex["id"]

    def question(self, ex: dict) -> str:
        return ex["question"]

    @abstractmethod
    def instruction(self, ex: dict) -> str:
        pass

    @abstractmethod
    def context(self, ex: dict) -> str:
        return ex["context"]

    @abstractmethod
    def answer(self, ex: dict) -> list[str]:
        return ex["answer"]


class TriviaQA(Benchmark):
    def __init__(self, use_web: bool = False):
        super().__init__("triviaqa")
        self.use_web = use_web

    def download(self) -> Dataset:
        return load_dataset("trivia_qa", "rc")

    def question(self, ex: dict) -> str:
        return ex["question"]

    def id(self, ex: dict) -> str:
        return ex["question_id"]

    def instruction(self) -> str:
        return "Answer the following question."

    def context(self, ex: dict) -> str:
        """
        Create a single string from the entity pages and (optionally) search results.
        TODO: Format uniformly with other datasets and decide on whether to include search results.
        """
        wikis = ex["entity_pages"]
        webs = ex["search_results"]

        wiki_n = len(wikis["title"])
        web_n = len(webs["title"])

        contexts = []

        for i in range(wiki_n):
            contexts.append("# " + wikis["title"][i] + "\n" + wikis["wiki_context"][i])

        if self.use_web:
            for j in range(web_n):
                contexts.append(
                    "# "
                    + webs["title"][j]
                    + "\n"
                    + webs["description"][j]
                    + "\n"
                    + webs["search_context"][j]
                )

        context_str = "\n\n".join(contexts)

        return context_str.strip()

    def answer(self, ex: dict) -> str | list[str]:
        # TriviaQA allows for any answer within a predefined set of aliases
        assert ex["answer"]["value"] in ex["answer"]["aliases"]
        return ex["answer"]["aliases"]


class Quality(Benchmark):
    DATA_URL = (
        "https://github.com/nyu-mll/quality/blob/main/data/v1.0.1/QuALITY.v1.0.1.zip"
    )
    VERSION = "QuALITY.v1.0.1.htmlstripped"

    def __init__(self, cache_dir: str = "/workspace/.cache/datasets"):
        os.makedirs(cache_dir, exist_ok=True)
        self.cache_dir = cache_dir
        super().__init__("quality")

    def dataset_from_fn(self, fn):
        # Load JSONlines file into list of dictionaries
        with open(fn, "r") as fd:
            lines = list(tqdm(map(json.loads, fd)))

        dataset = []
        for line in lines:
            line["article"] = line["article"].strip()
            line["year"] = (
                int(line["year"]) if line["year"] is not None else None
            )  # Inconsistent type in the dataset
            questions = line.pop("questions")

            for q in questions:
                options = q["options"]
                qid = q["question_unique_id"]
                label = q["gold_label"] - 1 if "gold_label" in q else None
                answer = options[label] if label is not None else None

                dataset.append(
                    {
                        "id": qid,
                        "question": q["question"],
                        "label": label,
                        "answer": answer,
                        "options": q["options"],
                        "difficult": q["difficult"],
                        **line,
                    }
                )

        print(f"Processed {len(dataset)} examples for {fn}.")
        return Dataset.from_list(dataset)

    def download(self) -> Dataset:
        # Path to save the downloaded zip file
        data_dir = os.path.join(self.cache_dir, "QuALITY.v1.0.1")
        if not os.path.exists(data_dir):
            raise FileNotFoundError(
                f"Download the data from {Quality.DATA_URL} and unzip the directory to location {data_dir}."
            )

        return DatasetDict(
            {
                "train": self.dataset_from_fn(
                    os.path.join(data_dir, f"{Quality.VERSION}.train")
                ),
                "validation": self.dataset_from_fn(
                    os.path.join(data_dir, f"{Quality.VERSION}.dev")
                ),
                "test": self.dataset_from_fn(
                    os.path.join(data_dir, f"{Quality.VERSION}.test")
                ),
            }
        )

    def question(self, ex: dict) -> str:
        q = ex["question"]
        options = "\n".join(
            [f"{string.ascii_uppercase[i]}) {o}" for i, o in enumerate(ex["options"])]
        )
        return f"{q}\n\n# Options\n{options}"

    def instruction(self) -> str:
        return "Which of the given options best answers the question? Return the uppercase letter corresponding to the correct option."

    def context(self, ex: dict) -> str:
        return ex["article"]

    def answer(self, ex: dict) -> str | list[str]:
        return string.ascii_uppercase[ex["label"]]


BENCHMARKS = {"triviaqa": TriviaQA, "quality": Quality}


if __name__ == "__main__":
    dataset = BENCHMARKS["quality"]()

    ex = dataset.get_train()[0]

    print(dataset.context(ex))
    print(dataset.question(ex))
    print(dataset.answer(ex))
