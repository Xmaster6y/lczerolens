"""Script to generate datasets.

Run with:
```bash
poetry run python -m scripts.make_datasets
```
"""

import random

import jsonlines
import tqdm

from lczerolens import BoardDataset, GameDataset, ModelWrapper
from lczerolens.xai import ConceptDataset, PieceBestLegalMoveConcept

#######################################
# HYPERPARAMETERS
#######################################
make_test_10 = False
make_test_5000 = False
n_history = 7
skip_book_exit = True
skip_first_n = 20
make_tcec_random = False
tcec_random_sample_rate = 0.1
tcec_random_seed = 42
make_test_10_knight = False
make_tcec_knight = False
model_name = "64x6-2018_0627_1913_08_161.onnx"
#######################################


convert_to_boards = []

if make_test_10:
    convert_to_boards.append("test_stockfish_10.jsonl")

if make_test_5000:
    convert_to_boards.append("test_stockfish_5000.jsonl")

for dataset_name in convert_to_boards:
    dataset = GameDataset(f"./assets/{dataset_name}")
    written_boards = 0
    with jsonlines.open(
        f"./assets/{dataset_name.replace('.jsonl', '_boards.jsonl')}", "w"
    ) as writer:
        for game in tqdm.tqdm(dataset.games):
            lines = BoardDataset.preprocess_game(
                game,
                n_history=n_history,
                skip_book_exit=skip_book_exit,
                skip_first_n=skip_first_n,
            )
            writer.write_all(lines)
            written_boards += len(lines)
    print(f"[INFO] Board written: {written_boards}")


if make_tcec_random:
    dataset_name = "TCEC_game_collection.jsonl"
    dataset = GameDataset(f"./assets/{dataset_name}")
    written_boards = 0
    random.seed(tcec_random_seed)
    with jsonlines.open(
        f"./assets/{dataset_name.replace('.jsonl', '_random_boards.jsonl')}",
        "w",
    ) as writer:
        for game in tqdm.tqdm(dataset.games):
            lines = BoardDataset.preprocess_game(
                game,
                n_history=n_history,
                skip_book_exit=skip_book_exit,
                skip_first_n=skip_first_n,
            )
            k = int(tcec_random_sample_rate * len(lines))
            lines = random.sample(lines, k)
            writer.write_all(lines)
            written_boards += len(lines)
    print(f"[INFO] Board written: {written_boards}")

sample_knights = []
if make_test_10_knight:
    sample_knights.append("test_stockfish_10_boards.jsonl")

if make_tcec_knight:
    sample_knights.append("TCEC_game_collection_random_boards.jsonl")

for dataset_name in sample_knights:
    dataset = BoardDataset(f"./assets/{dataset_name}")
    model = ModelWrapper.from_path(f"./assets/{model_name}")
    concept = PieceBestLegalMoveConcept(model, "N")

    concept_dataset = ConceptDataset.from_board_dataset(dataset, concept)
    concept_dataset.label_resample(1)
    concept_dataset.save(
        f"./assets/{dataset_name.replace('.jsonl', '_knight_concept.jsonl')}"
    )
