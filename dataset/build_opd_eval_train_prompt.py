"""Build AIME/AMC evaluation parquets with the DeepMath training prompt format."""

from __future__ import annotations

import argparse
import os
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


SYSTEM_PROMPT = r"Please reason step by step, and put your final answer within \boxed{}."
EXPECTED_ROWS = {"aime24.parquet": 30, "aime25.parquet": 30, "amc23.parquet": 83}


def convert_row(row: Mapping[str, object]) -> dict[str, object]:
    """Move the evaluation instruction from the user suffix into a system message."""
    assert "prompt" in row, "source row is missing prompt"
    prompt = row["prompt"]
    assert isinstance(prompt, list) and len(prompt) == 1, (
        f"source prompt must contain exactly one user message, got {prompt!r}"
    )
    message = prompt[0]
    assert isinstance(message, dict), f"source message must be a dict, got {message!r}"
    assert message == {"role": "user", "content": message["content"]}, (
        f"source prompt must be one role/content user message, got {message!r}"
    )
    content = message["content"]
    assert isinstance(content, str), f"user content must be a string, got {type(content).__name__}"
    instruction_suffix = f" {SYSTEM_PROMPT}"
    assert content.endswith(instruction_suffix), (
        f"user content does not end with the exact instruction suffix: {content!r}"
    )
    question = content[: -len(instruction_suffix)]
    assert question.strip(), "question is empty after removing the instruction suffix"

    converted = deepcopy(dict(row))
    converted["prompt"] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    validate_converted_row(converted)
    return converted


def validate_converted_row(row: Mapping[str, object]) -> None:
    """Fail unless a row uses exactly the DeepMath training prompt structure."""
    assert "prompt" in row, "converted row is missing prompt"
    prompt = row["prompt"]
    assert isinstance(prompt, list) and len(prompt) == 2, f"invalid converted prompt: {prompt!r}"
    assert prompt[0] == {"role": "system", "content": SYSTEM_PROMPT}, (
        f"invalid system message: {prompt[0]!r}"
    )
    user_message = prompt[1]
    assert isinstance(user_message, dict), f"invalid user message: {user_message!r}"
    assert set(user_message) == {"role", "content"} and user_message["role"] == "user", (
        f"invalid user message: {user_message!r}"
    )
    question = user_message["content"]
    assert isinstance(question, str) and question.strip(), f"invalid user question: {question!r}"
    assert SYSTEM_PROMPT not in question, "instruction remains in the converted user message"


def validate_output(source: pa.Table, converted: pa.Table, expected_rows: int) -> None:
    """Exhaustively verify schema, metadata, row count, prompts, and untouched columns."""
    assert expected_rows > 0, f"expected_rows must be positive, got {expected_rows}"
    assert source.num_rows == expected_rows, (
        f"source has {source.num_rows} rows, expected {expected_rows}"
    )
    assert converted.num_rows == expected_rows, (
        f"converted output has {converted.num_rows} rows, expected {expected_rows}"
    )
    assert converted.schema == source.schema, "converted schema or schema metadata changed"
    assert source.column_names == converted.column_names
    for column_name in source.column_names:
        if column_name == "prompt":
            continue
        assert source[column_name].equals(converted[column_name]), (
            f"non-prompt column changed: {column_name}"
        )
    for row in converted.select(["prompt"]).to_pylist():
        validate_converted_row(row)


def convert_file(source_path: Path, output_path: Path, overwrite: bool) -> None:
    """Convert one parquet and atomically publish it after exhaustive validation."""
    assert source_path.is_file(), f"source parquet does not exist: {source_path}"
    assert source_path.name in EXPECTED_ROWS, f"unexpected source filename: {source_path.name}"
    assert not output_path.exists() or overwrite, (
        f"output already exists (pass --overwrite to replace it): {output_path}"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    if temporary_path.exists():
        temporary_path.unlink()

    source = pq.read_table(source_path)
    expected_rows = EXPECTED_ROWS[source_path.name]
    assert source.num_rows == expected_rows, (
        f"{source_path} has {source.num_rows} rows, expected {expected_rows}"
    )
    converted_rows = [convert_row(row) for row in source.to_pylist()]
    converted = pa.Table.from_pylist(converted_rows, schema=source.schema)
    validate_output(source, converted, expected_rows)

    pq.write_table(converted, temporary_path)
    reloaded = pq.read_table(temporary_path)
    validate_output(source, reloaded, expected_rows)
    os.replace(temporary_path, output_path)
    print(f"Wrote and validated {expected_rows} rows: {output_path}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=Path("datasets/opd_paper"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("datasets/opd_paper_train_prompt")
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_dir = args.source_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    assert source_dir.is_dir(), f"source directory does not exist: {source_dir}"
    assert source_dir != output_dir, "source and output directories must differ"
    for filename in EXPECTED_ROWS:
        convert_file(source_dir / filename, output_dir / filename, args.overwrite)


if __name__ == "__main__":
    main()
