# SPDX-License-Identifier: Apache-2.0
"""Detecting a model whose weight files are not all there.

The case that motivated this listed as a 4.29 GB model, transferred to a peer
intact, and only failed at load with `Missing 1621 parameters` - six layers
away from the cause, which was that 120 of its 126 files had never arrived.
"""

from __future__ import annotations

import json

from omlx import model_integrity


def _shards(directory, present, expected):
    directory.mkdir(parents=True, exist_ok=True)
    for i in range(1, present + 1):
        (directory / f"model-{i:05d}-of-{expected:05d}.safetensors").write_bytes(b"x")
    return str(directory)


def test_a_partial_download_is_detected_from_the_filenames(tmp_path):
    path = _shards(tmp_path / "partial", present=6, expected=126)

    result = model_integrity.describe(path)

    assert result["complete"] is False
    assert (result["present"], result["expected"]) == (6, 126)
    assert "6 of 126" in result["detail"]


def test_a_complete_model_says_nothing(tmp_path):
    path = _shards(tmp_path / "whole", present=4, expected=4)

    result = model_integrity.describe(path)

    assert result["complete"] is True
    assert result["detail"] == ""


def test_the_index_is_preferred_over_the_filename(tmp_path):
    """The index names every shard the loader's weight map needs, which is the
    definition the loader itself works from."""
    directory = tmp_path / "indexed"
    _shards(directory, present=2, expected=2)
    (directory / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {f"w{i}": f"shard-{i}.safetensors" for i in range(5)}})
    )

    result = model_integrity.census(str(directory))

    assert result["expected"] == 5
    assert result["complete"] is False


def test_a_shard_in_a_subdirectory_still_counts(tmp_path):
    """A real false positive: `optiq/optiq_vision.safetensors` is referenced by
    the index but invisible to a top-level glob, so two complete models were
    reported one file short."""
    directory = tmp_path / "nested"
    _shards(directory, present=4, expected=4)
    (directory / "optiq").mkdir()
    (directory / "optiq" / "optiq_vision.safetensors").write_bytes(b"x")
    (directory / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {
            **{f"w{i}": f"model-{i:05d}-of-00004.safetensors" for i in range(1, 5)},
            "vision": "optiq/optiq_vision.safetensors",
        }})
    )

    result = model_integrity.describe(str(directory))

    assert result["expected"] == 5
    assert result["present"] == 5
    assert result["complete"] is True
    assert result["detail"] == ""


def test_an_index_naming_a_file_that_is_not_there_is_incomplete(tmp_path):
    directory = tmp_path / "missing-one"
    _shards(directory, present=3, expected=4)
    (directory / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {
            f"w{i}": f"model-{i:05d}-of-00004.safetensors" for i in range(1, 5)
        }})
    )

    result = model_integrity.describe(str(directory))

    assert (result["present"], result["expected"]) == (3, 4)
    assert result["complete"] is False


def test_a_model_with_no_count_in_its_names_is_not_flagged(tmp_path):
    """Guessing here would mark every single-file model broken."""
    directory = tmp_path / "single"
    directory.mkdir()
    (directory / "model.safetensors").write_bytes(b"x")

    result = model_integrity.census(str(directory))

    assert result["complete"] is True
    assert result["expected"] == 1


def test_a_directory_with_no_weights_is_not_a_verdict(tmp_path):
    """`checked` is what keeps a config-only directory from being reported as
    a broken model."""
    directory = tmp_path / "empty"
    directory.mkdir()

    assert model_integrity.census(str(directory))["checked"] is False
    assert model_integrity.census("/nonexistent/path")["checked"] is False


def test_the_answer_is_recomputed_when_the_directory_changes(tmp_path):
    """A finished download has to stop reporting itself as partial, without
    anyone having to invalidate a cache by hand."""
    directory = tmp_path / "growing"
    _shards(directory, present=1, expected=3)
    assert model_integrity.census(str(directory))["complete"] is False

    for i in (2, 3):
        (directory / f"model-{i:05d}-of-00003.safetensors").write_bytes(b"x")

    assert model_integrity.census(str(directory))["complete"] is True
