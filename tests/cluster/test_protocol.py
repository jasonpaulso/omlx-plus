# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the cluster wire protocol (JSON only, closed command schema)."""

from __future__ import annotations

import json
import pickle

import pytest

from omlx.cluster.protocol import (
    PHASE_ADMIT,
    PHASE_BATCHED,
    PHASE_STANDALONE,
    PROTOCOL_VERSION,
    Backend,
    CommandKind,
    GenerationSpec,
    ProtocolError,
    RankOp,
    SpawnRankCommand,
    StepMessage,
    StopTextBuffer,
    command_to_wire,
    parse_command,
)

# -- GenerationSpec ----------------------------------------------------------


def test_generation_spec_round_trip():
    spec = GenerationSpec(
        prompt_ids=[1, 2, 3],
        max_tokens=16,
        temperature=0.7,
        stop=["</s>"],
        stop_token_ids=[2],
        request_id="req-1",
    )
    restored = GenerationSpec.from_dict(spec.to_dict())
    assert restored == spec


def test_generation_spec_ignores_unknown_keys():
    spec = GenerationSpec.from_dict(
        {"prompt_ids": [5], "max_tokens": 4, "bogus": "ignored"}
    )
    assert spec.prompt_ids == [5]
    assert spec.max_tokens == 4


# -- StepMessage (JSON only, pickle rejected) --------------------------------


def test_step_message_round_trip():
    message = StepMessage(
        step=3,
        tokens={"req": 42},
        deltas=[{"op": "finish", "reason": "stop"}],
        done=True,
    )
    restored = StepMessage.from_json_bytes(message.to_json_bytes())
    assert restored == message


def test_step_message_rejects_pickle_frame():
    # An on-link attacker who can inject into the ring must not get code
    # execution: a pickle frame is rejected, not unpickled (D4).
    hostile = pickle.dumps({"step": 0, "tokens": {}})
    with pytest.raises(ProtocolError):
        StepMessage.from_json_bytes(hostile)


def test_step_message_rejects_non_json_bytes():
    with pytest.raises(ProtocolError):
        StepMessage.from_json_bytes(b"\x80\x04not json")


def test_step_message_rejects_wrong_shape():
    # Valid JSON but missing required fields.
    with pytest.raises(ProtocolError):
        StepMessage.from_json_bytes(json.dumps({"nope": 1}).encode("utf-8"))


# -- RankOp (S3 forward-replay message) --------------------------------------


def test_rank_op_round_trip_standalone():
    op = RankOp(
        tags=[7],
        token_ids=[[1, 2, 3]],
        release=[3],
        phase=PHASE_STANDALONE,
    )
    restored = RankOp.from_dict(op.to_dict())
    assert restored == op


def test_rank_op_round_trip_batched():
    op = RankOp(
        tags=[1, 2, 3],
        token_ids=[[10], [11], [12]],
        release=[],
        phase=PHASE_BATCHED,
    )
    restored = RankOp.from_dict(op.to_dict())
    assert restored == op


def test_rank_op_round_trip_admit():
    op = RankOp(
        tags=[4],
        token_ids=[[9]],
        release=[],
        phase=PHASE_ADMIT,
    )
    restored = RankOp.from_dict(op.to_dict())
    assert restored == op


def test_rank_op_default_phase_is_batched():
    op = RankOp(tags=[1], token_ids=[[5]])
    assert op.phase == PHASE_BATCHED
    assert RankOp.from_dict(op.to_dict()).phase == PHASE_BATCHED


def test_rank_op_rejects_non_object():
    with pytest.raises(ProtocolError):
        RankOp.from_dict(["not", "a", "dict"])


def test_rank_op_rejects_unknown_op_kind():
    with pytest.raises(ProtocolError):
        RankOp.from_dict(
            {"op": "reset", "tags": [1], "token_ids": [[1]], "phase": PHASE_BATCHED}
        )


def test_rank_op_rejects_unknown_phase():
    with pytest.raises(ProtocolError):
        RankOp.from_dict(
            {"op": "forward", "tags": [1], "token_ids": [[1]], "phase": "sideways"}
        )


def test_rank_op_rejects_unknown_field():
    with pytest.raises(ProtocolError):
        RankOp.from_dict(
            {
                "op": "forward",
                "tags": [1],
                "token_ids": [[1]],
                "phase": PHASE_BATCHED,
                "cwd": "/etc",
            }
        )


def test_rank_op_rejects_missing_required_fields():
    with pytest.raises(ProtocolError):
        RankOp.from_dict({"op": "forward", "phase": PHASE_BATCHED})


def test_rank_op_rejects_tags_token_ids_length_mismatch():
    with pytest.raises(ProtocolError):
        RankOp.from_dict(
            {
                "op": "forward",
                "phase": PHASE_BATCHED,
                "tags": [1, 2],
                "token_ids": [[1]],
            }
        )


def test_rank_op_rejects_standalone_with_multiple_tags():
    with pytest.raises(ProtocolError):
        RankOp.from_dict(
            {
                "op": "forward",
                "phase": PHASE_STANDALONE,
                "tags": [1, 2],
                "token_ids": [[1], [2]],
            }
        )


def test_rank_op_rejects_admit_with_multiple_tags():
    with pytest.raises(ProtocolError):
        RankOp.from_dict(
            {
                "op": "forward",
                "phase": PHASE_ADMIT,
                "tags": [1, 2],
                "token_ids": [[1], [2]],
            }
        )


def test_rank_op_rejects_wrong_field_types():
    with pytest.raises(ProtocolError):
        RankOp.from_dict(
            {
                "op": "forward",
                "phase": PHASE_BATCHED,
                "tags": ["not-an-int"],
                "token_ids": [[1]],
            }
        )


# -- closed command schema (CL2-04) ------------------------------------------


def _spawn_payload(**overrides):
    payload = {
        "kind": "spawn_rank",
        "schema_version": PROTOCOL_VERSION,
        "job_id": "job-1",
        "step": 0,
        "rank": 1,
        "world_size": 2,
        "backend": "ring",
        "model_id": "mlx-community/Llama-3.2-1B-Instruct-4bit",
        "peers": ["10.0.2.1", "10.0.2.2"],
        "base_port": 41100,
        "seed": 0,
    }
    payload.update(overrides)
    return payload


def test_parse_spawn_rank_command():
    command = parse_command(_spawn_payload())
    assert isinstance(command, SpawnRankCommand)
    assert command.kind is CommandKind.SPAWN_RANK
    assert command.backend is Backend.RING
    assert command.rank == 1
    assert command.model_id.endswith("Llama-3.2-1B-Instruct-4bit")


def test_spawn_rank_ring_omits_ibv_devices():
    command = parse_command(_spawn_payload())
    assert command.ibv_devices is None
    # The wire dict round-trips (ring carries an explicit null matrix).
    wire = command_to_wire(command)
    assert wire["ibv_devices"] is None
    assert parse_command(wire).ibv_devices is None


def test_spawn_rank_jaccl_carries_ibv_matrix():
    matrix = [[None, "rdma_en2"], ["rdma_en4", None]]
    command = parse_command(_spawn_payload(backend="jaccl", ibv_devices=matrix))
    assert command.backend is Backend.JACCL
    assert command.ibv_devices == matrix
    # Survives a wire round-trip unchanged.
    assert command_to_wire(command)["ibv_devices"] == matrix
    assert parse_command(command_to_wire(command)).ibv_devices == matrix


def test_parse_sweep_and_teardown_and_presence():
    base = {"schema_version": PROTOCOL_VERSION, "job_id": "j", "step": 1}
    assert parse_command({**base, "kind": "sweep"}).kind is CommandKind.SWEEP
    assert parse_command({**base, "kind": "teardown"}).kind is CommandKind.TEARDOWN
    presence = parse_command({**base, "kind": "presence", "model_id": "m"})
    assert presence.kind is CommandKind.PRESENCE


# -- S5: TRANSFER_START/TRANSFER_ROUND/TRANSFER_ABORT ------------------------


def _transfer_start_payload(**over):
    base = {
        "kind": "transfer_start",
        "schema_version": PROTOCOL_VERSION,
        "job_id": "j",
        "step": 1,
        "model_id": "m",
        "manifest": [{"relative_path": "a.json", "size": 1, "sha256": "0" * 64}],
        "source": "peer",
        "epoch": "ep1",
    }
    base.update(over)
    return base


def test_parse_transfer_start_round_trip():
    command = parse_command(_transfer_start_payload())
    assert command.kind is CommandKind.TRANSFER_START
    assert command.repair is False
    assert command.hf_repo_id is None
    assert command.hf_revision is None
    wire = command_to_wire(command)
    assert "hf_token" not in wire
    assert "endpoint" not in wire


def test_transfer_start_never_accepts_hf_token_or_endpoint():
    # extra="forbid": these fields were never declared, so an attempt to
    # smuggle them in is a rejected command, not a silently-accepted one.
    with pytest.raises(ProtocolError):
        parse_command(_transfer_start_payload(hf_token="secret"))
    with pytest.raises(ProtocolError):
        parse_command(_transfer_start_payload(endpoint="https://evil.example"))


def test_parse_transfer_round_and_abort():
    round_cmd = parse_command(
        {
            "kind": "transfer_round",
            "schema_version": PROTOCOL_VERSION,
            "job_id": "j",
            "step": 2,
            "subset": ["a.json"],
            "peers": ["10.0.2.1", "10.0.2.2"],
            "base_port": 41164,
        }
    )
    assert round_cmd.kind is CommandKind.TRANSFER_ROUND

    abort_cmd = parse_command(
        {
            "kind": "transfer_abort",
            "schema_version": PROTOCOL_VERSION,
            "job_id": "j",
            "step": 3,
        }
    )
    assert abort_cmd.kind is CommandKind.TRANSFER_ABORT


def test_transfer_start_source_is_a_closed_enum():
    with pytest.raises(ProtocolError):
        parse_command(_transfer_start_payload(source="ftp"))


def test_transfer_round_base_port_is_bounded():
    with pytest.raises(ProtocolError):
        parse_command(
            {
                "kind": "transfer_round",
                "schema_version": PROTOCOL_VERSION,
                "job_id": "j",
                "step": 2,
                "subset": [],
                "peers": ["10.0.2.1", "10.0.2.2"],
                "base_port": 70000,
            }
        )


def test_parse_command_rejects_unknown_kind():
    with pytest.raises(ProtocolError):
        parse_command(
            {
                "kind": "exec_shell",
                "schema_version": PROTOCOL_VERSION,
                "job_id": "j",
                "step": 0,
            }
        )


def test_parse_command_rejects_unknown_field():
    # extra="forbid": an unexpected field is rejected, never ignored.
    with pytest.raises(ProtocolError):
        parse_command(_spawn_payload(cwd="/etc"))


def test_parse_command_rejects_wire_env():
    # CL2-01: no command shape can carry an environment; the schema forbids it.
    with pytest.raises(ProtocolError):
        parse_command(_spawn_payload(env={"PYTHONPATH": "/evil"}))


def test_parse_command_rejects_wire_path():
    # CL2-02: no command shape can carry a filesystem path for the model.
    with pytest.raises(ProtocolError):
        parse_command(_spawn_payload(model_path="/tmp/evil"))


def test_parse_command_rejects_schema_version_skew():
    with pytest.raises(ProtocolError):
        parse_command(_spawn_payload(schema_version=PROTOCOL_VERSION + 1))


def test_parse_command_rejects_non_object():
    with pytest.raises(ProtocolError):
        parse_command(["not", "a", "dict"])


# -- StopTextBuffer straddle handling ----------------------------------------


def test_stop_buffer_passthrough_without_stops():
    buf = StopTextBuffer([])
    assert buf.push("hello ") == "hello "
    assert buf.push("world") == "world"
    assert buf.text == "hello world"


def test_stop_buffer_holds_back_partial_match():
    buf = StopTextBuffer(["<|im_end|>"])
    # "<|im_" could still grow into the stop string, so it is held back.
    emitted = buf.push("hi <|im_")
    assert emitted == "hi "
    assert buf.hit is None


def test_stop_buffer_truncates_on_hit():
    buf = StopTextBuffer(["<|im_end|>"])
    buf.push("answer<|im_")
    tail = buf.push("end|> trailing")
    assert buf.hit == "<|im_end|>"
    assert "trailing" not in buf.text
    assert buf.text == "answer"
    assert tail == ""


def test_stop_buffer_flush_releases_tail():
    buf = StopTextBuffer(["STOP"])
    buf.push("nearly ST")
    # No hit; flush releases the held-back suffix once no more tokens arrive.
    assert buf.flush() == "ST"
    assert buf.text == "nearly ST"
