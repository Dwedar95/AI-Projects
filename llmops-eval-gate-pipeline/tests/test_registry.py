import pytest

from llmops_gate.registry import (
    PromotionBlocked,
    Registry,
    RegistryError,
    Stage,
    render_prompt,
)

PASS = {"passed": True, "checks": []}
FAIL = {
    "passed": False,
    "checks": [{"passed": False, "message": "accuracy 0.6 >= min 0.8"}],
}


def _model(registry: Registry, tmp_path, content: bytes = b"weights-1"):
    artifact = tmp_path / "model.onnx"
    artifact.write_bytes(content)
    return registry.register_model("clf", artifact, {"owner": "ml-platform"})


def test_model_versions_increment_and_are_content_addressed(registry, tmp_path):
    first = _model(registry, tmp_path, b"weights-1")
    second = _model(registry, tmp_path, b"weights-2")
    assert (first.version, second.version) == (1, 2)
    assert first.sha256 != second.sha256 and len(first.sha256) == 64
    assert first.metadata == {"size_bytes": 9, "owner": "ml-platform"}
    assert registry.artifact_path(first).read_bytes() == b"weights-1"
    assert [e.version for e in registry.list("model", "clf")] == [1, 2]
    assert registry.get("model", "clf").version == 2
    assert registry.names("model") == ["clf"]


def test_artifact_tampering_is_detected(registry, tmp_path):
    entry = _model(registry, tmp_path)
    assert registry.verify_artifact(entry)
    registry.artifact_path(entry).write_bytes(b"tampered")
    assert not registry.verify_artifact(entry)


def test_prompt_registration_extracts_variables_and_rejects_duplicates(registry):
    entry = registry.register_prompt("triage", "Classify {ticket} for {team}.")
    assert entry.variables == ["team", "ticket"] and entry.version == 1
    with pytest.raises(RegistryError, match="identical"):
        registry.register_prompt("triage", "Classify {ticket} for {team}.")
    assert registry.register_prompt("triage", "Classify {ticket}.").version == 2
    with pytest.raises(RegistryError, match="empty"):
        registry.register_prompt("other", "   ")


def test_render_prompt_requires_exact_variables():
    assert render_prompt("Hi {name}, {name}!", name="Ada") == "Hi Ada, Ada!"
    with pytest.raises(RegistryError, match=r"missing \['name'\]"):
        render_prompt("Hi {name}")
    with pytest.raises(RegistryError, match="unexpected"):
        render_prompt("Hi", name="Ada")


@pytest.mark.parametrize("name", ["Bad Name", "../escape", "", "UPPER"])
def test_invalid_names_are_rejected(registry, name):
    with pytest.raises(RegistryError, match="invalid name"):
        registry.register_prompt(name, "x {y}")


def test_missing_entries_raise(registry, tmp_path):
    with pytest.raises(RegistryError, match="no versions"):
        registry.get("model", "clf")
    _model(registry, tmp_path)
    with pytest.raises(RegistryError, match="does not exist"):
        registry.get("model", "clf", 7)
    with pytest.raises(RegistryError, match="does not exist"):
        registry.register_model("clf", tmp_path / "missing.onnx")


def test_promotion_requires_a_passing_gate(registry, tmp_path):
    entry = _model(registry, tmp_path)
    with pytest.raises(PromotionBlocked, match="has not been evaluated"):
        registry.promote("model", "clf", 1, Stage.STAGING)
    registry.record_evaluation(entry, {"accuracy": 0.6}, FAIL)
    with pytest.raises(PromotionBlocked, match="accuracy 0.6 >= min 0.8"):
        registry.promote("model", "clf", 1, Stage.STAGING)
    registry.record_evaluation(entry, {"accuracy": 0.9}, PASS)
    assert registry.promote("model", "clf", 1, Stage.STAGING).stage is Stage.STAGING
    assert registry.get("model", "clf", 1).metrics == {"accuracy": 0.9}


def test_transitions_are_validated(registry, tmp_path):
    entry = _model(registry, tmp_path)
    registry.record_evaluation(entry, {}, PASS)
    with pytest.raises(RegistryError, match="none -> production is not allowed"):
        registry.promote("model", "clf", 1, Stage.PRODUCTION)


def test_new_production_version_archives_the_previous_one(registry, tmp_path):
    for content in (b"a", b"b"):
        entry = _model(registry, tmp_path, content)
        registry.record_evaluation(entry, {}, PASS)
        registry.promote("model", "clf", entry.version, Stage.STAGING)
        registry.promote("model", "clf", entry.version, Stage.PRODUCTION)
    assert registry.get("model", "clf", 1).stage is Stage.ARCHIVED
    assert registry.in_stage("model", "clf", Stage.PRODUCTION).version == 2


def test_override_is_possible_and_recorded(registry, tmp_path):
    entry = _model(registry, tmp_path)
    registry.record_evaluation(entry, {}, FAIL)
    registry.promote("model", "clf", 1, Stage.STAGING, actor="oncall", override_reason="hotfix")
    last = registry.history()[-1]
    assert last["override_reason"] == "hotfix" and last["actor"] == "oncall"
    assert last["from"] == "none" and last["to"] == "staging"


def test_rollback_restores_the_previous_production_version(registry, tmp_path):
    with pytest.raises(RegistryError, match="no earlier production"):
        registry.rollback("model", "clf")
    for content in (b"a", b"b"):
        entry = _model(registry, tmp_path, content)
        registry.record_evaluation(entry, {}, PASS)
        registry.promote("model", "clf", entry.version, Stage.STAGING)
        registry.promote("model", "clf", entry.version, Stage.PRODUCTION)
    restored = registry.rollback("model", "clf", actor="oncall")
    assert restored.version == 1 and restored.stage is Stage.PRODUCTION
    assert registry.get("model", "clf", 2).stage is Stage.ARCHIVED
    events = [h["event"] for h in registry.history()]
    assert events.count("registered") == 2 and events.count("evaluated") == 2
