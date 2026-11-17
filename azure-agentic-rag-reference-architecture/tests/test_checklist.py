import pytest

from archkit.checklist import RuleError, Status, assess, evaluate, load_architecture, load_rules, lookup, parse_rules


def rule(rule_id, check, pillar="security", severity="high", applies_when=None):
    item = {"id": rule_id, "pillar": pillar, "severity": severity, "title": rule_id, "check": check}
    if applies_when:
        item["applies_when"] = applies_when
    return item


def test_lookup_dotted_paths_and_list_indexes():
    data = {"a": {"b": [{"c": 1}, {"c": 2}]}}
    assert lookup(data, "a.b.1.c") == 2
    assert evaluate({"path": "a.b.5.c", "exists": False}, data) is True


@pytest.mark.parametrize(
    ("condition", "expected"),
    [
        ({"path": "x", "op": "eq", "value": 1}, True),
        ({"path": "flag", "op": "eq", "value": 1}, False),  # True is not 1
        ({"path": "x", "op": "gte", "value": 1}, True),
        ({"path": "name", "op": "gte", "value": 1}, False),  # no numeric comparison on strings
        ({"path": "name", "op": "in", "value": ["a", "b"]}, True),
        ({"path": "tags", "op": "contains_all", "value": ["t1", "t2"]}, True),
        ({"path": "tags", "op": "contains_all", "value": ["t1", "t9"]}, False),
        ({"path": "tags", "op": "len_gte", "value": 3}, False),
        ({"path": "missing", "op": "eq", "value": 1}, None),
        ({"path": "missing", "exists": True}, False),
        ({"not": {"path": "flag", "op": "eq", "value": True}}, False),
    ],
)
def test_leaf_operators(condition, expected):
    data = {"x": 1, "flag": True, "name": "a", "tags": ["t1", "t2"]}
    assert evaluate(condition, data) is expected


def test_three_valued_combinators():
    data = {"yes": True, "no": False}
    t = {"path": "yes", "op": "eq", "value": True}
    f = {"path": "no", "op": "eq", "value": True}
    u = {"path": "absent", "op": "eq", "value": True}
    assert evaluate({"all": [t, u]}, data) is None
    assert evaluate({"all": [f, u]}, data) is False
    assert evaluate({"any": [t, u]}, data) is True
    assert evaluate({"any": [f, u]}, data) is None
    assert evaluate({"not": u}, data) is None


def test_quantifiers_over_lists():
    data = {"models": [{"filter": True}, {"filter": False}], "empty": []}
    inner = {"path": "filter", "op": "eq", "value": True}
    assert evaluate({"path": "models", "every": inner}, data) is False
    assert evaluate({"path": "models", "some": inner}, data) is True
    assert evaluate({"path": "empty", "every": inner}, data) is False
    assert evaluate({"path": "models", "every": {"path": "absent", "op": "eq", "value": 1}}, data) is None


def test_invalid_conditions_and_rules_raise():
    with pytest.raises(RuleError, match="unknown operator"):
        evaluate({"path": "x", "op": "matches", "value": 1}, {"x": 1})
    with pytest.raises(RuleError, match="needs 'path'"):
        evaluate({"op": "eq"}, {})
    with pytest.raises(RuleError, match="unknown pillar"):
        parse_rules([rule("A", {"path": "x", "exists": True}, pillar="speed")])
    with pytest.raises(RuleError, match="duplicate"):
        parse_rules([rule("A", {"path": "x", "exists": True})] * 2)
    with pytest.raises(RuleError, match="missing"):
        parse_rules([{"id": "A", "pillar": "cost"}])


def test_scoring_is_severity_weighted():
    rules = parse_rules(
        [
            rule("A", {"path": "a", "op": "eq", "value": True}, severity="critical"),
            rule("B", {"path": "b", "op": "eq", "value": True}, severity="low"),
            rule("C", {"path": "c", "op": "eq", "value": True}, pillar="cost", severity="medium"),
        ]
    )
    result = assess({"a": True, "b": False, "c": True}, rules)
    scores = {p.pillar: p.score for p in result.pillars}
    assert scores == {"security": pytest.approx(100 * 5 / 6), "cost": 100.0}
    assert result.overall == pytest.approx(100 * 7 / 8)
    assert result.passed


def test_critical_gap_fails_the_gate_even_with_a_high_score():
    rules = parse_rules(
        [rule("CRIT", {"path": "a", "op": "eq", "value": True}, severity="critical")]
        + [rule(f"R{i}", {"path": "ok", "op": "eq", "value": True}) for i in range(20)]
    )
    result = assess({"a": False, "ok": True}, rules)
    assert result.overall > 90
    assert not result.passed
    assert [g.rule.id for g in result.critical_gaps] == ["CRIT"]


def test_pillar_floor_fails_the_gate():
    rules = parse_rules(
        [rule(f"S{i}", {"path": "ok", "op": "eq", "value": True}) for i in range(10)]
        + [rule("C1", {"path": "bad", "op": "eq", "value": True}, pillar="cost", severity="low")]
    )
    result = assess({"ok": True, "bad": False}, rules)
    assert result.overall > 80 and not result.passed


def test_not_applicable_rules_are_excluded_and_unknown_counts_as_gap():
    rules = parse_rules(
        [
            rule(
                "NA",
                {"path": "x", "op": "eq", "value": True},
                applies_when={"path": "env", "op": "eq", "value": "prod"},
            ),
            rule("UNK", {"path": "undocumented", "op": "eq", "value": True}),
        ]
    )
    result = assess({"env": "dev"}, rules)
    statuses = {r.rule.id: r.status for r in result.results}
    assert statuses == {"NA": Status.NOT_APPLICABLE, "UNK": Status.UNKNOWN}
    assert result.pillars[0].applicable == 1 and result.overall == 0
    assert "not documented" in result.to_markdown()


def test_bundled_rules_cover_all_six_pillars():
    rules = load_rules()
    assert len(rules) == 44
    assert {r.pillar for r in rules} == {
        "security",
        "reliability",
        "cost",
        "operations",
        "performance",
        "responsible_ai",
    }
    assert all(r.recommendation for r in rules)


def test_reference_and_weak_examples(repo_root):
    reference = assess(load_architecture(repo_root / "examples" / "architecture.yaml"))
    weak = assess(load_architecture(repo_root / "examples" / "architecture-weak.yaml"))
    assert reference.passed and not reference.critical_gaps
    assert {g.rule.id for g in reference.gaps} == {"PERF-01", "RAI-07", "REL-06", "OPS-07"}
    assert not weak.passed and len(weak.critical_gaps) == 3
    assert weak.overall < reference.overall
    assert reference.to_dict()["overall"] == round(reference.overall, 1)
