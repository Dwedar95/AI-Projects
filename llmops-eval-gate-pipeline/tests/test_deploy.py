"""Syntactic and consistency checks of the deployment files.

These tests parse the YAML and cross-check references. They do not prove the
manifests work on a cluster.
"""

import json

import yaml
from fastapi.routing import APIRoute

from llmops_gate.cost import TokenBudget
from llmops_gate.registry import Registry
from llmops_gate.service import create_app

from .conftest import ROOT

K8S = ROOT / "k8s"


def _load(name):
    return yaml.safe_load((K8S / name).read_text())


def test_all_manifests_parse_and_are_listed_in_kustomization():
    resources = _load("kustomization.yaml")["resources"]
    manifests = sorted(p.name for p in K8S.glob("*.yaml") if p.name != "kustomization.yaml")
    assert sorted(resources) == manifests
    kinds = {_load(name)["kind"] for name in resources}
    expected = {"ConfigMap", "PersistentVolumeClaim", "Deployment", "Service"}
    assert kinds == expected | {"HorizontalPodAutoscaler"}


def test_selectors_ports_and_references_are_consistent():
    deployment, service, hpa = _load("deployment.yaml"), _load("service.yaml"), _load("hpa.yaml")
    pod = deployment["spec"]["template"]
    assert deployment["spec"]["selector"]["matchLabels"] == pod["metadata"]["labels"]
    assert service["spec"]["selector"] == pod["metadata"]["labels"]
    container = pod["spec"]["containers"][0]
    port_names = {p["name"] for p in container["ports"]}
    assert service["spec"]["ports"][0]["targetPort"] in port_names
    assert hpa["spec"]["scaleTargetRef"]["name"] == deployment["metadata"]["name"]
    assert hpa["spec"]["minReplicas"] <= hpa["spec"]["maxReplicas"]
    volumes = {v["name"] for v in pod["spec"]["volumes"]}
    assert {m["name"] for m in container["volumeMounts"]} == volumes
    config_name = _load("configmap.yaml")["metadata"]["name"]
    assert container["envFrom"][0]["configMapRef"]["name"] == config_name
    claim = pod["spec"]["volumes"][0]["persistentVolumeClaim"]["claimName"]
    assert claim == _load("pvc.yaml")["metadata"]["name"]
    assert container["securityContext"]["readOnlyRootFilesystem"] is True
    assert set(container["resources"]) == {"requests", "limits"}


def test_probes_point_at_routes_the_service_really_has(tmp_path):
    app = create_app(Registry(tmp_path))
    routes = {r.path for r in app.routes if isinstance(r, APIRoute)}
    container = _load("deployment.yaml")["spec"]["template"]["spec"]["containers"][0]
    assert container["readinessProbe"]["httpGet"]["path"] in routes
    assert container["livenessProbe"]["httpGet"]["path"] in routes


def test_configmap_budgets_are_valid_for_the_service():
    data = _load("configmap.yaml")["data"]
    budgets = {
        k: TokenBudget.model_validate(v) for k, v in json.loads(data["budgets.json"]).items()
    }
    assert "default" in budgets
    assert data["LLMOPS_BUDGETS"].endswith("budgets.json")


def test_workflows_parse_and_reference_existing_files():
    for name in ("ci.yml", "eval-gate.yml"):
        workflow = yaml.safe_load((ROOT / ".github" / "workflows" / name).read_text())
        assert "jobs" in workflow
    text = (ROOT / ".github" / "workflows" / "eval-gate.yml").read_text()
    for path in (
        "examples/train_intent_model.py",
        "benchmarks/intent_benchmark.jsonl",
        "benchmarks/gates.yaml",
        "benchmarks/baseline_metrics.json",
    ):
        assert path in text and (ROOT / path).exists()
    dockerfile = (ROOT / "Dockerfile").read_text()
    assert "llmops_gate.service:app" in dockerfile and "USER 10001" in dockerfile
