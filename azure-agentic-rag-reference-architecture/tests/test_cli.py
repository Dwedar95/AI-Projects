import json

from archkit.cli import main


def test_estimate_json_output(repo_root, capsys):
    code = main(["estimate", str(repo_root / "examples" / "workload.yaml"), "--format", "json"])
    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["name"] == "internal-knowledge-assistant"
    assert set(payload["models"]) == {"chat-small", "chat-large"}


def test_assess_exit_codes_and_output_file(repo_root, tmp_path, capsys):
    target = tmp_path / "report.md"
    assert main(["assess", str(repo_root / "examples" / "architecture.yaml"), "--output", str(target)]) == 0
    assert "Gate: PASS" in target.read_text(encoding="utf-8")
    assert main(["assess", str(repo_root / "examples" / "architecture-weak.yaml")]) == 1
    assert "Gate: FAIL" in capsys.readouterr().out
    assert main(["assess", str(repo_root / "examples" / "architecture.yaml"), "--threshold", "99"]) == 1


def test_validate_bicep_exit_codes(repo_root, tmp_path, capsys):
    assert main(["validate-bicep", str(repo_root / "infra"), "--strict"]) == 0
    (tmp_path / "bad.bicep").write_text(
        "resource kv 'Microsoft.KeyVault/vaults@2023-07-01' = {\n  name: 'kv'\n  location: 'x'\n}\n", encoding="utf-8"
    )
    assert main(["validate-bicep", str(tmp_path)]) == 1
    assert "NET-001" in capsys.readouterr().out


def test_errors_return_exit_code_two(tmp_path, capsys):
    assert main(["validate-bicep", str(tmp_path)]) == 2
    assert "no .bicep files" in capsys.readouterr().err
    (tmp_path / "broken.bicep").write_text("resource x 'T@1' = {\n", encoding="utf-8")
    assert main(["validate-bicep", str(tmp_path)]) == 2
