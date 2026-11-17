from archkit.bicep import validate_directory
from archkit.bicep.policy import default_rules

SECURE_STORAGE = """
param workspaceId string
param tags object

resource account 'Microsoft.Storage/storageAccounts@2023-05-01' = {
  name: 'stsecure'
  location: 'westeurope'
  tags: tags
  properties: {
    publicNetworkAccess: 'Disabled'
    allowBlobPublicAccess: false
    allowSharedKeyAccess: false
    minimumTlsVersion: 'TLS1_2'
    supportsHttpsTrafficOnly: true
  }
}

module pe 'private-endpoint.bicep' = {
  name: 'pe'
  params: {
    targetResourceId: account.id
  }
}

resource diag 'Microsoft.Insights/diagnosticSettings@2021-05-01-preview' = {
  name: 'diag'
  scope: account
  properties: {
    workspaceId: workspaceId
    metrics: [
      { category: 'Transaction', enabled: true }
    ]
  }
}
"""


def rule_ids(report):
    return sorted({f.rule_id for f in report.findings})


def test_secure_storage_passes(write_bicep):
    report = validate_directory(write_bicep(SECURE_STORAGE))
    assert report.passed
    assert report.findings == []
    assert report.resources_checked == 2


def test_public_access_enabled_is_an_error(write_bicep):
    report = validate_directory(write_bicep(SECURE_STORAGE.replace("'Disabled'", "'Enabled'")))
    assert not report.passed
    assert [f.rule_id for f in report.errors] == ["NET-001"]
    assert "'Enabled'" in report.errors[0].message


def test_missing_property_is_an_error_not_a_pass(write_bicep):
    source = SECURE_STORAGE.replace("    publicNetworkAccess: 'Disabled'\n", "")
    report = validate_directory(write_bicep(source))
    assert any(f.rule_id == "NET-001" and "not set" in f.message for f in report.errors)


def test_expression_value_is_a_warning(write_bicep):
    source = SECURE_STORAGE.replace(
        "publicNetworkAccess: 'Disabled'", "publicNetworkAccess: allowPublic ? 'Enabled' : 'Disabled'"
    )
    report = validate_directory(write_bicep(source))
    assert report.passed
    assert [f.rule_id for f in report.warnings] == ["NET-001"]


def test_missing_private_endpoint_diagnostics_and_tags(write_bicep):
    source = """
resource search 'Microsoft.Search/searchServices@2023-11-01' = {
  name: 'srch'
  location: 'westeurope'
  properties: {
    publicNetworkAccess: 'disabled'
    disableLocalAuth: true
  }
}
"""
    report = validate_directory(write_bicep(source))
    assert rule_ids(report) == ["GOV-001", "IAM-001", "NET-002", "OPS-001"]


def test_keyless_auth_rule(write_bicep):
    report = validate_directory(
        write_bicep(SECURE_STORAGE.replace("allowSharedKeyAccess: false", "allowSharedKeyAccess: true"))
    )
    assert rule_ids(report) == ["IAM-002"]


def test_identity_type_none_is_rejected(write_bicep):
    source = """
resource app 'Microsoft.App/containerApps@2024-03-01' = {
  name: 'app'
  location: 'westeurope'
  tags: {}
  identity: {
    type: 'None'
  }
  properties: {
    configuration: {
      ingress: {
        external: false
        allowInsecure: false
      }
    }
  }
}
"""
    report = validate_directory(write_bicep(source))
    assert rule_ids(report) == ["IAM-001"]


def test_privileged_role_resolved_through_variable(write_bicep):
    source = """
param principalId string
var roles = {
  contributor: 'b24988ac-6180-42a0-ab88-20f7382dd24c'
  reader: 'acdd72a7-3385-48ef-bd42-f606fba81ae7'
}
resource bad 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(principalId, roles.contributor)
  properties: {
    principalId: principalId
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', roles.contributor)
  }
}
resource good 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(principalId, roles.reader)
  properties: {
    principalId: principalId
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', roles.reader)
  }
}
"""
    report = validate_directory(write_bicep(source))
    assert len(report.errors) == 1
    assert report.errors[0].rule_id == "IAM-003"
    assert "Contributor" in report.errors[0].message and report.errors[0].target.startswith("bad")


def test_secret_parameters_and_outputs(write_bicep):
    source = """
param sqlPassword string
@secure()
param apiToken string
param keyVaultName string
resource account 'Microsoft.Storage/storageAccounts@2023-05-01' existing = {
  name: 'st'
}
output key string = account.listKeys().keys[0].value
output token string = apiToken
output name string = keyVaultName
"""
    report = validate_directory(write_bicep(source))
    findings = {(f.rule_id, f.target) for f in report.errors}
    assert findings == {
        ("SEC-001", "param sqlPassword"),
        ("SEC-002", "output key"),
        ("SEC-002", "output token"),
    }


def test_required_tags_rule_is_configurable(write_bicep):
    directory = write_bicep("var tags = {\n  environment: 'dev'\n  owner: 'team'\n}\n")
    assert rule_ids(validate_directory(directory)) == ["GOV-002"]
    assert validate_directory(directory, default_rules(["environment", "owner"])).passed


def test_suppression_comment_needs_matching_rule_id(write_bicep):
    suppressed = SECURE_STORAGE.replace("'Disabled'", "'Enabled'").replace(
        "resource account", "// archkit-ignore: NET-001 public access approved in ticket 123\nresource account"
    )
    report = validate_directory(write_bicep(suppressed))
    assert report.passed and len(report.suppressed) == 1
    wrong_id = suppressed.replace("NET-001 public", "NET-002 public")
    assert not validate_directory(write_bicep(wrong_id)).passed


def test_existing_resources_are_not_checked(write_bicep):
    source = "resource kv 'Microsoft.KeyVault/vaults@2023-07-01' existing = {\n  name: 'kv'\n}\n"
    report = validate_directory(write_bicep(source))
    assert report.passed and report.resources_checked == 0


def test_repository_infra_passes_all_rules(repo_root):
    report = validate_directory(repo_root / "infra")
    assert report.passed
    assert report.findings == []
    assert len(report.files) == 12
    assert len(report.rules) == 11


def test_report_serialisation(write_bicep):
    report = validate_directory(write_bicep(SECURE_STORAGE.replace("'Disabled'", "'Enabled'")))
    payload = report.to_dict()
    assert payload["passed"] is False and payload["errors"] == 1
    markdown = report.to_markdown()
    assert "Result: FAIL" in markdown and "| NET-001 | error |" in markdown
