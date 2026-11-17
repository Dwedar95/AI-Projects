import pytest

from archkit.bicep import BicepParseError, Expr, ForLoop, parse_bicep, parse_bicep_file
from archkit.bicep.parser import strip_comments

SAMPLE = """
// a comment with 'quotes' and { braces
targetScope = 'resourceGroup'

@description('Region')
param location string = resourceGroup().location

@secure()
param adminPassword string

/* block
   comment */
var tags = {
  environment: 'dev'
  'cost-center': 'CC-1' // trailing comment
}

resource account 'Microsoft.Storage/storageAccounts@2023-05-01' = {
  name: 'st${uniqueString(resourceGroup().id)}'
  location: location
  tags: tags
  sku: { name: 'Standard_LRS' }
  properties: {
    publicNetworkAccess: 'Disabled'
    allowBlobPublicAccess: false
    retention: 14
    url: 'https://example.com/a//b'
    rules: [
      { name: 'a', priority: 100 }
      { name: 'b', priority: 200 }
    ]
  }

  resource blob 'blobServices' = {
    name: 'default'
  }
}

resource vault 'Microsoft.KeyVault/vaults@2023-07-01' existing = {
  name: 'kv'
}

resource optional 'Microsoft.Network/publicIPAddresses@2024-05-01' = if (location == 'westeurope') {
  name: 'pip'
  location: location
}

resource zones 'Microsoft.Network/privateDnsZones@2020-06-01' = [
  for zone in ['a', 'b']: {
    name: zone
    location: 'global'
  }
]

module pe 'modules/private-endpoint.bicep' = {
  name: 'pe'
  params: {
    targetResourceId: account.id
    groupIds: [
      'blob'
    ]
  }
}

output accountId string = account.id
"""


@pytest.fixture(scope="module")
def parsed():
    return parse_bicep(SAMPLE, "sample.bicep")


def test_strip_comments_keeps_strings_and_line_count():
    source = "a: 'x // not a comment' // real\n/* one\ntwo */ b: 1\n"
    stripped = strip_comments(source)
    assert "'x // not a comment'" in stripped
    assert "real" not in stripped and "one" not in stripped
    assert stripped.count("\n") == source.count("\n")


def test_params_and_decorators(parsed):
    names = {p.name: p for p in parsed.params}
    assert set(names) == {"location", "adminPassword"}
    assert names["adminPassword"].secure
    assert not names["location"].secure
    assert isinstance(names["location"].default, Expr)
    assert names["location"].type == "string"


def test_variables_with_quoted_keys(parsed):
    assert parsed.variables["tags"] == {"environment": "dev", "cost-center": "CC-1"}


def test_resource_literals_and_expressions(parsed):
    account = parsed.resource("account")
    assert account.type == "Microsoft.Storage/storageAccounts"
    assert account.api_version == "2023-05-01"
    assert account.get("properties.publicNetworkAccess") == "Disabled"
    assert account.get("properties.allowBlobPublicAccess") is False
    assert account.get("properties.retention") == 14
    assert account.get("properties.url") == "https://example.com/a//b"
    assert account.get("sku.name") == "Standard_LRS"
    assert isinstance(account.get("name"), Expr)
    assert account.get("tags") == Expr("tags")
    assert account.get("properties.rules") == [{"name": "a", "priority": 100}, {"name": "b", "priority": 200}]


def test_property_lookup_is_case_insensitive_and_defaults(parsed):
    account = parsed.resource("account")
    assert account.get("PROPERTIES.publicnetworkaccess") == "Disabled"
    assert account.get("properties.missing", "fallback") == "fallback"


def test_nested_child_resource(parsed):
    blob = parsed.resource("blob")
    assert blob.type == "Microsoft.Storage/storageAccounts/blobServices"
    assert blob.parent == "account"
    assert blob.is_child
    assert "blob" not in parsed.resource("account").body


def test_existing_conditional_and_loop_resources(parsed):
    assert parsed.resource("vault").existing
    assert parsed.resource("optional").condition == "location == 'westeurope'"
    zones = parsed.resource("zones")
    assert zones.loop
    assert zones.get("location") == "global"


def test_module_params_and_outputs(parsed):
    module = parsed.modules[0]
    assert module.path == "modules/private-endpoint.bicep"
    assert module.params["targetResourceId"] == Expr("account.id")
    assert module.params["groupIds"] == ["blob"]
    assert parsed.outputs[0].name == "accountId"
    assert parsed.outputs[0].value.references("account")
    assert not parsed.outputs[0].value.references("acc")


def test_for_loop_inside_property():
    file = parse_bicep("var x = {\n  items: [for (id, i) in ids: {\n    name: 'zone-${i}'\n    id: id\n  }]\n}\n")
    loop = file.variables["x"]["items"]
    assert isinstance(loop, ForLoop)
    assert loop.header == "(id, i) in ids"
    assert loop.body["id"] == Expr("id")


def test_line_numbers_are_reported(parsed):
    assert parsed.resource("account").line == SAMPLE[: SAMPLE.index("resource account")].count("\n") + 1


def test_parse_errors_carry_line_numbers():
    with pytest.raises(BicepParseError) as error:
        parse_bicep("param a string\nresource x 'T@1' = {\n  name: 'a'\n")
    assert "unterminated object" in str(error.value)
    with pytest.raises(BicepParseError, match="line 2"):
        parse_bicep("param a string\nbogus thing\n")


def test_repository_templates_parse(repo_root):
    files = sorted((repo_root / "infra").rglob("*.bicep"))
    assert len(files) == 12
    parsed = {f.name: parse_bicep_file(f) for f in files}
    assert len(parsed["main.bicep"].modules) == 10
    assert all(not m.loop for m in parsed["main.bicep"].modules)
    deployments = parsed["openai.bicep"].resource("modelDeployments")
    assert deployments.loop and deployments.parent == "account"
    assert deployments.decorators == ["batchSize(1)"]
