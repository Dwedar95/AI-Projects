// Azure AI Foundry (AI Services) account that hosts the Azure OpenAI model
// deployments. Key-based auth is disabled and the account is reachable only
// through its private endpoint.

@description('Azure region for the resources. Must be a region that satisfies the data residency decision (ADR 0005).')
param location string

@description('Name prefix, for example "ragref-dev".')
param namePrefix string

@description('Resource tags applied to every resource.')
param tags object

@description('Subnet that hosts the private endpoint.')
param privateEndpointSubnetId string

@description('Private DNS zones for the account (openai, cognitiveservices and services.ai).')
param privateDnsZoneIds array

@description('Log Analytics workspace that receives request and audit logs.')
param workspaceId string

@description('Model deployments. Each item: { name, modelName, modelVersion, skuName, capacity }.')
param deployments array

resource account 'Microsoft.CognitiveServices/accounts@2024-10-01' = {
  name: 'aif-${namePrefix}'
  location: location
  tags: tags
  kind: 'AIServices'
  sku: {
    name: 'S0'
  }
  identity: {
    type: 'SystemAssigned'
  }
  properties: {
    customSubDomainName: 'aif-${namePrefix}-${take(uniqueString(resourceGroup().id), 6)}'
    disableLocalAuth: true
    publicNetworkAccess: 'Disabled'
    networkAcls: {
      defaultAction: 'Deny'
    }
    restrictOutboundNetworkAccess: true
  }
}

@batchSize(1)
resource modelDeployments 'Microsoft.CognitiveServices/accounts/deployments@2024-10-01' = [
  for deployment in deployments: {
    parent: account
    name: deployment.name
    sku: {
      name: deployment.skuName
      capacity: deployment.capacity
    }
    properties: {
      model: {
        format: 'OpenAI'
        name: deployment.modelName
        version: deployment.modelVersion
      }
      versionUpgradeOption: 'NoAutoUpgrade'
      raiPolicyName: 'Microsoft.DefaultV2'
    }
  }
]

module accountPrivateEndpoint 'private-endpoint.bicep' = {
  name: 'pe-ai-foundry'
  params: {
    location: location
    name: 'pe-${namePrefix}-aif'
    tags: tags
    subnetId: privateEndpointSubnetId
    targetResourceId: account.id
    groupId: 'account'
    privateDnsZoneIds: privateDnsZoneIds
  }
}

resource accountDiagnostics 'Microsoft.Insights/diagnosticSettings@2021-05-01-preview' = {
  name: 'diag-to-log-analytics'
  scope: account
  properties: {
    workspaceId: workspaceId
    logs: [
      {
        categoryGroup: 'audit'
        enabled: true
      }
      {
        categoryGroup: 'allLogs'
        enabled: true
      }
    ]
    metrics: [
      {
        category: 'AllMetrics'
        enabled: true
      }
    ]
  }
}

output accountId string = account.id
output accountName string = account.name
output endpoint string = account.properties.endpoint
output principalId string = account.identity.principalId
