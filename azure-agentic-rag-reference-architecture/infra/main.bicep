// Reference architecture: secure enterprise agentic RAG platform on Azure.
// Scope: one resource group per environment. See docs/architecture.md and docs/adr.
//
// This template was compiled with the Bicep CLI and checked by the archkit
// static validator. It has NOT been deployed to a subscription by the author
// of this repository. Review it against your landing zone before use.

targetScope = 'resourceGroup'

@description('Deployment environment.')
@allowed([
  'dev'
  'test'
  'prod'
])
param environmentName string

@description('Short workload name used in resource names.')
@minLength(3)
@maxLength(10)
param workloadName string = 'ragref'

@description('Azure region. Must satisfy the data residency decision (ADR 0005).')
param location string = resourceGroup().location

@description('Cost center tag used for showback (ADR 0006).')
param costCenter string

@description('Owner tag: team or distribution list responsible for the workload.')
param owner string

@description('Data classification tag.')
@allowed([
  'public'
  'internal'
  'confidential'
])
param dataClassification string = 'confidential'

@description('Address space of the virtual network.')
param vnetAddressPrefix string = '10.40.0.0/20'

@description('Model deployments. Each item: { name, modelName, modelVersion, skuName, capacity }.')
param modelDeployments array

@description('Deployment name of the default (small) chat model.')
param defaultChatDeployment string

@description('Deployment name of the escalation (large) chat model.')
param escalationChatDeployment string

@description('Deployment name of the embedding model.')
param embeddingDeployment string

@description('AI Search tier.')
param searchSkuName string = 'standard'

@description('AI Search replica count.')
param searchReplicaCount int = 2

@description('AI Search partition count.')
param searchPartitionCount int = 1

@description('Storage redundancy SKU.')
param storageSkuName string = 'Standard_ZRS'

@description('API Management SKU.')
param apimSkuName string = 'Developer'

@description('API Management scale units.')
param apimSkuCapacity int = 1

@description('API Management publisher e-mail.')
param apimPublisherEmail string

@description('API Management publisher name.')
param apimPublisherName string

@description('Application ID URI that callers request tokens for.')
param apiAudience string

@description('Tokens per minute per API Management subscription.')
param tokensPerMinutePerSubscription int = 20000

@description('Container image of the orchestrator service.')
param orchestratorImage string

@description('Minimum orchestrator replicas.')
param orchestratorMinReplicas int = 2

@description('Maximum orchestrator replicas.')
param orchestratorMaxReplicas int = 10

@description('Spread the Container Apps environment across availability zones.')
param zoneRedundant bool = true

@description('Log retention in days.')
param logRetentionInDays int = 90

@description('Log Analytics daily cap in GB. Use -1 for no cap.')
param logDailyQuotaGb int = -1

var namePrefix = '${workloadName}-${environmentName}'

var tags = {
  workload: workloadName
  environment: environmentName
  costCenter: costCenter
  owner: owner
  dataClassification: dataClassification
  managedBy: 'bicep'
}

module monitoring 'modules/monitoring.bicep' = {
  name: 'monitoring'
  params: {
    location: location
    namePrefix: namePrefix
    tags: tags
    retentionInDays: logRetentionInDays
    dailyQuotaGb: logDailyQuotaGb
  }
}

module network 'modules/network.bicep' = {
  name: 'network'
  params: {
    location: location
    namePrefix: namePrefix
    tags: tags
    vnetAddressPrefix: vnetAddressPrefix
    apimSubnetPrefix: cidrSubnet(vnetAddressPrefix, 27, 0)
    privateEndpointSubnetPrefix: cidrSubnet(vnetAddressPrefix, 26, 4)
    containerAppsSubnetPrefix: cidrSubnet(vnetAddressPrefix, 23, 1)
    workspaceId: monitoring.outputs.workspaceId
  }
}

module identity 'modules/identity.bicep' = {
  name: 'identity'
  params: {
    location: location
    namePrefix: namePrefix
    tags: tags
  }
}

module keyVault 'modules/keyvault.bicep' = {
  name: 'keyvault'
  params: {
    location: location
    namePrefix: namePrefix
    tags: tags
    privateEndpointSubnetId: network.outputs.privateEndpointSubnetId
    privateDnsZoneId: network.outputs.keyVaultDnsZoneId
    workspaceId: monitoring.outputs.workspaceId
  }
}

module storage 'modules/storage.bicep' = {
  name: 'storage'
  params: {
    location: location
    namePrefix: namePrefix
    tags: tags
    privateEndpointSubnetId: network.outputs.privateEndpointSubnetId
    privateDnsZoneId: network.outputs.blobDnsZoneId
    workspaceId: monitoring.outputs.workspaceId
    skuName: storageSkuName
  }
}

module openAi 'modules/openai.bicep' = {
  name: 'openai'
  params: {
    location: location
    namePrefix: namePrefix
    tags: tags
    privateEndpointSubnetId: network.outputs.privateEndpointSubnetId
    privateDnsZoneIds: [
      network.outputs.openAiDnsZoneId
      network.outputs.cognitiveServicesDnsZoneId
      network.outputs.aiServicesDnsZoneId
    ]
    workspaceId: monitoring.outputs.workspaceId
    deployments: modelDeployments
  }
}

module search 'modules/search.bicep' = {
  name: 'search'
  params: {
    location: location
    namePrefix: namePrefix
    tags: tags
    privateEndpointSubnetId: network.outputs.privateEndpointSubnetId
    privateDnsZoneId: network.outputs.searchDnsZoneId
    workspaceId: monitoring.outputs.workspaceId
    skuName: searchSkuName
    replicaCount: searchReplicaCount
    partitionCount: searchPartitionCount
  }
}

module containerApps 'modules/container-apps.bicep' = {
  name: 'container-apps'
  params: {
    location: location
    namePrefix: namePrefix
    tags: tags
    infrastructureSubnetId: network.outputs.containerAppsSubnetId
    workspaceId: monitoring.outputs.workspaceId
    workloadIdentityId: identity.outputs.identityId
    workloadIdentityClientId: identity.outputs.clientId
    containerImage: orchestratorImage
    openAiEndpoint: openAi.outputs.endpoint
    searchEndpoint: search.outputs.endpoint
    keyVaultUri: keyVault.outputs.vaultUri
    appInsightsConnectionString: monitoring.outputs.appInsightsConnectionString
    defaultChatDeployment: defaultChatDeployment
    escalationChatDeployment: escalationChatDeployment
    embeddingDeployment: embeddingDeployment
    zoneRedundant: zoneRedundant
    minReplicas: orchestratorMinReplicas
    maxReplicas: orchestratorMaxReplicas
  }
}

module apim 'modules/apim.bicep' = {
  name: 'apim'
  params: {
    location: location
    namePrefix: namePrefix
    tags: tags
    subnetId: network.outputs.apimSubnetId
    workspaceId: monitoring.outputs.workspaceId
    appInsightsId: monitoring.outputs.appInsightsId
    appInsightsConnectionString: monitoring.outputs.appInsightsConnectionString
    publisherEmail: apimPublisherEmail
    publisherName: apimPublisherName
    skuName: apimSkuName
    skuCapacity: apimSkuCapacity
    orchestratorBackendUrl: 'https://${containerApps.outputs.orchestratorFqdn}'
    apiAudience: apiAudience
    tokensPerMinutePerSubscription: tokensPerMinutePerSubscription
  }
}

module roleAssignments 'modules/role-assignments.bicep' = {
  name: 'role-assignments'
  params: {
    workloadPrincipalId: identity.outputs.principalId
    searchPrincipalId: search.outputs.principalId
    openAiPrincipalId: openAi.outputs.principalId
    openAiAccountName: openAi.outputs.accountName
    searchServiceName: search.outputs.searchServiceName
    storageAccountName: storage.outputs.storageAccountName
    keyVaultName: keyVault.outputs.vaultName
  }
}

output gatewayUrl string = apim.outputs.gatewayUrl
output orchestratorName string = containerApps.outputs.orchestratorName
output openAiEndpoint string = openAi.outputs.endpoint
output searchEndpoint string = search.outputs.endpoint
output keyVaultName string = keyVault.outputs.vaultName
output logAnalyticsWorkspaceName string = monitoring.outputs.workspaceName
