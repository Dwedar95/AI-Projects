// Azure AI Search with Entra-only authentication and a private endpoint.

@description('Azure region for the resources.')
param location string

@description('Name prefix, for example "ragref-dev".')
param namePrefix string

@description('Resource tags applied to every resource.')
param tags object

@description('Subnet that hosts the private endpoint.')
param privateEndpointSubnetId string

@description('Private DNS zone for privatelink.search.windows.net.')
param privateDnsZoneId string

@description('Log Analytics workspace that receives query and indexing logs.')
param workspaceId string

@description('Search tier.')
@allowed([
  'basic'
  'standard'
  'standard2'
  'standard3'
  'storage_optimized_l1'
])
param skuName string = 'standard'

@description('Replica count. Two or more are required for a read availability SLA.')
@minValue(1)
@maxValue(12)
param replicaCount int = 2

@description('Partition count.')
@allowed([
  1
  2
  3
  4
  6
  12
])
param partitionCount int = 1

resource searchService 'Microsoft.Search/searchServices@2023-11-01' = {
  name: 'srch-${namePrefix}-${take(uniqueString(resourceGroup().id), 6)}'
  location: location
  tags: tags
  sku: {
    name: skuName
  }
  identity: {
    type: 'SystemAssigned'
  }
  properties: {
    replicaCount: replicaCount
    partitionCount: partitionCount
    hostingMode: 'default'
    publicNetworkAccess: 'disabled'
    disableLocalAuth: true
    semanticSearch: 'standard'
  }
}

module searchPrivateEndpoint 'private-endpoint.bicep' = {
  name: 'pe-search'
  params: {
    location: location
    name: 'pe-${namePrefix}-srch'
    tags: tags
    subnetId: privateEndpointSubnetId
    targetResourceId: searchService.id
    groupId: 'searchService'
    privateDnsZoneIds: [
      privateDnsZoneId
    ]
  }
}

resource searchDiagnostics 'Microsoft.Insights/diagnosticSettings@2021-05-01-preview' = {
  name: 'diag-to-log-analytics'
  scope: searchService
  properties: {
    workspaceId: workspaceId
    logs: [
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

output searchServiceId string = searchService.id
output searchServiceName string = searchService.name
output endpoint string = 'https://${searchService.name}.search.windows.net'
output principalId string = searchService.identity.principalId
