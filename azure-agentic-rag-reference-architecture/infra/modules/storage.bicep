// Storage account for source documents that the indexer reads.

@description('Azure region for the resources.')
param location string

@description('Name prefix, for example "ragref-dev".')
param namePrefix string

@description('Resource tags applied to every resource.')
param tags object

@description('Subnet that hosts the private endpoint.')
param privateEndpointSubnetId string

@description('Private DNS zone for the blob endpoint.')
param privateDnsZoneId string

@description('Log Analytics workspace that receives diagnostics.')
param workspaceId string

@description('Storage redundancy. Use a zone-redundant SKU in production.')
@allowed([
  'Standard_LRS'
  'Standard_ZRS'
  'Standard_GZRS'
])
param skuName string = 'Standard_ZRS'

resource account 'Microsoft.Storage/storageAccounts@2023-05-01' = {
  name: take('st${replace(namePrefix, '-', '')}${uniqueString(resourceGroup().id)}', 24)
  location: location
  tags: tags
  kind: 'StorageV2'
  sku: {
    name: skuName
  }
  properties: {
    accessTier: 'Hot'
    allowBlobPublicAccess: false
    allowSharedKeyAccess: false
    defaultToOAuthAuthentication: true
    minimumTlsVersion: 'TLS1_2'
    supportsHttpsTrafficOnly: true
    publicNetworkAccess: 'Disabled'
    networkAcls: {
      bypass: 'AzureServices'
      defaultAction: 'Deny'
    }
  }
}

resource blobService 'Microsoft.Storage/storageAccounts/blobServices@2023-05-01' = {
  parent: account
  name: 'default'
  properties: {
    deleteRetentionPolicy: {
      enabled: true
      days: 14
    }
    isVersioningEnabled: true
  }
}

resource documentsContainer 'Microsoft.Storage/storageAccounts/blobServices/containers@2023-05-01' = {
  parent: blobService
  name: 'documents'
  properties: {
    publicAccess: 'None'
  }
}

module blobPrivateEndpoint 'private-endpoint.bicep' = {
  name: 'pe-storage-blob'
  params: {
    location: location
    name: 'pe-${namePrefix}-blob'
    tags: tags
    subnetId: privateEndpointSubnetId
    targetResourceId: account.id
    groupId: 'blob'
    privateDnsZoneIds: [
      privateDnsZoneId
    ]
  }
}

resource accountDiagnostics 'Microsoft.Insights/diagnosticSettings@2021-05-01-preview' = {
  name: 'diag-to-log-analytics'
  scope: account
  properties: {
    workspaceId: workspaceId
    metrics: [
      {
        category: 'Transaction'
        enabled: true
      }
    ]
  }
}

resource blobDiagnostics 'Microsoft.Insights/diagnosticSettings@2021-05-01-preview' = {
  name: 'diag-to-log-analytics'
  scope: blobService
  properties: {
    workspaceId: workspaceId
    logs: [
      {
        categoryGroup: 'allLogs'
        enabled: true
      }
    ]
  }
}

output storageAccountId string = account.id
output storageAccountName string = account.name
output documentsContainerName string = documentsContainer.name
