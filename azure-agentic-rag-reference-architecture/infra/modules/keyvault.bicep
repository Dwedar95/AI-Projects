// Key Vault with RBAC authorization, purge protection and a private endpoint.
// The platform itself is keyless (managed identity everywhere); the vault holds
// customer-managed keys and any third-party secrets a workload still needs.

@description('Azure region for the resources.')
param location string

@description('Name prefix, for example "ragref-dev".')
param namePrefix string

@description('Resource tags applied to every resource.')
param tags object

@description('Subnet that hosts the private endpoint.')
param privateEndpointSubnetId string

@description('Private DNS zone for privatelink.vaultcore.azure.net.')
param privateDnsZoneId string

@description('Log Analytics workspace that receives audit logs.')
param workspaceId string

@description('Soft delete retention in days.')
@minValue(7)
@maxValue(90)
param softDeleteRetentionInDays int = 90

resource vault 'Microsoft.KeyVault/vaults@2023-07-01' = {
  name: 'kv-${namePrefix}-${take(uniqueString(resourceGroup().id), 6)}'
  location: location
  tags: tags
  properties: {
    tenantId: subscription().tenantId
    sku: {
      family: 'A'
      name: 'standard'
    }
    enableRbacAuthorization: true
    enableSoftDelete: true
    softDeleteRetentionInDays: softDeleteRetentionInDays
    enablePurgeProtection: true
    publicNetworkAccess: 'Disabled'
    networkAcls: {
      bypass: 'AzureServices'
      defaultAction: 'Deny'
    }
  }
}

module vaultPrivateEndpoint 'private-endpoint.bicep' = {
  name: 'pe-keyvault'
  params: {
    location: location
    name: 'pe-${namePrefix}-kv'
    tags: tags
    subnetId: privateEndpointSubnetId
    targetResourceId: vault.id
    groupId: 'vault'
    privateDnsZoneIds: [
      privateDnsZoneId
    ]
  }
}

resource vaultDiagnostics 'Microsoft.Insights/diagnosticSettings@2021-05-01-preview' = {
  name: 'diag-to-log-analytics'
  scope: vault
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

output vaultId string = vault.id
output vaultName string = vault.name
output vaultUri string = vault.properties.vaultUri
