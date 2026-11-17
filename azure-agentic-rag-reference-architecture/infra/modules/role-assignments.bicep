// Least-privilege data plane role assignments. No identity receives a
// control-plane role (Contributor or Owner) on any service.

@description('Principal ID of the orchestrator workload identity.')
param workloadPrincipalId string

@description('Principal ID of the AI Search system-assigned identity.')
param searchPrincipalId string

@description('Principal ID of the AI Foundry account system-assigned identity.')
param openAiPrincipalId string

@description('Name of the AI Foundry account.')
param openAiAccountName string

@description('Name of the AI Search service.')
param searchServiceName string

@description('Name of the storage account.')
param storageAccountName string

@description('Name of the Key Vault.')
param keyVaultName string

// Built-in role definition IDs.
var roleIds = {
  cognitiveServicesOpenAiUser: '5e0bd9bd-7b93-4f28-af87-19fc36ad61bd'
  searchIndexDataReader: '1407120a-92aa-4202-b7e9-c0e197c71c8f'
  searchIndexDataContributor: '8ebe5a00-799e-43f5-93ac-243d3dce84a7'
  storageBlobDataReader: '2a2b9908-6ea1-4ae2-8e65-a410df84e7d1'
  keyVaultSecretsUser: '4633458b-17de-408a-b874-0445c86b69e6'
}

resource openAiAccount 'Microsoft.CognitiveServices/accounts@2024-10-01' existing = {
  name: openAiAccountName
}

resource searchService 'Microsoft.Search/searchServices@2023-11-01' existing = {
  name: searchServiceName
}

resource storageAccount 'Microsoft.Storage/storageAccounts@2023-05-01' existing = {
  name: storageAccountName
}

resource keyVault 'Microsoft.KeyVault/vaults@2023-07-01' existing = {
  name: keyVaultName
}

// Orchestrator: call chat and embedding deployments.
resource workloadOpenAiUser 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(openAiAccount.id, workloadPrincipalId, roleIds.cognitiveServicesOpenAiUser)
  scope: openAiAccount
  properties: {
    principalId: workloadPrincipalId
    principalType: 'ServicePrincipal'
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', roleIds.cognitiveServicesOpenAiUser)
  }
}

// Orchestrator: query indexes, never modify them.
resource workloadSearchReader 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(searchService.id, workloadPrincipalId, roleIds.searchIndexDataReader)
  scope: searchService
  properties: {
    principalId: workloadPrincipalId
    principalType: 'ServicePrincipal'
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', roleIds.searchIndexDataReader)
  }
}

// Orchestrator: read secrets for third-party tools, if any.
resource workloadVaultSecretsUser 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(keyVault.id, workloadPrincipalId, roleIds.keyVaultSecretsUser)
  scope: keyVault
  properties: {
    principalId: workloadPrincipalId
    principalType: 'ServicePrincipal'
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', roleIds.keyVaultSecretsUser)
  }
}

// AI Search indexer: read source documents.
resource searchBlobReader 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(storageAccount.id, searchPrincipalId, roleIds.storageBlobDataReader)
  scope: storageAccount
  properties: {
    principalId: searchPrincipalId
    principalType: 'ServicePrincipal'
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', roleIds.storageBlobDataReader)
  }
}

// AI Search integrated vectorization: call the embedding deployment.
resource searchOpenAiUser 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(openAiAccount.id, searchPrincipalId, roleIds.cognitiveServicesOpenAiUser)
  scope: openAiAccount
  properties: {
    principalId: searchPrincipalId
    principalType: 'ServicePrincipal'
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', roleIds.cognitiveServicesOpenAiUser)
  }
}

// AI Foundry "on your data" style calls: read the index with the account identity.
resource openAiSearchReader 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(searchService.id, openAiPrincipalId, roleIds.searchIndexDataReader)
  scope: searchService
  properties: {
    principalId: openAiPrincipalId
    principalType: 'ServicePrincipal'
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', roleIds.searchIndexDataReader)
  }
}
