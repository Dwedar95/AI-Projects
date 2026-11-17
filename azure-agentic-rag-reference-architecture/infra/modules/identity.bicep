// User-assigned managed identity used by the orchestrator workload.

@description('Azure region for the resources.')
param location string

@description('Name prefix, for example "ragref-dev".')
param namePrefix string

@description('Resource tags applied to every resource.')
param tags object

resource workloadIdentity 'Microsoft.ManagedIdentity/userAssignedIdentities@2023-01-31' = {
  name: 'id-${namePrefix}-orchestrator'
  location: location
  tags: tags
}

output identityId string = workloadIdentity.id
output identityName string = workloadIdentity.name
output principalId string = workloadIdentity.properties.principalId
output clientId string = workloadIdentity.properties.clientId
