// Internal Container Apps environment and the agent orchestrator app.
// The app authenticates to Azure OpenAI, AI Search, Storage and Key Vault with
// a user-assigned managed identity. No service keys are injected. The only
// secret is the Application Insights connection string (telemetry ingestion).

@description('Azure region for the resources.')
param location string

@description('Name prefix, for example "ragref-dev".')
param namePrefix string

@description('Resource tags applied to every resource.')
param tags object

@description('Delegated subnet for the environment.')
param infrastructureSubnetId string

@description('Log Analytics workspace that receives environment and app diagnostics.')
param workspaceId string

@description('Resource ID of the user-assigned managed identity of the workload.')
param workloadIdentityId string

@description('Client ID of the user-assigned managed identity of the workload.')
param workloadIdentityClientId string

@description('Container image of the orchestrator service.')
param containerImage string

@description('Endpoint of the AI Foundry account.')
param openAiEndpoint string

@description('Endpoint of the AI Search service.')
param searchEndpoint string

@description('Key Vault URI.')
param keyVaultUri string

@description('Application Insights connection string. Stored as a Container Apps secret.')
@secure()
param appInsightsConnectionString string

@description('Deployment name of the default (small) chat model.')
param defaultChatDeployment string

@description('Deployment name of the escalation (large) chat model.')
param escalationChatDeployment string

@description('Deployment name of the embedding model.')
param embeddingDeployment string

@description('Spread replicas across availability zones.')
param zoneRedundant bool = true

@description('Minimum number of replicas.')
@minValue(0)
param minReplicas int = 2

@description('Maximum number of replicas.')
@minValue(1)
param maxReplicas int = 10

resource environment 'Microsoft.App/managedEnvironments@2024-03-01' = {
  name: 'cae-${namePrefix}'
  location: location
  tags: tags
  properties: {
    zoneRedundant: zoneRedundant
    vnetConfiguration: {
      internal: true
      infrastructureSubnetId: infrastructureSubnetId
    }
    appLogsConfiguration: {
      destination: 'azure-monitor'
    }
    workloadProfiles: [
      {
        name: 'Consumption'
        workloadProfileType: 'Consumption'
      }
    ]
    peerAuthentication: {
      mtls: {
        enabled: true
      }
    }
  }
}

resource orchestrator 'Microsoft.App/containerApps@2024-03-01' = {
  name: 'ca-${namePrefix}-orchestrator'
  location: location
  tags: tags
  identity: {
    type: 'UserAssigned'
    userAssignedIdentities: {
      '${workloadIdentityId}': {}
    }
  }
  properties: {
    environmentId: environment.id
    workloadProfileName: 'Consumption'
    configuration: {
      activeRevisionsMode: 'Multiple'
      ingress: {
        external: false
        targetPort: 8000
        transport: 'http'
        allowInsecure: false
      }
      secrets: [
        {
          name: 'appinsights-connection-string'
          value: appInsightsConnectionString
        }
      ]
    }
    template: {
      containers: [
        {
          name: 'orchestrator'
          image: containerImage
          resources: {
            cpu: json('1.0')
            memory: '2Gi'
          }
          env: [
            {
              name: 'AZURE_CLIENT_ID'
              value: workloadIdentityClientId
            }
            {
              name: 'AZURE_OPENAI_ENDPOINT'
              value: openAiEndpoint
            }
            {
              name: 'AZURE_SEARCH_ENDPOINT'
              value: searchEndpoint
            }
            {
              name: 'KEY_VAULT_URI'
              value: keyVaultUri
            }
            {
              name: 'APPLICATIONINSIGHTS_CONNECTION_STRING'
              secretRef: 'appinsights-connection-string'
            }
            {
              name: 'CHAT_DEPLOYMENT_DEFAULT'
              value: defaultChatDeployment
            }
            {
              name: 'CHAT_DEPLOYMENT_ESCALATION'
              value: escalationChatDeployment
            }
            {
              name: 'EMBEDDING_DEPLOYMENT'
              value: embeddingDeployment
            }
          ]
          probes: [
            {
              type: 'Liveness'
              httpGet: {
                path: '/healthz'
                port: 8000
              }
              periodSeconds: 20
            }
            {
              type: 'Readiness'
              httpGet: {
                path: '/readyz'
                port: 8000
              }
              periodSeconds: 10
            }
          ]
        }
      ]
      scale: {
        minReplicas: minReplicas
        maxReplicas: maxReplicas
        rules: [
          {
            name: 'http-concurrency'
            http: {
              metadata: {
                concurrentRequests: '20'
              }
            }
          }
        ]
      }
    }
  }
}

resource environmentDiagnostics 'Microsoft.Insights/diagnosticSettings@2021-05-01-preview' = {
  name: 'diag-to-log-analytics'
  scope: environment
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

output environmentId string = environment.id
output orchestratorName string = orchestrator.name
output orchestratorFqdn string = orchestrator.properties.configuration.ingress.fqdn
