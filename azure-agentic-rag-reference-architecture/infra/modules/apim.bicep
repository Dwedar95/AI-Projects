// API Management in internal VNet mode as the AI gateway: Entra ID token
// validation, per-subscription token limits, token metrics and a single
// governed entry point in front of the orchestrator.

@description('Azure region for the resources.')
param location string

@description('Name prefix, for example "ragref-dev".')
param namePrefix string

@description('Resource tags applied to every resource.')
param tags object

@description('Subnet for the API Management instance.')
param subnetId string

@description('Log Analytics workspace that receives gateway logs.')
param workspaceId string

@description('Resource ID of the Application Insights component used by the gateway logger.')
param appInsightsId string

@description('Application Insights connection string used by the gateway logger.')
@secure()
param appInsightsConnectionString string

@description('Publisher e-mail shown in the developer portal and used for notifications.')
param publisherEmail string

@description('Publisher name.')
param publisherName string

@description('API Management SKU. Internal VNet mode needs Developer or Premium.')
@allowed([
  'Developer'
  'Premium'
])
param skuName string = 'Developer'

@description('Number of scale units.')
@minValue(1)
param skuCapacity int = 1

@description('Backend URL of the orchestrator (internal Container Apps FQDN).')
param orchestratorBackendUrl string

@description('Entra ID tenant that issues caller tokens.')
param tenantId string = subscription().tenantId

@description('Application ID URI (audience) that callers must request a token for.')
param apiAudience string

@description('Tokens per minute allowed per API Management subscription.')
param tokensPerMinutePerSubscription int = 20000

resource service 'Microsoft.ApiManagement/service@2023-05-01-preview' = {
  name: 'apim-${namePrefix}-${take(uniqueString(resourceGroup().id), 6)}'
  location: location
  tags: tags
  sku: {
    name: skuName
    capacity: skuCapacity
  }
  identity: {
    type: 'SystemAssigned'
  }
  properties: {
    publisherEmail: publisherEmail
    publisherName: publisherName
    virtualNetworkType: 'Internal'
    virtualNetworkConfiguration: {
      subnetResourceId: subnetId
    }
    customProperties: {
      'Microsoft.WindowsAzure.ApiManagement.Gateway.Security.Protocols.Tls10': 'False'
      'Microsoft.WindowsAzure.ApiManagement.Gateway.Security.Protocols.Tls11': 'False'
      'Microsoft.WindowsAzure.ApiManagement.Gateway.Security.Backend.Protocols.Tls10': 'False'
      'Microsoft.WindowsAzure.ApiManagement.Gateway.Security.Backend.Protocols.Tls11': 'False'
    }
  }
}

resource tenantNamedValue 'Microsoft.ApiManagement/service/namedValues@2023-05-01-preview' = {
  parent: service
  name: 'entra-tenant-id'
  properties: {
    displayName: 'entra-tenant-id'
    value: tenantId
  }
}

resource audienceNamedValue 'Microsoft.ApiManagement/service/namedValues@2023-05-01-preview' = {
  parent: service
  name: 'api-audience'
  properties: {
    displayName: 'api-audience'
    value: apiAudience
  }
}

resource tokenLimitNamedValue 'Microsoft.ApiManagement/service/namedValues@2023-05-01-preview' = {
  parent: service
  name: 'tokens-per-minute'
  properties: {
    displayName: 'tokens-per-minute'
    value: string(tokensPerMinutePerSubscription)
  }
}

resource appInsightsLogger 'Microsoft.ApiManagement/service/loggers@2023-05-01-preview' = {
  parent: service
  name: 'appinsights'
  properties: {
    loggerType: 'applicationInsights'
    resourceId: appInsightsId
    credentials: {
      connectionString: appInsightsConnectionString
      identityClientId: 'SystemAssigned'
    }
  }
}

resource orchestratorBackend 'Microsoft.ApiManagement/service/backends@2023-05-01-preview' = {
  parent: service
  name: 'orchestrator'
  properties: {
    protocol: 'http'
    url: orchestratorBackendUrl
    circuitBreaker: {
      rules: [
        {
          name: 'trip-on-5xx'
          failureCondition: {
            count: 5
            interval: 'PT30S'
            statusCodeRanges: [
              {
                min: 500
                max: 599
              }
            ]
          }
          tripDuration: 'PT30S'
        }
      ]
    }
  }
}

resource chatApi 'Microsoft.ApiManagement/service/apis@2023-05-01-preview' = {
  parent: service
  name: 'agentic-rag'
  properties: {
    displayName: 'Agentic RAG API'
    path: 'rag'
    protocols: [
      'https'
    ]
    subscriptionRequired: true
    serviceUrl: orchestratorBackendUrl
  }
  dependsOn: [
    orchestratorBackend
  ]
}

resource chatOperation 'Microsoft.ApiManagement/service/apis/operations@2023-05-01-preview' = {
  parent: chatApi
  name: 'chat'
  properties: {
    displayName: 'Chat with grounded answers'
    method: 'POST'
    urlTemplate: '/chat'
  }
}

resource chatApiPolicy 'Microsoft.ApiManagement/service/apis/policies@2023-05-01-preview' = {
  parent: chatApi
  name: 'policy'
  properties: {
    format: 'rawxml'
    value: loadTextContent('../policies/chat-api-policy.xml')
  }
  dependsOn: [
    tenantNamedValue
    audienceNamedValue
    tokenLimitNamedValue
    appInsightsLogger
  ]
}

resource serviceDiagnostics 'Microsoft.Insights/diagnosticSettings@2021-05-01-preview' = {
  name: 'diag-to-log-analytics'
  scope: service
  properties: {
    workspaceId: workspaceId
    logAnalyticsDestinationType: 'Dedicated'
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

output serviceId string = service.id
output serviceName string = service.name
output gatewayUrl string = service.properties.gatewayUrl
output principalId string = service.identity.principalId
