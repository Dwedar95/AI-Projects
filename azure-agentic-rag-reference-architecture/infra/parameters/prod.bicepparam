using '../main.bicep'

// Production environment: zone redundancy, search replicas for the read SLA,
// provisioned throughput for the default chat model and standard spillover.

param environmentName = 'prod'
param workloadName = 'ragref'
param costCenter = 'CC-0000-PROD'
param owner = 'platform-team@example.com'
param dataClassification = 'confidential'
param vnetAddressPrefix = '10.41.0.0/20'

// Model names, versions and capacities are examples. Size the provisioned
// deployment with `archkit estimate` and your own measured traffic.
param modelDeployments = [
  {
    name: 'chat-small-ptu'
    modelName: 'gpt-4o-mini'
    modelVersion: '2024-07-18'
    skuName: 'DataZoneProvisionedManaged'
    capacity: 50
  }
  {
    name: 'chat-small'
    modelName: 'gpt-4o-mini'
    modelVersion: '2024-07-18'
    skuName: 'DataZoneStandard'
    capacity: 200
  }
  {
    name: 'chat-large'
    modelName: 'gpt-4o'
    modelVersion: '2024-08-06'
    skuName: 'DataZoneStandard'
    capacity: 150
  }
  {
    name: 'embedding'
    modelName: 'text-embedding-3-large'
    modelVersion: '1'
    skuName: 'Standard'
    capacity: 200
  }
]
param defaultChatDeployment = 'chat-small-ptu'
param escalationChatDeployment = 'chat-large'
param embeddingDeployment = 'embedding'

param searchSkuName = 'standard'
param searchReplicaCount = 3
param searchPartitionCount = 1
param storageSkuName = 'Standard_ZRS'

param apimSkuName = 'Premium'
param apimSkuCapacity = 2
param apimPublisherEmail = 'platform-team@example.com'
param apimPublisherName = 'Platform Team'
param apiAudience = 'api://ragref-prod'
param tokensPerMinutePerSubscription = 60000

param orchestratorImage = 'mcr.microsoft.com/k8se/quickstart:latest'
param orchestratorMinReplicas = 3
param orchestratorMaxReplicas = 20
param zoneRedundant = true

param logRetentionInDays = 180
param logDailyQuotaGb = -1
