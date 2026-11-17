using '../main.bicep'

// Development environment: smallest SKUs that still keep the private network
// and keyless identity posture identical to production.

param environmentName = 'dev'
param workloadName = 'ragref'
param costCenter = 'CC-0000-DEV'
param owner = 'platform-team@example.com'
param dataClassification = 'internal'
param vnetAddressPrefix = '10.40.0.0/20'

// Model names and versions are examples. Replace them with models that are
// available in your region and approved by your organisation.
param modelDeployments = [
  {
    name: 'chat-small'
    modelName: 'gpt-4o-mini'
    modelVersion: '2024-07-18'
    skuName: 'Standard'
    capacity: 50
  }
  {
    name: 'chat-large'
    modelName: 'gpt-4o'
    modelVersion: '2024-08-06'
    skuName: 'Standard'
    capacity: 30
  }
  {
    name: 'embedding'
    modelName: 'text-embedding-3-large'
    modelVersion: '1'
    skuName: 'Standard'
    capacity: 50
  }
]
param defaultChatDeployment = 'chat-small'
param escalationChatDeployment = 'chat-large'
param embeddingDeployment = 'embedding'

param searchSkuName = 'basic'
param searchReplicaCount = 1
param searchPartitionCount = 1
param storageSkuName = 'Standard_LRS'

param apimSkuName = 'Developer'
param apimSkuCapacity = 1
param apimPublisherEmail = 'platform-team@example.com'
param apimPublisherName = 'Platform Team'
param apiAudience = 'api://ragref-dev'
param tokensPerMinutePerSubscription = 10000

param orchestratorImage = 'mcr.microsoft.com/k8se/quickstart:latest'
param orchestratorMinReplicas = 1
param orchestratorMaxReplicas = 3
param zoneRedundant = false

param logRetentionInDays = 30
param logDailyQuotaGb = 2
