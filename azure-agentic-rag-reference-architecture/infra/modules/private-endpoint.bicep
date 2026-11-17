// Reusable private endpoint with a private DNS zone group.

@description('Azure region for the resources.')
param location string

@description('Name of the private endpoint.')
param name string

@description('Resource tags applied to every resource.')
param tags object

@description('Subnet that hosts the private endpoint.')
param subnetId string

@description('Resource ID of the service to connect to.')
param targetResourceId string

@description('Private link sub-resource, for example "account", "searchService", "vault" or "blob".')
param groupId string

@description('Private DNS zones that receive the A record of the endpoint.')
param privateDnsZoneIds array

resource privateEndpoint 'Microsoft.Network/privateEndpoints@2024-05-01' = {
  name: name
  location: location
  tags: tags
  properties: {
    subnet: {
      id: subnetId
    }
    privateLinkServiceConnections: [
      {
        name: name
        properties: {
          privateLinkServiceId: targetResourceId
          groupIds: [
            groupId
          ]
        }
      }
    ]
  }
}

resource dnsZoneGroup 'Microsoft.Network/privateEndpoints/privateDnsZoneGroups@2024-05-01' = {
  parent: privateEndpoint
  name: 'default'
  properties: {
    privateDnsZoneConfigs: [
      for (zoneId, i) in privateDnsZoneIds: {
        name: 'zone-${i}'
        properties: {
          privateDnsZoneId: zoneId
        }
      }
    ]
  }
}

output privateEndpointId string = privateEndpoint.id
