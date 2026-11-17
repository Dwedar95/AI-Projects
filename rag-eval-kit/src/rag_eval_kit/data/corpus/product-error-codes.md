# MW-200 Edge Gateway Error Codes

## Power and hardware
E-101 means the supply voltage is below 19 V. Check the power supply and the fuse. E-105 means the internal temperature is above 85 degrees Celsius and the device throttles. Improve ventilation.

## Network
E-210 means no link on the Ethernet port. E-214 means the device did not receive an address from the DHCP server within 60 seconds. E-230 means the cloud endpoint rejected the device certificate. Renew the certificate in the web interface.

## Storage and firmware
E-310 means the internal flash memory is more than 95 percent full. Delete old log files. E-402 means a firmware image failed the signature check and was not installed.
