# MW-200 Firmware Update Procedure

## Preparation
Download the firmware package from the customer portal and compare its SHA-256 checksum with the value in the release notes. Export the device configuration first. An update takes about 6 minutes.

## Update
Upload the package under System, Firmware in the web interface. Do not switch off power during the update. The device restarts twice. A steady green status LED shows that the update is complete.

## Rollback
The gateway keeps the previous firmware in a second partition. If the new firmware fails to boot three times in a row, the device returns to the previous version automatically. A manual rollback is available in the web interface for 30 days.
