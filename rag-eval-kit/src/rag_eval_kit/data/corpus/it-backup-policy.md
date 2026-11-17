# Backup Policy

## Schedule
Production databases are backed up incrementally every 4 hours and fully every night at 01:00. File shares are backed up once per day.

## Retention of backups
Daily backups are kept for 35 days, weekly backups for 12 weeks and monthly backups for 12 months. One copy is stored in a second region and one copy is immutable for 14 days as protection against ransomware.

## Restore tests
A restore test is carried out every quarter for each critical system. The recovery time objective for critical systems is 4 hours and the recovery point objective is 1 hour. Results are documented by the system owner.
