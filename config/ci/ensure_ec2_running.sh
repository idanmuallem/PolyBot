#!/usr/bin/env bash
# Starts the deploy target's EC2 instance if it is not running, and waits
# for SSM to come online. Runs on the GitHub runner before any SSM command
# (deploy-staging and deploy-production in .github/workflows/deploy.yml).
#
# Requires env: EC2_INSTANCE_ID, and AWS credentials with
# ec2:DescribeInstances / ec2:StartInstances on that instance.
set -euo pipefail

STATE=$(aws ec2 describe-instances \
  --instance-ids "$EC2_INSTANCE_ID" \
  --query 'Reservations[0].Instances[0].State.Name' \
  --output text)
if [ "$STATE" != "running" ]; then
  echo "Instance is $STATE — starting it"
  aws ec2 start-instances --instance-ids "$EC2_INSTANCE_ID"
  aws ec2 wait instance-running --instance-ids "$EC2_INSTANCE_ID"
  echo "Waiting 30s for SSM agent to come online..."
  sleep 30
else
  echo "Instance already running"
fi
