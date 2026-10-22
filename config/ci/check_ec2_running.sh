#!/usr/bin/env bash
# Reports whether the deploy target's EC2 instance is running. It never
# starts the instance: the owner stops it on purpose, and starting it would
# also bring the Production container back (`--restart always`) without any
# deploy. Starting the instance is a manual step.
#
# Writes `running=true|false` to $GITHUB_OUTPUT.
#   check_ec2_running.sh            exit 0 either way (deploy-staging skips
#                                   its deploy and the rest of the pipeline)
#   check_ec2_running.sh --require  exit 1 when not running (deploy-production,
#                                   after the owner approved)
#
# Requires env: EC2_INSTANCE_ID, and AWS credentials with
# ec2:DescribeInstances on that instance.
set -euo pipefail

STATE=$(aws ec2 describe-instances \
  --instance-ids "$EC2_INSTANCE_ID" \
  --query 'Reservations[0].Instances[0].State.Name' \
  --output text)

if [ "$STATE" = "running" ]; then
  echo "Instance is running"
  echo "running=true" >> "$GITHUB_OUTPUT"
  exit 0
fi

echo "running=false" >> "$GITHUB_OUTPUT"
if [ "${1:-}" = "--require" ]; then
  echo "::error::EC2 instance is $STATE. Start it by hand, then re-run this job."
  exit 1
fi
echo "::notice::EC2 instance is $STATE, so nothing was deployed. Start it by hand and re-run this workflow to deploy."
echo "EC2 instance is $STATE: deploy skipped (the instance is never started automatically)." >> "$GITHUB_STEP_SUMMARY"
