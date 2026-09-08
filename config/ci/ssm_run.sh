#!/usr/bin/env bash
# Runs a shell command on the deploy target's EC2 instance via AWS SSM and
# prints its stdout, forwarding the remote script's exit status. Shared by
# the pre-deploy snapshot step and every post-deploy meta-harness check in
# .github/workflows/deploy.yml so none of them re-implement the
# send-command/poll/fetch-output dance that already exists once in that
# file's "Deploy to EC2 via AWS SSM" step.
#
# Usage: ssm_run.sh "<remote shell command or script body>"
# Requires env: AWS_REGION, EC2_INSTANCE_ID. AWS credentials must already be
# configured (e.g. via aws-actions/configure-aws-credentials) with
# ssm:SendCommand / ssm:GetCommandInvocation on that instance — the same
# permissions the deploy step already relies on, nothing broader.
#
# The remote command always runs under bash, regardless of the instance's
# default shell (AWS-RunShellScript respects a leading shebang) — callers
# can rely on bash features (e.g. [[, arrays) without re-declaring this.
set -euo pipefail

remote_cmd="$1"
script_body="#!/bin/bash
set -euo pipefail
${remote_cmd}"

params_file="$(mktemp)"
trap 'rm -f "$params_file"' EXIT
jq -n --arg cmd "$script_body" '{commands: [$cmd]}' > "$params_file"

command_id=$(aws ssm send-command \
  --instance-ids "$EC2_INSTANCE_ID" \
  --document-name "AWS-RunShellScript" \
  --comment "CI meta-harness check" \
  --parameters "file://$params_file" \
  --query 'Command.CommandId' --output text)

status="InProgress"
for _ in $(seq 1 60); do
  status=$(aws ssm get-command-invocation \
    --command-id "$command_id" --instance-id "$EC2_INSTANCE_ID" \
    --query 'Status' --output text 2>/dev/null || echo "Pending")
  case "$status" in
    Success|Failed|Cancelled|TimedOut) break ;;
  esac
  sleep 5
done

stdout=$(aws ssm get-command-invocation --command-id "$command_id" --instance-id "$EC2_INSTANCE_ID" \
  --query 'StandardOutputContent' --output text || true)
stderr=$(aws ssm get-command-invocation --command-id "$command_id" --instance-id "$EC2_INSTANCE_ID" \
  --query 'StandardErrorContent' --output text || true)

printf '%s' "$stdout"

if [ "$status" != "Success" ]; then
  echo "$stderr" >&2
  echo "ssm_run.sh: remote command did not succeed (status=$status)" >&2
  exit 1
fi
