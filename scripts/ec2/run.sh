#!/usr/bin/env bash
# Ephemeral EC2 runner: launch -> run -> ALWAYS terminate.
#
#   ./scripts/ec2/run.sh "python3 -m tiny.run" [instance-type]
#
# Two independent guards so a run can never leak cost:
#   1. a shell trap that terminates on exit, error or Ctrl-C
#   2. `shutdown -h +45` baked into user-data, with the instance set to
#      TERMINATE on shutdown - so it dies even if this script is killed -9
set -euo pipefail

CMD="${1:?usage: run.sh <remote-command> [instance-type]}"
TYPE="${2:-c7i.xlarge}"
REG="${AWS_REGION:-us-east-1}"
KEY=~/.ssh/systemone-ec2.pem
IID=""

cleanup() {
  if [ -n "$IID" ]; then
    echo "--> terminating $IID"
    aws ec2 terminate-instances --region "$REG" --instance-ids "$IID" >/dev/null 2>&1 || true
    aws ec2 wait instance-terminated --region "$REG" --instance-ids "$IID" 2>/dev/null || true
    echo "--> terminated"
  fi
}
trap cleanup EXIT INT TERM

MYIP=$(curl -s --max-time 10 https://checkip.amazonaws.com | tr -d '\n')
SG=$(aws ec2 describe-security-groups --region "$REG" --group-names systemone-sg \
       --query 'SecurityGroups[0].GroupId' --output text)
aws ec2 authorize-security-group-ingress --region "$REG" --group-id "$SG" \
  --protocol tcp --port 22 --cidr "$MYIP/32" >/dev/null 2>&1 || true
AMI=$(aws ssm get-parameter --region "$REG" \
  --name /aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-x86_64 \
  --query Parameter.Value --output text)

UD=$(mktemp)
cat > "$UD" <<'UDEOF'
#!/bin/bash
shutdown -h +45
exec > /var/log/setup.log 2>&1
dnf -y install python3-pip gcc-c++
python3 -m pip install --quiet numpy
python3 -m pip install --quiet torch --index-url https://download.pytorch.org/whl/cpu
touch /tmp/SETUP_DONE
UDEOF

echo "--> launching $TYPE in $REG"
IID=$(aws ec2 run-instances --region "$REG" --image-id "$AMI" --instance-type "$TYPE" \
  --key-name systemone-ec2 --security-group-ids "$SG" \
  --instance-initiated-shutdown-behavior terminate \
  --user-data "file://$UD" \
  --block-device-mappings 'DeviceName=/dev/xvda,Ebs={VolumeSize=20,VolumeType=gp3,DeleteOnTermination=true}' \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=systemone},{Key=ephemeral,Value=true}]' \
  --query 'Instances[0].InstanceId' --output text)
aws ec2 wait instance-running --region "$REG" --instance-ids "$IID"
DNS=$(aws ec2 describe-instances --region "$REG" --instance-ids "$IID" \
  --query 'Reservations[0].Instances[0].PublicDnsName' --output text)
echo "--> $IID at $DNS"

SSHO=(-i "$KEY" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null
      -o LogLevel=ERROR -o ConnectTimeout=10)
for _ in $(seq 1 40); do ssh "${SSHO[@]}" ec2-user@"$DNS" true 2>/dev/null && break; sleep 10; done
echo "--> waiting for deps"
for _ in $(seq 1 60); do
  ssh "${SSHO[@]}" ec2-user@"$DNS" "test -f /tmp/SETUP_DONE" 2>/dev/null && break; sleep 10
done

ssh "${SSHO[@]}" ec2-user@"$DNS" "mkdir -p ~/work"
scp "${SSHO[@]}" -qr tiny systemone ec2-user@"$DNS":~/work/ 2>/dev/null || \
  scp "${SSHO[@]}" -qr tiny ec2-user@"$DNS":~/work/
echo "--> running: $CMD"
ssh "${SSHO[@]}" ec2-user@"$DNS" "cd ~/work && OMP_NUM_THREADS=4 $CMD"
