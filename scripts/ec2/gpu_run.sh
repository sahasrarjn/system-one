#!/usr/bin/env bash
# Ephemeral GPU runner: launch -> sync -> run -> fetch results -> ALWAYS terminate.
#
#   ./scripts/ec2/gpu_run.sh scripts/ec2/text_experiment.sh g6.2xlarge 360
#
# Same two independent cost guards as the CPU runner, with a longer fuse
# because training runs for hours rather than minutes:
#   1. a shell trap that terminates on exit, error or Ctrl-C
#   2. `shutdown -h +MINUTES` in user-data, with the instance set to TERMINATE
#      on shutdown, so it dies even if this script is killed -9
set -euo pipefail

REMOTE_SCRIPT="${1:?usage: gpu_run.sh <remote-script> [instance-type] [max-minutes]}"
TYPE="${2:-g6.2xlarge}"
MAXMIN="${3:-360}"
REG="${AWS_REGION:-us-east-1}"
KEY=~/.ssh/systemone-ec2.pem
IID=""
STAMP=$(date +%Y%m%d-%H%M%S)

cleanup() {
  if [ -n "$IID" ]; then
    echo "--> terminating $IID"
    aws ec2 terminate-instances --region "$REG" --instance-ids "$IID" >/dev/null 2>&1 || true
    aws ec2 wait instance-terminated --region "$REG" --instance-ids "$IID" 2>/dev/null || true
    echo "--> terminated $IID"
  fi
}
trap cleanup EXIT INT TERM

MYIP=$(curl -s --max-time 10 https://checkip.amazonaws.com | tr -d '\n')
SG=$(aws ec2 describe-security-groups --region "$REG" --group-names systemone-sg \
       --query 'SecurityGroups[0].GroupId' --output text)
aws ec2 authorize-security-group-ingress --region "$REG" --group-id "$SG" \
  --protocol tcp --port 22 --cidr "$MYIP/32" >/dev/null 2>&1 || true

# Base AMI with the NVIDIA driver but no framework: the PyTorch DLAMIs are
# pinned to torch 2.1-2.3 and transformers 5.x needs newer.
AMI=$(aws ssm get-parameter --region "$REG" \
  --name /aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-amazon-linux-2023/latest/ami-id \
  --query Parameter.Value --output text)

UD=$(mktemp)
cat > "$UD" <<UDEOF
#!/bin/bash
shutdown -h +${MAXMIN}
exec > /var/log/setup.log 2>&1
set -x
dnf -y install python3.11 python3.11-pip git unzip
python3.11 -m venv /opt/so
/opt/so/bin/pip install --quiet --upgrade pip
/opt/so/bin/pip install --quiet torch --index-url https://download.pytorch.org/whl/cu128
/opt/so/bin/pip install --quiet 'transformers==5.17.0' numpy pandas datasets accelerate
chown -R ec2-user:ec2-user /opt/so
touch /tmp/SETUP_DONE
UDEOF

echo "--> launching $TYPE in $REG (hard shutdown in ${MAXMIN}m)"
IID=$(aws ec2 run-instances --region "$REG" --image-id "$AMI" --instance-type "$TYPE" \
  --key-name systemone-ec2 --security-group-ids "$SG" \
  --instance-initiated-shutdown-behavior terminate \
  --user-data "file://$UD" \
  --block-device-mappings 'DeviceName=/dev/xvda,Ebs={VolumeSize=100,VolumeType=gp3,DeleteOnTermination=true}' \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=systemone-gpu},{Key=ephemeral,Value=true}]' \
  --query 'Instances[0].InstanceId' --output text)
aws ec2 wait instance-running --region "$REG" --instance-ids "$IID"
DNS=$(aws ec2 describe-instances --region "$REG" --instance-ids "$IID" \
  --query 'Reservations[0].Instances[0].PublicDnsName' --output text)
echo "--> $IID at $DNS"

SSHO=(-i "$KEY" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null
      -o LogLevel=ERROR -o ConnectTimeout=10 -o ServerAliveInterval=30)

for _ in $(seq 1 60); do ssh "${SSHO[@]}" ec2-user@"$DNS" true 2>/dev/null && break; sleep 10; done
echo "--> ssh up; waiting for deps (torch + transformers, a few minutes)"
for _ in $(seq 1 120); do
  ssh "${SSHO[@]}" ec2-user@"$DNS" "test -f /tmp/SETUP_DONE" 2>/dev/null && break; sleep 15
done
ssh "${SSHO[@]}" ec2-user@"$DNS" "test -f /tmp/SETUP_DONE" || {
  echo "!! setup never finished; tail of /var/log/setup.log:"
  ssh "${SSHO[@]}" ec2-user@"$DNS" "sudo tail -40 /var/log/setup.log" || true
  exit 1
}

ssh "${SSHO[@]}" ec2-user@"$DNS" "nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader"

echo "--> syncing code"
ssh "${SSHO[@]}" ec2-user@"$DNS" "mkdir -p ~/work"
tar czf - systemone scripts tiny vision 2>/dev/null | \
  ssh "${SSHO[@]}" ec2-user@"$DNS" "tar xzf - -C ~/work"

echo "--> running $REMOTE_SCRIPT"
set +e
ssh "${SSHO[@]}" ec2-user@"$DNS" "cd ~/work && bash $REMOTE_SCRIPT 2>&1"
RC=$?
set -e
echo "--> remote exited $RC"

echo "--> fetching results"
mkdir -p "artifacts/ec2-$STAMP"
scp "${SSHO[@]}" -qr ec2-user@"$DNS":'~/work/artifacts/runs' "artifacts/ec2-$STAMP/" 2>/dev/null || \
  echo "   (no runs directory to fetch)"
ssh "${SSHO[@]}" ec2-user@"$DNS" "cd ~/work && tail -200 train.log" \
  > "artifacts/ec2-$STAMP/train.log" 2>/dev/null || true
echo "--> results in artifacts/ec2-$STAMP"
exit $RC
