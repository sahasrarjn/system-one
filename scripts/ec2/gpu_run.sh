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
  RC=$?
  [ -n "${SYNC_PID:-}" ] && kill "$SYNC_PID" 2>/dev/null || true
  if [ $RC -ne 0 ]; then echo "--> exiting with status $RC"; fi
  if [ -n "$IID" ]; then
    echo "--> terminating $IID"
    aws ec2 terminate-instances --region "$REG" --instance-ids "$IID" >/dev/null 2>&1 || true
    aws ec2 wait instance-terminated --region "$REG" --instance-ids "$IID" 2>/dev/null || true
    echo "--> terminated $IID"
  else
    echo "--> no instance was launched"
  fi
}
trap cleanup EXIT INT TERM

# checkip.amazonaws.com is not always reachable; try a few and fail loudly
# rather than silently proceeding with an empty CIDR. An earlier version let
# `set -e` kill the script here, which looked exactly like a clean no-op run.
MYIP=""
for SVC in https://api.ipify.org https://ifconfig.me/ip https://icanhazip.com https://checkip.amazonaws.com; do
  MYIP=$(curl -s --max-time 8 "$SVC" 2>/dev/null | tr -d '[:space:]' || true)
  if [[ "$MYIP" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then break; fi
  MYIP=""
done
if [ -z "$MYIP" ]; then echo "!! could not determine public IP; refusing to launch"; exit 1; fi
echo "--> my ip $MYIP"
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
# torchvision is NOT optional: the Qwen3-VL AutoProcessor pulls in its
# video processor, and the Pets loader uses torchvision.datasets. Its
# absence took out all four vision stages on the first GPU run.
/opt/so/bin/pip install --quiet torchvision --index-url https://download.pytorch.org/whl/cu128
/opt/so/bin/pip install --quiet 'transformers==5.17.0' numpy pandas datasets accelerate pillow
chown -R ec2-user:ec2-user /opt/so
touch /tmp/SETUP_DONE
UDEOF

# InsufficientInstanceCapacity is common for GPU types and is per-AZ, not
# regional. Walk a preference list of (market, type, AZ) rather than taking one
# shot. Spot goes first: it draws from a DIFFERENT capacity pool than
# on-demand, so it can succeed in exactly the situation where on-demand has
# nothing, and it costs roughly 40-60% less. g4dn is deliberately absent: the
# T4 is Turing and has no native bf16.
TYPES="$TYPE g5.2xlarge g6.xlarge g5.xlarge g6e.xlarge"
AZS="us-east-1a us-east-1b us-east-1c us-east-1d us-east-1f"
MARKETS="${MARKETS:-spot ondemand}"
IS_SPOT=0

echo "--> launching (hard shutdown in ${MAXMIN}m); markets: $MARKETS"
for MKT in $MARKETS; do
  # NOTE: bash 3.2 (the macOS default) errors on "${arr[@]}" for an EMPTY
  # array under `set -u`. The on-demand path uses an empty array, so the
  # unguarded expansion turned all 25 on-demand attempts into false
  # "no capacity" results. ${arr[@]+"${arr[@]}"} is the portable form.
  if [ "$MKT" = "spot" ]; then
    MKTOPT=(--instance-market-options MarketType=spot)
  else
    MKTOPT=()
  fi
  for T in $TYPES; do
    for AZ in $AZS; do
      SN=$(aws ec2 describe-subnets --region "$REG" \
        --filters "Name=default-for-az,Values=true" "Name=availability-zone,Values=$AZ" \
        --query 'Subnets[0].SubnetId' --output text 2>/dev/null)
      [ "$SN" = "None" ] || [ -z "$SN" ] && continue
      IID=$(aws ec2 run-instances --region "$REG" --image-id "$AMI" --instance-type "$T" \
        --key-name systemone-ec2 --security-group-ids "$SG" --subnet-id "$SN" \
        --instance-initiated-shutdown-behavior terminate \
        ${MKTOPT[@]+"${MKTOPT[@]}"} \
        --user-data "file://$UD" \
        --block-device-mappings 'DeviceName=/dev/xvda,Ebs={VolumeSize=100,VolumeType=gp3,DeleteOnTermination=true}' \
        --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=systemone-gpu},{Key=ephemeral,Value=true}]' \
        --query 'Instances[0].InstanceId' --output text 2>/dev/null) || IID=""
      if [ -n "$IID" ] && [ "$IID" != "None" ]; then
        TYPE="$T"
        if [ "$MKT" = "spot" ]; then IS_SPOT=1; fi
        echo "--> got $T in $AZ on $MKT: $IID"
        break 3
      fi
      IID=""
      printf "    no capacity: %-8s %-14s %s\n" "$MKT" "$T" "$AZ"
    done
  done
done
if [ -z "$IID" ]; then echo "!! no GPU capacity on any market/type/AZ"; exit 1; fi

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

mkdir -p "artifacts/ec2-$STAMP"
# A spot instance can be reclaimed with two minutes' notice, and the fetch at
# the end only runs if the script returns. Pull checkpoints and history every
# couple of minutes so an interruption costs the remaining steps rather than
# the whole run.
if [ "$IS_SPOT" = "1" ]; then
  ( while true; do
      sleep 120
      scp "${SSHO[@]}" -qr ec2-user@"$DNS":'~/work/artifacts/runs' \
        "artifacts/ec2-$STAMP/" 2>/dev/null || true
      ssh "${SSHO[@]}" ec2-user@"$DNS" \
        "cd ~/work && tar czf - artifacts/vision/*.json 2>/dev/null" \
        | tar xzf - -C "artifacts/ec2-$STAMP" 2>/dev/null || true
    done ) &
  SYNC_PID=$!
  echo "--> spot: syncing artifacts every 120s (pid $SYNC_PID)"
fi

echo "--> running $REMOTE_SCRIPT"
set +e
ssh "${SSHO[@]}" ec2-user@"$DNS" "cd ~/work && bash $REMOTE_SCRIPT 2>&1"
RC=$?
set -e
[ -n "${SYNC_PID:-}" ] && kill "$SYNC_PID" 2>/dev/null || true
echo "--> remote exited $RC"

echo "--> fetching results"
# text results live in artifacts/runs, vision results in artifacts/vision.
# Skip the vision .npz feature caches: they are hundreds of MB and trivially
# regenerated, while the json/pt outputs are what carry the findings.
scp "${SSHO[@]}" -qr ec2-user@"$DNS":'~/work/artifacts/runs' "artifacts/ec2-$STAMP/" 2>/dev/null || \
  echo "   (no runs directory to fetch)"
ssh "${SSHO[@]}" ec2-user@"$DNS" "cd ~/work && tar czf - artifacts/vision/*.json artifacts/vision/*.pt 2>/dev/null" \
  | tar xzf - -C "artifacts/ec2-$STAMP" 2>/dev/null || echo "   (no vision results to fetch)"
ssh "${SSHO[@]}" ec2-user@"$DNS" "cd ~/work && tail -200 train.log" \
  > "artifacts/ec2-$STAMP/train.log" 2>/dev/null || true
echo "--> results in artifacts/ec2-$STAMP"
exit $RC
