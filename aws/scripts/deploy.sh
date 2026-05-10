#!/usr/bin/env bash
# PokerMind ECS deployment script
# Usage: AWS_ACCOUNT_ID=123456789012 ./aws/scripts/deploy.sh [--region us-east-1]
set -euo pipefail

ACCOUNT_ID="${AWS_ACCOUNT_ID:?'Set AWS_ACCOUNT_ID'}"
REGION="${AWS_REGION:-us-east-1}"
CLUSTER="pokermind"
SERVICES=("strategist" "historian" "humanizer" "orchestrator")
ECR_BASE="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"

echo "==> Authenticating with ECR"
aws ecr get-login-password --region "$REGION" | \
  docker login --username AWS --password-stdin "${ECR_BASE}"

echo "==> Creating ECR repositories (idempotent)"
for svc in "${SERVICES[@]}"; do
  aws ecr describe-repositories --repository-names "pokermind-${svc}" \
    --region "$REGION" 2>/dev/null || \
  aws ecr create-repository --repository-name "pokermind-${svc}" \
    --region "$REGION" --image-scanning-configuration scanOnPush=true
done

echo "==> Building and pushing images"
for svc in "${SERVICES[@]}"; do
  TAG="${ECR_BASE}/pokermind-${svc}:latest"
  echo "  Building pokermind-${svc}..."
  docker build \
    -f "services/${svc}/Dockerfile" \
    -t "${TAG}" \
    .
  docker push "${TAG}"
done

echo "==> Creating CloudWatch log groups"
for svc in "${SERVICES[@]}"; do
  aws logs create-log-group --log-group-name "/pokermind/${svc}" \
    --region "$REGION" 2>/dev/null || true
  aws logs put-retention-policy \
    --log-group-name "/pokermind/${svc}" \
    --retention-in-days 14 \
    --region "$REGION"
done

echo "==> Registering task definitions"
for svc in "${SERVICES[@]}"; do
  DEF_FILE="aws/task-definitions/${svc}.json"
  # Substitute placeholder account ID
  PATCHED=$(sed "s/ACCOUNT_ID/${ACCOUNT_ID}/g" "$DEF_FILE")
  aws ecs register-task-definition \
    --cli-input-json "$PATCHED" \
    --region "$REGION"
  echo "  Registered pokermind-${svc}"
done

echo "==> Creating ECS cluster (if not exists)"
aws ecs describe-clusters --clusters "$CLUSTER" --region "$REGION" \
  | grep -q '"status": "ACTIVE"' || \
aws ecs create-cluster \
  --cluster-name "$CLUSTER" \
  --capacity-providers FARGATE FARGATE_SPOT \
  --default-capacity-provider-strategy \
    capacityProvider=FARGATE_SPOT,weight=3 \
    capacityProvider=FARGATE,weight=1 \
  --region "$REGION"

echo "==> Updating ECS services"
for svc in "${SERVICES[@]}"; do
  aws ecs describe-services \
    --cluster "$CLUSTER" --services "pokermind-${svc}" \
    --region "$REGION" | grep -q '"status": "ACTIVE"' \
  && aws ecs update-service \
      --cluster "$CLUSTER" \
      --service "pokermind-${svc}" \
      --task-definition "pokermind-${svc}" \
      --force-new-deployment \
      --region "$REGION" \
  || aws ecs create-service \
      --cluster "$CLUSTER" \
      --service-name "pokermind-${svc}" \
      --task-definition "pokermind-${svc}" \
      --desired-count 1 \
      --launch-type FARGATE \
      --network-configuration "awsvpcConfiguration={subnets=[SUBNET_ID],securityGroups=[SG_ID],assignPublicIp=ENABLED}" \
      --region "$REGION"
done

echo "==> Deployment complete"
echo "    Monitor: aws ecs describe-services --cluster $CLUSTER --services ${SERVICES[*]} --region $REGION"
