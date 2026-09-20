#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------------
# Grant the Databricks App's service principal READ access to a secret scope.
#
# Usage:
#   bash scripts/grant_secret_scope_acl.sh <app_name> <secret_scope>
#
# Example:
#   bash scripts/grant_secret_scope_acl.sh ecom-agent-app dbx-secret-scope
#
# Requires: Databricks CLI installed and authenticated.
# ---------------------------------------------------------------------------

APP_NAME="${1:?Usage: $0 <app_name> <secret_scope>}"
SCOPE="${2:?Usage: $0 <app_name> <secret_scope>}"

echo "Looking up service principal for app: ${APP_NAME}"

SP_ID=$(databricks apps get "${APP_NAME}" --output JSON 2>/dev/null \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['service_principal_client_id'])")

if [ -z "${SP_ID}" ]; then
  echo "ERROR: Could not find service principal for app '${APP_NAME}'."
  exit 1
fi

echo "App service principal: ${SP_ID}"
echo "Granting READ on scope '${SCOPE}'..."

databricks secrets put-acl "${SCOPE}" "${SP_ID}" READ

echo "Done. Current ACLs for '${SCOPE}':"
databricks secrets list-acls "${SCOPE}" --output JSON