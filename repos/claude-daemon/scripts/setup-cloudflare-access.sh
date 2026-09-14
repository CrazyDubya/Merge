#!/bin/bash
# Setup Cloudflare Access for daemon.claude-play.com
# This script configures email-based authentication for the dashboard

set -e

# Configuration
ZONE_NAME="claude-play.com"
HOSTNAME="daemon.claude-play.com"
ACCOUNT_ID="18e002ebc857b38bc8fd572fee926f75"  # From tunnel credentials

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Cloudflare Access Setup for ${HOSTNAME}${NC}"
echo "=============================================="
echo ""

# Check if CF_API_TOKEN is set
if [ -z "$CF_API_TOKEN" ]; then
    echo -e "${RED}ERROR: CF_API_TOKEN environment variable not set${NC}"
    echo ""
    echo "Please obtain a Cloudflare API token with the following permissions:"
    echo "  - Account > Cloudflare Access > Edit"
    echo "  - Zone > Zone > Read"
    echo ""
    echo "Then run: export CF_API_TOKEN='your-token-here'"
    echo "          $0"
    exit 1
fi

echo -e "${YELLOW}Step 1: Getting Zone ID${NC}"
ZONE_ID=$(curl -s -X GET "https://api.cloudflare.com/client/v4/zones?name=${ZONE_NAME}" \
    -H "Authorization: Bearer ${CF_API_TOKEN}" \
    -H "Content-Type: application/json" | jq -r '.result[0].id')

if [ "$ZONE_ID" = "null" ] || [ -z "$ZONE_ID" ]; then
    echo -e "${RED}ERROR: Could not fetch zone ID for ${ZONE_NAME}${NC}"
    echo "Please check your API token permissions"
    exit 1
fi

echo -e "${GREEN}✓ Zone ID: ${ZONE_ID}${NC}"
echo ""

echo -e "${YELLOW}Step 2: Creating Access Application${NC}"
ACCESS_APP=$(curl -s -X POST "https://api.cloudflare.com/client/v4/accounts/${ACCOUNT_ID}/access/apps" \
    -H "Authorization: Bearer ${CF_API_TOKEN}" \
    -H "Content-Type: application/json" \
    -d '{
        "name": "Claude Daemon Dashboard",
        "domain": "'"${HOSTNAME}"'",
        "type": "self_hosted",
        "session_duration": "24h",
        "auto_redirect_to_identity": false,
        "enable_binding_cookie": false,
        "allowed_idps": [],
        "cors_headers": {
            "allow_all_origins": false,
            "allow_all_methods": false,
            "allow_all_headers": false,
            "allow_credentials": false
        }
    }')

APP_ID=$(echo "$ACCESS_APP" | jq -r '.result.id')

if [ "$APP_ID" = "null" ] || [ -z "$APP_ID" ]; then
    echo -e "${RED}ERROR: Could not create Access application${NC}"
    echo "Response: $ACCESS_APP"
    exit 1
fi

echo -e "${GREEN}✓ Access Application created: ${APP_ID}${NC}"
echo ""

echo -e "${YELLOW}Step 3: Creating Access Policy (Email-based)${NC}"
echo "Enter the email addresses that should have access (comma-separated):"
read -p "Emails: " ALLOWED_EMAILS

# Convert comma-separated emails to JSON array
IFS=',' read -ra EMAIL_ARRAY <<< "$ALLOWED_EMAILS"
EMAIL_JSON="["
for email in "${EMAIL_ARRAY[@]}"; do
    email=$(echo "$email" | xargs)  # Trim whitespace
    EMAIL_JSON+="\"$email\","
done
EMAIL_JSON="${EMAIL_JSON%,}]"  # Remove trailing comma and close array

ACCESS_POLICY=$(curl -s -X POST "https://api.cloudflare.com/client/v4/accounts/${ACCOUNT_ID}/access/apps/${APP_ID}/policies" \
    -H "Authorization: Bearer ${CF_API_TOKEN}" \
    -H "Content-Type: application/json" \
    -d '{
        "name": "Allow Specific Emails",
        "decision": "allow",
        "include": [
            {
                "email": {
                    "email": '"${EMAIL_JSON}"'
                }
            }
        ]
    }')

POLICY_ID=$(echo "$ACCESS_POLICY" | jq -r '.result.id')

if [ "$POLICY_ID" = "null" ] || [ -z "$POLICY_ID" ]; then
    echo -e "${RED}ERROR: Could not create Access policy${NC}"
    echo "Response: $ACCESS_POLICY"
    exit 1
fi

echo -e "${GREEN}✓ Access Policy created: ${POLICY_ID}${NC}"
echo ""

echo -e "${GREEN}SUCCESS! Cloudflare Access is now configured${NC}"
echo "=============================================="
echo ""
echo "Configuration Details:"
echo "  Application ID: ${APP_ID}"
echo "  Policy ID: ${POLICY_ID}"
echo "  Protected URL: https://${HOSTNAME}"
echo "  Allowed Emails: ${ALLOWED_EMAILS}"
echo ""
echo "The dashboard is now protected with email-based authentication."
echo "Users will receive a one-time code to their email to log in."
echo ""
echo "To test:"
echo "  1. Visit https://${HOSTNAME}/dashboard.html"
echo "  2. Enter your email address"
echo "  3. Check your email for the one-time code"
echo "  4. Enter the code to access the dashboard"
echo ""

# Save configuration
CONFIG_FILE="/home/opc/.claude/daemon/security/cloudflare-access-config.json"
mkdir -p "$(dirname "$CONFIG_FILE")"
cat > "$CONFIG_FILE" <<EOF
{
  "application_id": "${APP_ID}",
  "policy_id": "${POLICY_ID}",
  "hostname": "${HOSTNAME}",
  "allowed_emails": ${EMAIL_JSON},
  "configured_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "session_duration": "24h"
}
EOF

chmod 600 "$CONFIG_FILE"
echo -e "${GREEN}✓ Configuration saved to: ${CONFIG_FILE}${NC}"
