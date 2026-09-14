#!/bin/bash
# Automated Cloudflare Access Setup for daemon.claude-play.com
# This implements email-based authentication WITHOUT stopping the dashboard

set -e

# Configuration from existing tunnel
ZONE_NAME="claude-play.com"
ZONE_ID="503a9ce9576d2ce339cbc6afaa147faf"
HOSTNAME="daemon.claude-play.com"
ACCOUNT_ID="18e002ebc857b38bc8fd572fee926f75"
CF_API_TOKEN="__REDACTED__"

# Default email (can be overridden as argument)
DEFAULT_EMAIL="${1:-user@example.com}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}════════════════════════════════════════════════${NC}"
echo -e "${GREEN}   Cloudflare Access Setup - Daemon Dashboard${NC}"
echo -e "${BLUE}════════════════════════════════════════════════${NC}"
echo ""
echo "Protected URL: https://${HOSTNAME}"
echo "Account ID: ${ACCOUNT_ID}"
echo ""

# Verify API token works
echo -e "${YELLOW}[1/4] Verifying API token...${NC}"
TOKEN_TEST=$(curl -s -X GET "https://api.cloudflare.com/client/v4/user/tokens/verify" \
    -H "Authorization: Bearer ${CF_API_TOKEN}" \
    -H "Content-Type: application/json")

TOKEN_STATUS=$(echo "$TOKEN_TEST" | jq -r '.success')
if [ "$TOKEN_STATUS" != "true" ]; then
    echo -e "${RED}✗ API token verification failed${NC}"
    echo "Response: $TOKEN_TEST"
    exit 1
fi
echo -e "${GREEN}✓ API token verified${NC}"
echo ""

# Check if Access application already exists
echo -e "${YELLOW}[2/4] Checking for existing Access applications...${NC}"
EXISTING_APPS=$(curl -s -X GET "https://api.cloudflare.com/client/v4/accounts/${ACCOUNT_ID}/access/apps" \
    -H "Authorization: Bearer ${CF_API_TOKEN}" \
    -H "Content-Type: application/json")

APP_EXISTS=$(echo "$EXISTING_APPS" | jq -r --arg domain "$HOSTNAME" '.result[] | select(.domain == $domain) | .id')

if [ -n "$APP_EXISTS" ] && [ "$APP_EXISTS" != "null" ]; then
    echo -e "${YELLOW}⚠ Access application already exists: ${APP_EXISTS}${NC}"
    echo "To reconfigure, delete the existing application in Cloudflare dashboard first."
    echo ""
    echo "Existing configuration:"
    echo "$EXISTING_APPS" | jq -r --arg domain "$HOSTNAME" '.result[] | select(.domain == $domain)'
    exit 0
fi
echo -e "${GREEN}✓ No existing Access application found${NC}"
echo ""

# Create Access Application
echo -e "${YELLOW}[3/4] Creating Access Application...${NC}"
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
        "http_only_cookie_attribute": true,
        "same_site_cookie_attribute": "strict",
        "logo_url": "",
        "app_launcher_visible": true
    }')

APP_SUCCESS=$(echo "$ACCESS_APP" | jq -r '.success')
APP_ID=$(echo "$ACCESS_APP" | jq -r '.result.id')

if [ "$APP_SUCCESS" != "true" ] || [ "$APP_ID" = "null" ] || [ -z "$APP_ID" ]; then
    echo -e "${RED}✗ Failed to create Access application${NC}"
    echo "Response:"
    echo "$ACCESS_APP" | jq '.'
    exit 1
fi

echo -e "${GREEN}✓ Access application created${NC}"
echo "  Application ID: ${APP_ID}"
echo ""

# Create Access Policy
echo -e "${YELLOW}[4/4] Creating Access Policy (email-based)...${NC}"
echo "Configuring access for email: ${DEFAULT_EMAIL}"

ACCESS_POLICY=$(curl -s -X POST "https://api.cloudflare.com/client/v4/accounts/${ACCOUNT_ID}/access/apps/${APP_ID}/policies" \
    -H "Authorization: Bearer ${CF_API_TOKEN}" \
    -H "Content-Type: application/json" \
    -d '{
        "name": "Allow Authorized Emails",
        "decision": "allow",
        "include": [
            {
                "email": {
                    "email": "'"${DEFAULT_EMAIL}"'"
                }
            }
        ],
        "require": [],
        "exclude": []
    }')

POLICY_SUCCESS=$(echo "$ACCESS_POLICY" | jq -r '.success')
POLICY_ID=$(echo "$ACCESS_POLICY" | jq -r '.result.id')

if [ "$POLICY_SUCCESS" != "true" ] || [ "$POLICY_ID" = "null" ] || [ -z "$POLICY_ID" ]; then
    echo -e "${RED}✗ Failed to create Access policy${NC}"
    echo "Response:"
    echo "$ACCESS_POLICY" | jq '.'
    # Clean up the application
    curl -s -X DELETE "https://api.cloudflare.com/client/v4/accounts/${ACCOUNT_ID}/access/apps/${APP_ID}" \
        -H "Authorization: Bearer ${CF_API_TOKEN}" > /dev/null
    exit 1
fi

echo -e "${GREEN}✓ Access policy created${NC}"
echo "  Policy ID: ${POLICY_ID}"
echo ""

# Save configuration
CONFIG_FILE="/home/opc/.claude/daemon/security/cloudflare-access-config.json"
mkdir -p "$(dirname "$CONFIG_FILE")"
cat > "$CONFIG_FILE" <<EOF
{
  "application_id": "${APP_ID}",
  "policy_id": "${POLICY_ID}",
  "hostname": "${HOSTNAME}",
  "allowed_emails": ["${DEFAULT_EMAIL}"],
  "configured_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "session_duration": "24h",
  "zone_id": "${ZONE_ID}",
  "account_id": "${ACCOUNT_ID}"
}
EOF
chmod 600 "$CONFIG_FILE"

echo -e "${GREEN}════════════════════════════════════════════════${NC}"
echo -e "${GREEN}         SUCCESS! Authentication Enabled${NC}"
echo -e "${GREEN}════════════════════════════════════════════════${NC}"
echo ""
echo "Configuration Details:"
echo "  Protected URL: https://${HOSTNAME}"
echo "  Application ID: ${APP_ID}"
echo "  Policy ID: ${POLICY_ID}"
echo "  Authorized Email: ${DEFAULT_EMAIL}"
echo "  Session Duration: 24 hours"
echo "  Config File: ${CONFIG_FILE}"
echo ""
echo -e "${BLUE}How to Access:${NC}"
echo "  1. Visit: https://${HOSTNAME}/dashboard.html"
echo "  2. Enter your email: ${DEFAULT_EMAIL}"
echo "  3. Check email for one-time code"
echo "  4. Enter code to access dashboard"
echo "  5. Session lasts 24 hours"
echo ""
echo -e "${BLUE}To add more emails:${NC}"
echo "  Edit the policy in Cloudflare dashboard:"
echo "  https://one.dash.cloudflare.com/${ACCOUNT_ID}/access/apps/${APP_ID}"
echo ""
echo -e "${BLUE}Services Status:${NC}"
systemctl is-active cloudflared-tunnel.service >/dev/null && echo -e "  Cloudflared Tunnel: ${GREEN}✓ Running${NC}" || echo -e "  Cloudflared Tunnel: ${RED}✗ Not Running${NC}"
systemctl is-active dashboard-http-server.service >/dev/null && echo -e "  Dashboard Server: ${GREEN}✓ Running${NC}" || echo -e "  Dashboard Server: ${RED}✗ Not Running${NC}"
echo ""
echo -e "${GREEN}✓ Authentication configured without service interruption${NC}"
echo ""
