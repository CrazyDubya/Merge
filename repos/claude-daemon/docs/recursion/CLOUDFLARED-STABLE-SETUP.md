# Cloudflared Stable Tunnel Setup

## Summary

Successfully migrated from temporary cloudflared tunnel to a permanent, production-grade named tunnel with systemd service management.

## Tunnel Details

- **Tunnel Name**: claude-daemon
- **Tunnel ID**: 9d3b9931-2fc2-4c1e-a376-1a19e5999be4
- **Public URL**: https://daemon.claude-play.com/dashboard.html
- **Local Service**: http://localhost:8888
- **Account**: 18e002ebc857b38bc8fd572fee926f75

## Configuration Files

### Cloudflared Config
- **Config**: `/home/opc/.cloudflared/config.yml`
- **Credentials**: `/home/opc/.cloudflared/9d3b9931-2fc2-4c1e-a376-1a19e5999be4.json`
- **Tunnel Secret**: DlOFGBZ/xk110ATbr1jjfLPjcu8MAvw5ZVP6czHLS1o=

### DNS Configuration
- **Domain**: claude-play.com (Zone ID: 503a9ce9576d2ce339cbc6afaa147faf)
- **Subdomain**: daemon.claude-play.com
- **CNAME**: 9d3b9931-2fc2-4c1e-a376-1a19e5999be4.cfargotunnel.com
- **DNS Record ID**: 7c6b5568eecd7ec44c7296e14c2fdc8c

## Systemd Services

### Cloudflared Tunnel Service
- **Service File**: `/etc/systemd/system/cloudflared-tunnel.service`
- **Status**: `sudo systemctl status cloudflared-tunnel.service`
- **Logs**: `sudo journalctl -u cloudflared-tunnel.service -f`
- **Restart**: `sudo systemctl restart cloudflared-tunnel.service`

### Dashboard HTTP Server
- **Service File**: `/etc/systemd/system/dashboard-http-server.service`
- **Status**: `sudo systemctl status dashboard-http-server.service`
- **Logs**: `sudo journalctl -u dashboard-http-server.service -f`
- **Restart**: `sudo systemctl restart dashboard-http-server.service`

Both services are:
- ✅ Enabled (start on boot)
- ✅ Active and running
- ✅ Auto-restart on failure (RestartSec=10)

## Redundancy & Reliability

### Cloudflared Connections
The tunnel maintains 4 redundant connections to Cloudflare edge locations:
- iad15 (198.41.192.107)
- iad11 (198.41.200.63)
- iad08 (198.41.200.13)
- iad15 (198.41.192.167)

### Auto-Restart
- **Cloudflared**: Restarts automatically on failure
- **HTTP Server**: Restarts automatically on failure
- **Boot**: Both services start automatically on system boot

## Previous Issue Analysis

### Root Cause (2025-11-02)
The temporary Python HTTP server (PID 638388) became unresponsive after ~2.5 hours, returning "Empty reply from server" errors. This was **NOT** related to Claude Code session closing.

### Evidence
- Cloudflared temporary tunnel ran successfully for 2.5+ hours after Claude session ended
- Tunnel connection remained active (PID 637689, started 12:48)
- Only Python HTTP server became unresponsive (started 12:52, failed ~15:06)

### Conclusion
✅ Cloudflared sessions are **NOT** tied to Claude Code sessions
❌ Python `http.server` backgrounded processes can become unresponsive over time

### Solution
Migrated both services to systemd for proper process management, automatic restart, and boot persistence.

## Management Commands

### Check Status
```bash
# Both services
sudo systemctl status cloudflared-tunnel.service dashboard-http-server.service

# Test public access
curl -I https://daemon.claude-play.com/dashboard.html

# Test local access
curl -I http://localhost:8888/dashboard.html
```

### View Logs
```bash
# Follow cloudflared logs
sudo journalctl -u cloudflared-tunnel.service -f

# Follow HTTP server logs
sudo journalctl -u dashboard-http-server.service -f

# View recent errors
sudo journalctl -u cloudflared-tunnel.service -n 50 --no-pager
sudo journalctl -u dashboard-http-server.service -n 50 --no-pager
```

### Restart Services
```bash
# Restart cloudflared
sudo systemctl restart cloudflared-tunnel.service

# Restart HTTP server
sudo systemctl restart dashboard-http-server.service

# Restart both
sudo systemctl restart cloudflared-tunnel.service dashboard-http-server.service
```

### Disable/Enable
```bash
# Disable (don't start on boot)
sudo systemctl disable cloudflared-tunnel.service

# Enable (start on boot)
sudo systemctl enable cloudflared-tunnel.service
```

## Benefits Over Temporary Tunnel

| Feature | Temporary Tunnel | Named Tunnel |
|---------|-----------------|--------------|
| **URL Persistence** | Changes on restart | Permanent |
| **Connections** | 1 connection | 4 redundant connections |
| **Uptime Guarantee** | None (subject to removal) | Account-linked, reliable |
| **Survives Reboot** | No | Yes (systemd) |
| **Auto-Restart** | No | Yes (systemd) |
| **Process Management** | Manual backgrounding | Proper systemd |
| **Logging** | File-based | journalctl integration |
| **Security** | Basic | systemd hardening |

## API Token

**Token**: `__REDACTED__`
**Status**: Active and valid
**Permissions**: Tunnel management for account 18e002ebc857b38bc8fd572fee926f75

## Next Steps (Optional)

### 1. Add Cloudflare Access Authentication
Secure the dashboard with Cloudflare Access for authentication:
```bash
# In Cloudflare dashboard: Access > Applications > Add an application
```

### 2. Monitor Tunnel Health
Set up monitoring for tunnel status via Cloudflare API or dashboard.

### 3. Additional Services
Use the same tunnel for multiple services by updating ingress rules in `/home/opc/.cloudflared/config.yml`.

## Troubleshooting

### Tunnel Not Connecting
```bash
# Check cloudflared logs
sudo journalctl -u cloudflared-tunnel.service -n 50

# Verify credentials file exists
ls -la ~/.cloudflared/

# Test tunnel manually
cloudflared tunnel run claude-daemon
```

### HTTP Server Not Responding
```bash
# Check if port is in use
ss -tlnp | grep 8888

# Test local connection
curl -v http://localhost:8888/

# Check working directory exists
ls -la ~/.claude/daemon/
```

### 502 Bad Gateway
This means cloudflared is working but can't reach the backend:
```bash
# Restart HTTP server
sudo systemctl restart dashboard-http-server.service

# Verify it's listening
ss -tlnp | grep 8888
```

## Security Notes

- Credentials file: Contains tunnel secret, keep secure
- API token: Full tunnel management access, keep secure
- HTTP server: Currently no authentication, consider adding Cloudflare Access
- Systemd: Configured with `NoNewPrivileges`, `PrivateTmp`, and `ProtectSystem`

---

**Created**: 2025-11-02
**Status**: ✅ Production Ready
**Uptime**: Indefinite (systemd managed)
