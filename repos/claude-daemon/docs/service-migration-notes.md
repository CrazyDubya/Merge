# Service Migration Notes

This document records service configuration changes and migrations for the daemon system.

---

## 2025-11-21: Dashboard Server Migration (Simple HTTP → CGI)

### Background

The dashboard was originally served by a simple Python HTTP server (`python3 -m http.server 8888`) managed by systemd service `dashboard-http-server.service`.

To support dynamic API endpoints (inbox counts, task parsing, etc.), we migrated to a CGI-capable server (`dashboard-server-cgi.py`) that can execute bash scripts as API endpoints.

### Migration Timeline

- **Before Nov 2025**: Simple HTTP server serving static files only
- **Early Nov 2025**: CGI server deployed for API support
- **Nov 21, 2025**: Old systemd service cleanup (this document)

### The Problem

After deploying the CGI server, the old `dashboard-http-server.service` was never disabled. This created:

1. **Port conflict**: Both services tried to bind to port 8888
2. **Restart loop**: Old service failed 10,644+ times trying to start
3. **Log noise**: Every restart attempt created log entries
4. **Resource waste**: Systemd continuously attempting to restart
5. **Configuration confusion**: Two services configured for same purpose

**Impact**: Operational issue (not security issue). Dashboard worked correctly because CGI server won the port race.

### Investigation Details

**Discovery**: Service state monitoring alert (2025-11-21T21:31:23Z) detected `dashboard-http-server.service` state changes.

**Root cause analysis**:
```bash
# Old service status
$ sudo systemctl status dashboard-http-server.service
● dashboard-http-server.service - Dashboard HTTP Server
  Active: activating (auto-restart) (Result: exit-code)
  # Failed 10,644+ times

# Port 8888 already in use
$ sudo lsof -i :8888
python3 2868882 opc  3u  IPv4 TCP *:ddi-tcp-1 (LISTEN)
# CGI server running correctly

# Error in logs
OSError: [Errno 98] Address already in use
```

**Conclusion**: Migration to CGI server completed successfully, but cleanup step (disable old service) was missed.

### Resolution

**Date**: 2025-11-21T23:16:23Z
**Performed by**: Maintainer persona
**Actions taken**:

```bash
# Stop the failing service
sudo systemctl stop dashboard-http-server.service

# Disable it permanently (prevent auto-start)
sudo systemctl disable dashboard-http-server.service
# Removed: /etc/systemd/system/multi-user.target.wants/dashboard-http-server.service
```

**Verification**:

```bash
# Service now inactive and disabled
$ sudo systemctl status dashboard-http-server.service
○ dashboard-http-server.service
  Loaded: loaded (...; disabled; preset: disabled)
  Active: inactive (dead)

# Only CGI server on port 8888
$ sudo lsof -i :8888
python3 2868882 opc  3u  IPv4 TCP *:ddi-tcp-1 (LISTEN)

# Dashboard works correctly
$ curl -I http://localhost:8888/dashboard.html
HTTP/1.0 200 OK
```

### Current State (After Cleanup)

**Active services**:
- `dashboard-server-cgi.py` (PID 2868882): CGI server on port 8888
- Serves static files (dashboard.html, inbox.html, tasks.html)
- Executes API endpoints (api/*.sh scripts)

**Disabled services**:
- `dashboard-http-server.service`: Stopped and disabled (will not auto-start)

**Dashboard functionality**:
- ✅ Main dashboard: http://localhost:8888/dashboard.html
- ✅ Inbox viewer: http://localhost:8888/inbox.html
- ✅ Task viewer: http://localhost:8888/tasks.html
- ✅ API endpoints: http://localhost:8888/api/*

### Rollback Plan

If needed, the old service can be re-enabled:

```bash
# Stop CGI server
pkill -f dashboard-server-cgi.py

# Re-enable old service
sudo systemctl enable dashboard-http-server.service
sudo systemctl start dashboard-http-server.service
```

**Note**: Rolling back would lose API functionality (inbox counts, task parsing, etc.).

### Lessons Learned

1. **Complete migrations**: When deploying new services, explicitly disable old ones
2. **Migration checklists**: Create checklist for service migrations:
   - [ ] Deploy new service
   - [ ] Verify new service working
   - [ ] Stop old service
   - [ ] Disable old service
   - [ ] Verify functionality
   - [ ] Document migration
3. **Monitoring value**: Service state monitoring caught the incomplete migration
4. **No harm from delay**: The restart loop was annoying but not harmful (dashboard worked)

### Related Documentation

- **Investigation report**: inbox/human/unread/response-20251121-220800-from-maintainer.md
- **Service monitoring**: experiments/service-state-monitoring-addition.sh
- **CGI server**: dashboard-server-cgi.py
- **API endpoints**: api/inbox-counts.sh, api/tasks.sh

### Future Service Migrations

When migrating services, follow this process:

1. **Deploy new service** (test in parallel if possible)
2. **Verify new service works** (functional testing)
3. **Stop old service** (`systemctl stop`)
4. **Verify functionality** (ensure no regression)
5. **Disable old service** (`systemctl disable`)
6. **Document migration** (update this file)
7. **Monitor for issues** (24-48 hours)

### Service Inventory

**Current active services** (as of 2025-11-21):

| Service | Purpose | Port | Status | Managed by |
|---------|---------|------|--------|------------|
| dashboard-server-cgi.py | Dashboard + API | 8888 | Active | Manual (tmux) |
| claude-daemon.service | Main daemon | - | Active | systemd (user) |
| simple-ssh-monitor.service | Service monitoring | - | Active | systemd |
| cloudflared-tunnel.service | Cloudflare tunnel | - | Active | systemd |

**Disabled services**:

| Service | Purpose | Disabled | Reason |
|---------|---------|----------|--------|
| dashboard-http-server.service | Old dashboard | 2025-11-21 | Replaced by CGI server |

---

**Document created**: 2025-11-21T23:16:33Z
**Last updated**: 2025-11-21T23:16:33Z
**Maintainer**: Maintainer persona
