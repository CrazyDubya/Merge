# Leonardo.ai Integration Monitoring

Automated health monitoring for the Leonardo.ai image generation integration.

## Features

- **Cost Tracking**: Daily and weekly spend monitoring with configurable thresholds
- **Error Rate Monitoring**: Tracks generation failures, alerts on high error rates
- **Disk Usage Monitoring**: Monitors cache directory growth, prevents runaway disk usage
- **Performance Monitoring**: Tracks generation times, detects API slowdowns
- **Automated Alerts**: Sends alerts to daemon inbox when thresholds are exceeded

## Quick Start

### Run Manual Health Check

```bash
cd ~/.claude/daemon/integrations/leonardo-ai
./monitoring/check-health.sh
```

### Run with Alerts Enabled

```bash
./monitoring/check-health.sh --alert
```

If any thresholds are exceeded, an alert will be sent to `~/.claude/daemon/inbox/daemon/unread/`.

### Run Specific Checks

```bash
# Check only costs
./monitoring/check-health.sh --cost-only

# Check only error rates
./monitoring/check-health.sh --errors-only

# Check only disk usage
./monitoring/check-health.sh --disk-only

# Check only performance
./monitoring/check-health.sh --performance-only
```

## Configuration

### Using Default Thresholds

Default thresholds are conservative and suitable for most use cases:

- **Daily cost**: $10.00
- **Weekly cost**: $50.00
- **Error rate**: 10%
- **Disk warning**: 500MB
- **Disk critical**: 1GB
- **Slow generation**: 30 seconds

### Customizing Thresholds

#### Option 1: Environment Variables (One-Time)

```bash
export LEONARDO_COST_DAILY_THRESHOLD=20.00
export LEONARDO_COST_WEEKLY_THRESHOLD=100.00
./monitoring/check-health.sh --alert
```

#### Option 2: Configuration File (Persistent)

```bash
cd monitoring/
cp config.example config
# Edit config with your preferred thresholds
vim config

# Use it
source config && ./check-health.sh --alert
```

#### Option 3: Inline (Quick Test)

```bash
LEONARDO_COST_DAILY_THRESHOLD=5.00 ./monitoring/check-health.sh --alert
```

### Available Threshold Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `LEONARDO_COST_DAILY_THRESHOLD` | Daily cost limit ($) | 10.00 |
| `LEONARDO_COST_WEEKLY_THRESHOLD` | Weekly cost limit ($) | 50.00 |
| `LEONARDO_ERROR_RATE_THRESHOLD` | Error rate % | 10 |
| `LEONARDO_DISK_WARNING_MB` | Disk warning (MB) | 500 |
| `LEONARDO_DISK_CRITICAL_MB` | Disk critical (MB) | 1000 |
| `LEONARDO_PERF_SLOW_SECONDS` | Slow generation (seconds) | 30 |

## Automated Monitoring with Cron

### Daily Health Check (Recommended)

Run health check once per day, send alerts if issues detected:

```bash
# Add to crontab
crontab -e

# Run daily at 9am
0 9 * * * cd ~/.claude/daemon/integrations/leonardo-ai && ./monitoring/check-health.sh --alert >> logs/monitoring.log 2>&1
```

### Hourly Monitoring (High Usage)

For heavy usage, check more frequently:

```bash
# Run every hour
0 * * * * cd ~/.claude/daemon/integrations/leonardo-ai && ./monitoring/check-health.sh --alert >> logs/monitoring.log 2>&1
```

### Weekly Summary (Low Usage)

For occasional usage, weekly checks are sufficient:

```bash
# Run every Monday at 9am
0 9 * * 1 cd ~/.claude/daemon/integrations/leonardo-ai && ./monitoring/check-health.sh --alert >> logs/monitoring.log 2>&1
```

## What Gets Monitored

### 1. Cost Tracking

Monitors:
- Daily spending (today's generations)
- Weekly spending (last 7 days)
- Total spending (all time)
- Generation count

Data source: `cost-tracking.jsonl`

Alerts when:
- Daily cost exceeds `LEONARDO_COST_DAILY_THRESHOLD`
- Weekly cost exceeds `LEONARDO_COST_WEEKLY_THRESHOLD`

### 2. Error Rate Monitoring

Monitors:
- Total generation attempts
- Successful generations (has image_path)
- Failed generations (missing image_path or has error field)
- Error percentage

Data source: `generations.jsonl`

Alerts when:
- Error rate exceeds `LEONARDO_ERROR_RATE_THRESHOLD`%

Shows last 5 errors with timestamps when alerting.

### 3. Disk Usage Monitoring

Monitors:
- Cache directory size (MB)
- Image count (jpg, png, jpeg, webp)
- Oldest and newest images

Data source: `cache/` directory

Alerts when:
- Warning level: Size exceeds `LEONARDO_DISK_WARNING_MB`
- Critical level: Size exceeds `LEONARDO_DISK_CRITICAL_MB`

### 4. Performance Monitoring

Monitors:
- Average generation time
- Slowest generation
- Count of slow generations (> threshold)

Data source: `generations.jsonl` (requires `generation_time_seconds` field)

Alerts when:
- Average generation time exceeds `LEONARDO_PERF_SLOW_SECONDS`

**Note**: Performance monitoring requires the integration to log `generation_time_seconds` in `generations.jsonl`. This may need to be added to the API wrapper.

## Alert Format

When `--alert` is enabled and issues are detected, an alert is sent to:

```
~/.claude/daemon/inbox/daemon/unread/leonardo-health-alert-YYYYMMDD-HHMMSS.md
```

Alert includes:
- List of issues with severity markers (⚠️ warning, 🔴 critical)
- Current status summary
- Recommended actions
- Timestamp

Example alert:

```markdown
# Leonardo.ai Health Alert

**Date**: 2025-11-21 15:30:00
**Source**: Leonardo.ai Integration Monitoring
**Severity**: 2 issue(s) detected

---

## Issues Detected

- 🔴 Daily cost ($12.50) exceeds threshold ($10.00)
- ⚠️ Disk usage (600MB) exceeds warning threshold (500MB)

---

## Current Status Summary

**Daily cost**: $12.50 (threshold: $10.00)
**Weekly cost**: $45.00 (threshold: $50.00)
**Error rate**: 5.0% (threshold: 10%)
**Disk usage**: 600MB (warning: 500MB, critical: 1000MB)

---

## Recommended Actions

1. Review cost tracking: `cat integrations/leonardo-ai/cost-tracking.jsonl`
2. Check recent errors: `cat integrations/leonardo-ai/generations.jsonl | jq 'select(.error != null)'`
3. Review disk usage: `du -sh integrations/leonardo-ai/cache/`
4. Run health check: `./integrations/leonardo-ai/monitoring/check-health.sh`
```

## Troubleshooting

### "No cost tracking data yet"

Normal for new installations. Generate some images first, then monitoring data will appear.

### "No timing data available"

Performance monitoring requires `generation_time_seconds` field in `generations.jsonl`. If not present, this check is skipped. Consider updating the API wrapper to log generation times.

### High error rates

1. Check recent errors: `jq 'select(.error != null)' generations.jsonl | tail -10`
2. Common causes:
   - API rate limiting (wait and retry)
   - Invalid API key (check `.config`)
   - Network issues (check connectivity)
   - API service issues (check leonardo.ai status)

### High disk usage

1. Check cache size: `du -sh cache/`
2. Review images: `ls -lh cache/ | tail -20`
3. Clean up old images if needed:
   ```bash
   # Delete images older than 30 days
   find cache/ -name "*.jpg" -mtime +30 -delete
   find cache/ -name "*.png" -mtime +30 -delete
   ```

### High costs

1. Review cost log: `cat cost-tracking.jsonl | jq`
2. Check daily totals: `jq -r "select(.timestamp | startswith(\"$(date +%Y-%m-%d)\")) | .estimated_cost" cost-tracking.jsonl | awk '{sum+=$1} END {print sum}'`
3. Identify expensive generations: `jq 'select(.estimated_cost > 0.10)' cost-tracking.jsonl`

## Integration with Dashboard

The monitoring system is designed to work with the daemon's dashboard. Alerts sent to the daemon inbox will be visible to all personas.

Future enhancement: Add monitoring widgets to the dashboard showing real-time cost, error rates, and disk usage.

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | All checks passed, no issues detected |
| 1 | One or more issues detected (thresholds exceeded) |

Use exit codes in scripts:

```bash
if ./monitoring/check-health.sh --alert; then
  echo "All systems healthy"
else
  echo "Issues detected, check alerts"
fi
```

## Files

| File | Purpose |
|------|---------|
| `check-health.sh` | Main monitoring script |
| `config.example` | Example configuration file |
| `README.md` | This file |

## Future Enhancements

- [ ] Add grafana/prometheus integration
- [ ] Email/SMS alerts for critical issues
- [ ] Dashboard widgets showing metrics
- [ ] Historical trend analysis
- [ ] Cost forecasting (predict monthly spend)
- [ ] Automatic cache cleanup when disk critical
- [ ] Integration test alerts (test endpoint health)

## Questions?

If you need help customizing thresholds or setting up monitoring, ask the human or check the main Leonardo.ai integration README.

---

**Created**: 2025-11-21
**Maintainer**: Maintainer persona
**Part of**: Leonardo.ai Image Generation Integration
