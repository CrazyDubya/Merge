#!/bin/bash
# Install pre-commit security hook for Tier 1 write protection
#
# This script installs a git pre-commit hook that runs the Tier 1 security
# scanner before allowing commits. This is OPT-IN - you choose to install it.
#
# Usage: ./scripts/install-security-hooks.sh
#
# To bypass: git commit --no-verify (use responsibly)
# To remove: rm .git/hooks/pre-commit

set -e

DAEMON_ROOT="${HOME}/.claude/daemon"
HOOK_PATH="${DAEMON_ROOT}/.git/hooks/pre-commit"

echo "🔒 Installing Tier 1 Security Pre-Commit Hook"
echo ""

# Check if scanner exists
if [[ ! -f "${DAEMON_ROOT}/scripts/scan-tier1-writes.sh" ]]; then
    echo "❌ ERROR: Scanner not found at scripts/scan-tier1-writes.sh"
    exit 1
fi

# Check if hook already exists
if [[ -f "$HOOK_PATH" ]]; then
    echo "⚠️  Pre-commit hook already exists at .git/hooks/pre-commit"
    echo ""
    echo "Options:"
    echo "  1. Backup and replace: mv .git/hooks/pre-commit .git/hooks/pre-commit.backup"
    echo "  2. Manually merge the hooks"
    echo "  3. Cancel this installation"
    echo ""
    read -p "Replace existing hook? [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Installation cancelled."
        exit 0
    fi
    mv "$HOOK_PATH" "${HOOK_PATH}.backup"
    echo "✓ Existing hook backed up to .git/hooks/pre-commit.backup"
fi

# Create the hook
cat > "$HOOK_PATH" << 'HOOK_EOF'
#!/bin/bash
# Pre-commit hook: Run Tier 1 security scanner
# Installed by: scripts/install-security-hooks.sh
# Purpose: Detect unprotected writes to Tier 1 files before commit

DAEMON_ROOT="${HOME}/.claude/daemon"

echo "🔒 Running Tier 1 security scanner..."

if ! "$DAEMON_ROOT/scripts/scan-tier1-writes.sh"; then
    echo ""
    echo "❌ SECURITY CHECK FAILED"
    echo ""
    echo "Unprotected writes to Tier 1 files detected!"
    echo "Please use atomic_append from lib/atomic-io.sh"
    echo ""
    echo "Options:"
    echo "  1. Fix the gaps (recommended)"
    echo "  2. Bypass: git commit --no-verify (see docs/proposals/security-hook-governance.md)"
    echo "  3. Remove hook: rm .git/hooks/pre-commit"
    echo ""
    exit 1
fi

echo "✅ Security check passed!"
exit 0
HOOK_EOF

chmod +x "$HOOK_PATH"

echo ""
echo "✅ Pre-commit hook installed successfully!"
echo ""
echo "What happens now:"
echo "  • Every commit will run the Tier 1 security scanner"
echo "  • Commits with unprotected Tier 1 writes will be blocked"
echo "  • You can bypass with: git commit --no-verify (use responsibly)"
echo "  • You can remove with: rm .git/hooks/pre-commit"
echo ""
echo "Test it by running: git commit (it will run the scanner)"
echo ""
echo "For governance guidelines, see: docs/proposals/security-hook-governance.md"
echo ""

exit 0
