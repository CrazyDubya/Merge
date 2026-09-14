# PRIMORDIAL TMUX UNLEASHED

## Quick Start

```bash
# Load the configuration
ln -sf /home/opc/recursion/.tmux.conf ~/.tmux.conf

# Launch a pre-configured environment
./tmux-unleashed.sh
```

## Key Bindings

### Essential Commands

| Key | Action |
|-----|--------|
| `Ctrl-a` | Prefix (instead of default Ctrl-b) |
| `Ctrl-a r` | Reload configuration |
| `Ctrl-a v` | Split vertically |
| `Ctrl-a s` | Split horizontally |
| `Ctrl-a x` | Close pane |
| `Ctrl-a z` | Zoom/unzoom pane |
| `Ctrl-a S` | Sync all panes (type in all at once) |

### Navigation (Vim-style)

| Key | Action |
|-----|--------|
| `Ctrl-a h/j/k/l` | Move between panes |
| `Ctrl-a H/J/K/L` | Resize panes |
| `Ctrl-a Tab` | Cycle through panes |
| `Ctrl-a Ctrl-h/l` | Switch windows |

### Copy Mode (Vim-style)

| Key | Action |
|-----|--------|
| `Ctrl-a [` | Enter copy mode |
| `v` | Start selection |
| `y` | Copy selection |
| `Enter` | Copy and exit |
| `q` | Exit copy mode |

## Pre-configured Environments

### 1. Development Environment
```bash
./tmux-unleashed.sh dev
```
Creates:
- Window 1: Editor with 3-pane layout
- Window 2: System monitoring (htop, disk, memory, uptime)
- Window 3: Git workspace
- Window 4: Log viewer
- Window 5: Shell

### 2. Matrix Display
```bash
./tmux-unleashed.sh matrix
```
Creates a 3x3 grid with random data streams.

### 3. System Dashboard
```bash
./tmux-unleashed.sh dash
```
Creates a monitoring dashboard with:
- System resources (htop)
- Network connections
- Disk usage

### 4. Recursion Demo
```bash
./tmux-unleashed.sh recursion
```
Creates a fractal-like pane layout showing nested levels.

## Advanced Features

### Mouse Support
Click to select panes, drag borders to resize, scroll to navigate history.

### Synchronized Panes
Press `Ctrl-a S` to type commands in all panes simultaneously. Perfect for:
- Multi-server administration
- Parallel testing
- Coordinated deployments

### Session Management
```bash
# Create named session
tmux new -s mysession

# List sessions
tmux ls

# Attach to session
tmux attach -t mysession

# Detach from session
Ctrl-a d

# Kill session
tmux kill-session -t mysession

# Kill all sessions
tmux kill-server
```

### Window Management
```bash
# Create window
Ctrl-a c

# Rename window
Ctrl-a ,

# Next/previous window
Ctrl-a n/p

# Select window by number
Ctrl-a 0-9
```

### Pane Management
```bash
# Rotate panes
Ctrl-a Ctrl-o

# Show pane numbers
Ctrl-a q

# Select pane by number
Ctrl-a q [number]

# Break pane into new window
Ctrl-a !

# Move pane to another window
Ctrl-a : move-pane -t [window]
```

## Power User Tips

### 1. Command Mode
Press `Ctrl-a :` to enter command mode for direct tmux commands:
```
:split-window -h
:resize-pane -D 10
:set-option status-position top
```

### 2. Layouts
Cycle through predefined layouts:
```bash
Ctrl-a Space    # Cycle layouts
Ctrl-a Alt-1    # Even horizontal
Ctrl-a Alt-2    # Even vertical
Ctrl-a Alt-3    # Main horizontal
Ctrl-a Alt-4    # Main vertical
Ctrl-a Alt-5    # Tiled
```

### 3. Scripting Sessions
Create custom workspace scripts:
```bash
#!/bin/bash
tmux new-session -d -s work
tmux send-keys -t work:1 'cd /project && vim' C-m
tmux split-window -h -t work:1
tmux send-keys -t work:1.2 'cd /project && git status' C-m
tmux attach -t work
```

### 4. Nested Sessions
Working on remote servers? Use `Ctrl-a a` to send prefix to nested session.

### 5. Save and Restore
To save/restore sessions, enable tmux-resurrect plugin in config.

## Color Scheme

The configuration uses a retro terminal aesthetic:
- **Primary**: Bright green (#00ff00)
- **Secondary**: Cyan (#00ffff)
- **Accent**: Magenta/Purple (#ff00ff)
- **Warning**: Yellow (#ffff00)
- **Background**: Dark gray (#1a1a1a)

## Customization

Edit `.tmux.conf` to customize:
- Change prefix key
- Modify color scheme
- Add custom key bindings
- Adjust status bar format
- Configure mouse behavior

## Troubleshooting

### Colors not working?
```bash
echo $TERM
# Should show "screen-256color" inside tmux
```

### Configuration not loading?
```bash
tmux kill-server
tmux source ~/.tmux.conf
```

### Mouse not working?
Ensure `set -g mouse on` is in your config and reload.

## Resources

- Config: `/home/opc/recursion/.tmux.conf`
- Script: `/home/opc/recursion/tmux-unleashed.sh`
- Man page: `man tmux`
- List all bindings: `Ctrl-a ?`

## Philosophy

This configuration embraces:
- **Vim keybindings**: Muscle memory across tools
- **Visual clarity**: High contrast, clear indicators
- **Speed**: Mouse support + keyboard mastery
- **Flexibility**: Easy to extend and customize
- **Power**: Synchronized panes, session management, scripting

Now go forth and multiplex with primordial power.
