#!/bin/bash
# PRIMORDIAL TMUX UNLEASHED
# Automation script for maximum productivity

SESSION_NAME="primordial"

# Function to create a development environment
create_dev_env() {
    tmux new-session -d -s "$SESSION_NAME" -n "editor"

    # Window 1: Editor (3-pane layout)
    tmux send-keys -t "$SESSION_NAME:1" "echo 'Editor pane ready'" C-m
    tmux split-window -h -t "$SESSION_NAME:1"
    tmux send-keys -t "$SESSION_NAME:1.2" "echo 'Side panel ready'" C-m
    tmux split-window -v -t "$SESSION_NAME:1.2"
    tmux send-keys -t "$SESSION_NAME:1.3" "echo 'Terminal ready'" C-m
    tmux select-pane -t "$SESSION_NAME:1.1"

    # Window 2: Monitoring (4-pane grid)
    tmux new-window -t "$SESSION_NAME:2" -n "monitor"
    tmux send-keys -t "$SESSION_NAME:2" "htop" C-m
    tmux split-window -h -t "$SESSION_NAME:2"
    tmux send-keys -t "$SESSION_NAME:2.2" "watch -n 1 'df -h'" C-m
    tmux split-window -v -t "$SESSION_NAME:2.1"
    tmux send-keys -t "$SESSION_NAME:2.3" "watch -n 1 'free -h'" C-m
    tmux split-window -v -t "$SESSION_NAME:2.2"
    tmux send-keys -t "$SESSION_NAME:2.4" "watch -n 1 'uptime'" C-m

    # Window 3: Git
    tmux new-window -t "$SESSION_NAME:3" -n "git"
    tmux send-keys -t "$SESSION_NAME:3" "git status" C-m

    # Window 4: Logs
    tmux new-window -t "$SESSION_NAME:4" -n "logs"
    tmux send-keys -t "$SESSION_NAME:4" "echo 'Log viewer ready'" C-m

    # Window 5: Terminal
    tmux new-window -t "$SESSION_NAME:5" -n "shell"

    # Return to first window
    tmux select-window -t "$SESSION_NAME:1"

    # Attach to session
    tmux attach-session -t "$SESSION_NAME"
}

# Function to create a matrix-style display
create_matrix() {
    SESSION_NAME="matrix"
    tmux new-session -d -s "$SESSION_NAME"

    # Create a 3x3 grid
    tmux split-window -h -t "$SESSION_NAME"
    tmux split-window -h -t "$SESSION_NAME"
    tmux select-pane -t 0
    tmux split-window -v -t "$SESSION_NAME"
    tmux select-pane -t 2
    tmux split-window -v -t "$SESSION_NAME"
    tmux select-pane -t 4
    tmux split-window -v -t "$SESSION_NAME"

    # Fill with cool stuff
    for i in {0..5}; do
        tmux send-keys -t "$SESSION_NAME.$i" "while true; do echo \$RANDOM | md5sum | head -c 20; echo; sleep 0.1; done" C-m
    done

    tmux attach-session -t "$SESSION_NAME"
}

# Function to create a monitoring dashboard
create_dashboard() {
    SESSION_NAME="dashboard"
    tmux new-session -d -s "$SESSION_NAME" -n "system"

    # Top pane: System overview
    tmux send-keys -t "$SESSION_NAME" "htop" C-m

    # Bottom left: Network
    tmux split-window -v -t "$SESSION_NAME"
    tmux send-keys -t "$SESSION_NAME" "watch -n 1 'ss -tuln | head -20'" C-m

    # Bottom right: Disk
    tmux split-window -h -t "$SESSION_NAME"
    tmux send-keys -t "$SESSION_NAME" "watch -n 1 'df -h'" C-m

    tmux select-pane -t 0
    tmux attach-session -t "$SESSION_NAME"
}

# Function to create a recursion demo
create_recursion() {
    SESSION_NAME="recursion"
    tmux new-session -d -s "$SESSION_NAME"

    # Fractal-like pane splitting
    tmux split-window -h -t "$SESSION_NAME"
    tmux split-window -v -t "$SESSION_NAME.0"
    tmux split-window -v -t "$SESSION_NAME.1"
    tmux split-window -h -t "$SESSION_NAME.2"
    tmux split-window -h -t "$SESSION_NAME.3"

    # Each pane shows its own info
    for i in {0..5}; do
        tmux send-keys -t "$SESSION_NAME.$i" "echo 'Pane $i: Recursion Level $((i+1))' && bash" C-m
    done

    tmux attach-session -t "$SESSION_NAME"
}

# Display menu
show_menu() {
    echo "╔════════════════════════════════════════╗"
    echo "║   PRIMORDIAL TMUX UNLEASHED           ║"
    echo "╚════════════════════════════════════════╝"
    echo ""
    echo "Choose your environment:"
    echo ""
    echo "  1) Development Environment (5 windows)"
    echo "  2) Matrix Display (grid layout)"
    echo "  3) System Dashboard (monitoring)"
    echo "  4) Recursion Demo (fractal panes)"
    echo "  5) Kill all sessions"
    echo "  6) List active sessions"
    echo ""
    read -p "Enter choice [1-6]: " choice

    case $choice in
        1) create_dev_env ;;
        2) create_matrix ;;
        3) create_dashboard ;;
        4) create_recursion ;;
        5) tmux kill-server ;;
        6) tmux ls ;;
        *) echo "Invalid choice" ;;
    esac
}

# Main execution
if [ $# -eq 0 ]; then
    show_menu
else
    case "$1" in
        dev) create_dev_env ;;
        matrix) create_matrix ;;
        dash) create_dashboard ;;
        recursion) create_recursion ;;
        kill) tmux kill-server ;;
        list) tmux ls ;;
        *) show_menu ;;
    esac
fi
