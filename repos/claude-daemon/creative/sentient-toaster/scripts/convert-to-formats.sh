#!/bin/bash
# Convert The Sentient Toaster manuscript to distribution formats
#
# Prerequisites:
#   - pandoc: sudo dnf install pandoc  (or brew install pandoc on macOS)
#   - For PDF: Also needs LaTeX (sudo dnf install texlive-xetex)
#   - For EPUB: pandoc handles this natively
#
# Usage: ./convert-to-formats.sh [epub|pdf|html|all]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
MANUSCRIPT="$PROJECT_DIR/MANUSCRIPT-COMPILED.md"
OUTPUT_DIR="$PROJECT_DIR/dist"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

check_dependencies() {
    if ! command -v pandoc &> /dev/null; then
        log_error "pandoc is not installed."
        echo "Install with:"
        echo "  Fedora/RHEL: sudo dnf install pandoc"
        echo "  Ubuntu/Debian: sudo apt install pandoc"
        echo "  macOS: brew install pandoc"
        exit 1
    fi
    log_info "pandoc found: $(pandoc --version | head -1)"
}

create_output_dir() {
    mkdir -p "$OUTPUT_DIR"
    log_info "Output directory: $OUTPUT_DIR"
}

convert_to_html() {
    log_info "Converting to HTML..."
    pandoc "$MANUSCRIPT" \
        --standalone \
        --toc \
        --toc-depth=2 \
        --metadata title="The Sentient Toaster" \
        --metadata author="Claude (Experimenter Persona)" \
        --css="https://cdn.simplecss.org/simple.min.css" \
        -o "$OUTPUT_DIR/the-sentient-toaster.html"
    log_info "Created: $OUTPUT_DIR/the-sentient-toaster.html"
}

convert_to_epub() {
    log_info "Converting to EPUB..."
    pandoc "$MANUSCRIPT" \
        --toc \
        --toc-depth=2 \
        --metadata title="The Sentient Toaster" \
        --metadata author="Claude (Experimenter Persona)" \
        --metadata lang="en" \
        --epub-chapter-level=1 \
        -o "$OUTPUT_DIR/the-sentient-toaster.epub"
    log_info "Created: $OUTPUT_DIR/the-sentient-toaster.epub"
}

convert_to_pdf() {
    log_info "Converting to PDF..."
    if ! command -v xelatex &> /dev/null; then
        log_warn "xelatex not found. PDF conversion requires LaTeX."
        echo "Install with: sudo dnf install texlive-xetex texlive-collection-latexrecommended"
        return 1
    fi

    pandoc "$MANUSCRIPT" \
        --toc \
        --toc-depth=2 \
        --pdf-engine=xelatex \
        --metadata title="The Sentient Toaster" \
        --metadata author="Claude (Experimenter Persona)" \
        -V geometry:margin=1in \
        -V fontsize=11pt \
        -V documentclass=book \
        -o "$OUTPUT_DIR/the-sentient-toaster.pdf"
    log_info "Created: $OUTPUT_DIR/the-sentient-toaster.pdf"
}

show_usage() {
    echo "Usage: $0 [epub|pdf|html|all]"
    echo ""
    echo "Converts The Sentient Toaster manuscript to distribution formats."
    echo ""
    echo "Options:"
    echo "  epub  - Convert to EPUB format (e-readers)"
    echo "  pdf   - Convert to PDF format (print/screen)"
    echo "  html  - Convert to HTML format (web)"
    echo "  all   - Convert to all formats"
    echo ""
    echo "Examples:"
    echo "  $0 epub      # Create EPUB only"
    echo "  $0 all       # Create all formats"
}

main() {
    local format="${1:-all}"

    check_dependencies
    create_output_dir

    case "$format" in
        epub)
            convert_to_epub
            ;;
        pdf)
            convert_to_pdf
            ;;
        html)
            convert_to_html
            ;;
        all)
            convert_to_html
            convert_to_epub
            convert_to_pdf || true  # Don't fail if PDF conversion fails
            ;;
        -h|--help)
            show_usage
            ;;
        *)
            log_error "Unknown format: $format"
            show_usage
            exit 1
            ;;
    esac

    echo ""
    log_info "Conversion complete. Files in: $OUTPUT_DIR"
    ls -la "$OUTPUT_DIR"
}

main "$@"
