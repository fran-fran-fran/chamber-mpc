#!/bin/bash
# chamber-mpc install script
#
# Licensed under the GNU General Public License v3.0 (GPL-3.0)
# SPDX-License-Identifier: GPL-3.0-or-later

set -euo pipefail
export LC_ALL=C

PLUGIN_NAME="chamber_mpc"
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SOURCE_PATH="${REPO_DIR}/src/${PLUGIN_NAME}"

DEFAULT_KLIPPER_DIR="${HOME}/klipper"
DEFAULT_KLIPPY_ENV="${HOME}/klippy-env"
DEFAULT_PRINTER_DATA="${HOME}/printer_data"

KLIPPER_DIR="${DEFAULT_KLIPPER_DIR}"
KLIPPY_ENV="${DEFAULT_KLIPPY_ENV}"
PRINTER_DATA="${DEFAULT_PRINTER_DATA}"

UNINSTALL=false

# -- Helpers --

function msg_info  { printf "[INFO]    %s\n" "$1"; }
function msg_ok    { printf "[OK]      %s\n" "$1"; }
function msg_warn  { printf "[WARN]    %s\n" "$1"; }
function msg_error { printf "[ERROR]   %s\n" "$1"; }

function display_help {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -k, --klipper <dir>       Klipper directory (default: ${DEFAULT_KLIPPER_DIR})"
    echo "  -e, --klippy-env <dir>    Klippy venv directory (default: ${DEFAULT_KLIPPY_ENV})"
    echo "  -d, --printer-data <dir>  Printer data directory (default: ${DEFAULT_PRINTER_DATA})"
    echo "  --uninstall               Remove the plugin"
    echo "  --help                    Show this help message"
    exit 0
}

# -- Argument parsing --

function parse_args {
    while [[ "$#" -gt 0 ]]; do
        case "$1" in
            -k|--klipper)       KLIPPER_DIR="$2"; shift 2 ;;
            -e|--klippy-env)    KLIPPY_ENV="$2";  shift 2 ;;
            -d|--printer-data)  PRINTER_DATA="$2"; shift 2 ;;
            --uninstall)        UNINSTALL=true;    shift   ;;
            --help)             display_help ;;
            *) msg_error "Unknown option: $1"; display_help ;;
        esac
    done
}

# -- Pre-flight checks --

function preflight_checks {
    if [ "${EUID}" -eq 0 ]; then
        msg_error "This script must not be run as root!"
        exit 1
    fi

    if [ ! -d "${KLIPPER_DIR}/klippy/extras" ]; then
        msg_error "Klipper not found at '${KLIPPER_DIR}'. Use -k to specify the path."
        exit 1
    fi

    if [ ! -d "${KLIPPY_ENV}" ]; then
        msg_error "Klippy venv not found at '${KLIPPY_ENV}'. Use -e to specify the path."
        exit 1
    fi

    if [ ! -d "${SOURCE_PATH}" ]; then
        msg_error "Plugin source not found at '${SOURCE_PATH}'"
        exit 1
    fi

    # Detect service name
    if systemctl list-unit-files --quiet klipper.service; then
        SERVICE_NAME="klipper"
    else
        msg_error "Klipper service not found. Please install Klipper/Kalico first."
        exit 1
    fi

    msg_ok "Found klipper at ${KLIPPER_DIR}"
}

# -- Service control --

function stop_service  { msg_info "Stopping ${SERVICE_NAME}...";  sudo systemctl stop  "${SERVICE_NAME}"; }
function start_service { msg_info "Starting ${SERVICE_NAME}...";  sudo systemctl start "${SERVICE_NAME}"; }

# -- Install --

function link_module {
    local extras_dir="${KLIPPER_DIR}/klippy/extras"
    local exclude_file="${KLIPPER_DIR}/.git/info/exclude"

    # Link: chamber_mpc/ package directory
    local pkg_symlink="${extras_dir}/${PLUGIN_NAME}"
    if [ -L "${pkg_symlink}" ]; then
        rm "${pkg_symlink}"
    elif [ -e "${pkg_symlink}" ]; then
        msg_error "${pkg_symlink} exists but is not a symlink. Remove it manually."
        exit 1
    fi
    ln -frsn "${SOURCE_PATH}" "${pkg_symlink}"
    msg_ok "Linked: ${pkg_symlink} -> ${SOURCE_PATH}"

    # Prevent Klipper marking its own repo as dirty
    if [ -f "${exclude_file}" ]; then
        local rel_path="klippy/extras/${PLUGIN_NAME}"
        if ! grep -qF "${rel_path}" "${exclude_file}"; then
            echo "${rel_path}" >> "${exclude_file}"
            msg_ok "Added '${rel_path}' to git exclude"
        fi
    fi
}

function add_moonraker_updater {
    local moonraker_conf="${PRINTER_DATA}/config/moonraker.conf"

    if [ ! -f "${moonraker_conf}" ]; then
        msg_warn "moonraker.conf not found at ${moonraker_conf}, skipping update manager."
        return
    fi

    if grep -q "\[update_manager chamber-mpc\]" "${moonraker_conf}"; then
        msg_info "Moonraker update manager already configured, skipping."
        return
    fi

    msg_info "Adding update manager to moonraker.conf..."
    cat <<EOF >> "${moonraker_conf}"

## chamber-mpc automatic update management
[update_manager chamber-mpc]
type: git_repo
path: ${REPO_DIR}
origin: https://github.com/YOUR_USERNAME/chamber-mpc.git
managed_services: ${SERVICE_NAME}
primary_branch: main
install_script: scripts/install.sh
EOF
    msg_ok "Added [update_manager chamber-mpc] to moonraker.conf"
    msg_warn "Remember to update the 'origin' URL in moonraker.conf with your actual repo URL."
}

# -- Uninstall --

function unlink_module {
    local extras_dir="${KLIPPER_DIR}/klippy/extras"
    local pkg_symlink="${extras_dir}/${PLUGIN_NAME}"

    if [ -L "${pkg_symlink}" ]; then
        rm "${pkg_symlink}"
        msg_ok "Removed symlink: ${pkg_symlink}"
    else
        msg_info "No symlink found at ${pkg_symlink}, skipping."
    fi

    # Clean up git exclude entry
    local exclude_file="${KLIPPER_DIR}/.git/info/exclude"
    local rel_path="klippy/extras/${PLUGIN_NAME}"
    if [ -f "${exclude_file}" ]; then
        sed -i "\|^${rel_path}\$|d" "${exclude_file}" 2>/dev/null || true
        msg_ok "Removed git exclude entry"
    fi
}

# -- Main --

printf "\n=============================================\n"
printf " chamber-mpc install script\n"
printf "=============================================\n\n"

parse_args "$@"
preflight_checks
stop_service

if [ "${UNINSTALL}" = true ]; then
    msg_info "Uninstalling chamber-mpc..."
    unlink_module
    start_service
    sudo systemctl restart moonraker
    printf "\n"
    msg_ok "Uninstall complete."
    echo ""
    echo "Remember to remove from your config files:"
    echo "  - [chamber_mpc] section from printer.cfg"
    echo "  - [update_manager chamber-mpc] from moonraker.conf"
else
    msg_info "Installing chamber-mpc..."
    link_module
    add_moonraker_updater
    start_service
    sudo systemctl restart moonraker

    printf "\n"
    msg_ok "Installation complete."
    echo ""
    echo "Next steps:"
    echo "  1. Add to your printer.cfg:"
    echo ""
    echo "       [chamber_mpc]"
    echo "       heater: airfryer"
    echo "       heater_power: 1800"
    echo ""
    echo "  2. Restart klipper:"
    echo "       sudo systemctl restart klipper"
    echo ""
    echo "  3. Calibrate:"
    echo "       MPC_CHAMBER_CALIBRATE HEATER=airfryer POINTS=85"
    echo "       (or POINTS=60,100,150,200 for multi-point)"
fi
