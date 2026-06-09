#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
PYTHON="${PYTHON:-$PROJECT_ROOT/venv/bin/python}"
APP="${APP:-$SCRIPT_DIR/Kalman_Proje.py}"

SWAP=0
PRINT_URIS=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --swap)
            SWAP=1
            shift
            ;;
        --invert)
            export KALMAN_PHASE_SIGN=-1
            shift
            ;;
        --print-uris)
            PRINT_URIS=1
            shift
            ;;
        --)
            shift
            break
            ;;
        *)
            break
            ;;
    esac
done

# These are the two Pluto serials currently attached on this machine.
# Override with KALMAN_TX_SERIAL / KALMAN_RX_SERIAL if the devices are replaced.
TX_SERIAL="${KALMAN_TX_SERIAL:-104473e6a60f000121000f00252c17d291}"
RX_SERIAL="${KALMAN_RX_SERIAL:-10447318ac0f0003ffff3500f16d178e29}"

if [[ "$SWAP" == "1" || "${KALMAN_SWAP:-0}" == "1" ]]; then
    tmp="$TX_SERIAL"
    TX_SERIAL="$RX_SERIAL"
    RX_SERIAL="$tmp"
fi

find_usb_uri_by_serial() {
    local serial="$1"
    local line

    while IFS= read -r line; do
        if [[ "$line" == *"serial=$serial"* && "$line" =~ \[(usb:[^]]+)\] ]]; then
            printf '%s\n' "${BASH_REMATCH[1]}"
            return 0
        fi
    done < <(iio_info -s)

    return 1
}

TX_URI="${KALMAN_TX_URI:-$(find_usb_uri_by_serial "$TX_SERIAL" || true)}"
RX_URI="${KALMAN_RX_URI:-$(find_usb_uri_by_serial "$RX_SERIAL" || true)}"

if [[ -z "$TX_URI" || -z "$RX_URI" ]]; then
    printf 'Could not resolve both Pluto USB URIs. Current iio contexts:\n' >&2
    iio_info -s >&2
    exit 1
fi

export KALMAN_TX_URI="$TX_URI"
export KALMAN_RX_URI="$RX_URI"
export KALMAN_STATE_SUFFIX="${KALMAN_STATE_SUFFIX:-rx_${RX_SERIAL}_tx_${TX_SERIAL}}"

printf 'TX URI: %s\nRX URI: %s\n' "$KALMAN_TX_URI" "$KALMAN_RX_URI"
printf 'TX serial: %s\nRX serial: %s\n' "$TX_SERIAL" "$RX_SERIAL"
printf 'phase sign: %s\nstate suffix: %s\n' "${KALMAN_PHASE_SIGN:-1}" "$KALMAN_STATE_SUFFIX"

if [[ "$PRINT_URIS" == "1" ]]; then
    exit 0
fi

exec "$PYTHON" "$APP" "$@"
