#!/bin/bash
set -eux

USAGE="Usage: $0 <package_name> <analysis_config.json> <analysis_output.json> [options]"

if [[ $# -lt 3 ]]; then
	echo "$USAGE"
	exit 1
fi

SRCDIR=$(dirname $0)/../src
export PYTHONPATH="${SRCDIR}:${PYTHONPATH:-}"

PACKAGE=$1
shift
python "${SRCDIR}/${PACKAGE}/run.py" $*
