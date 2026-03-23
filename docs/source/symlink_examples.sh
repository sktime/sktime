#!/bin/bash

set -euo pipefail

# helper script to create symbolic links to notebooks in the examples folder

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXAMPLES_DIR="${SCRIPT_DIR}/examples"
SOURCE_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)/examples"

# guard: ensure source exists and is a directory
if [[ ! -d "${SOURCE_DIR}" ]]; then
	echo "Source examples directory not found: ${SOURCE_DIR}" >&2
	exit 1
fi

# if docs/source/examples is a symlink, remove the symlink only (do not touch target)
if [[ -L "${EXAMPLES_DIR}" ]]; then
	rm -f "${EXAMPLES_DIR}"
fi

# ensure destination directory exists and is a real directory
mkdir -p "${EXAMPLES_DIR}"

# clear existing contents inside docs/source/examples without following symlinks elsewhere
rm -rf "${EXAMPLES_DIR}"/*

# create symlinks from top-level examples into docs/source/examples
for src in "${SOURCE_DIR}"/*; do
	name="$(basename "${src}")"
	ln -s "${src}" "${EXAMPLES_DIR}/${name}"
done
