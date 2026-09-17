#!/usr/bin/env bash
# Run okvis on a wildfire sequence.
#
#   OKVIS_BUILD     directory containing the okvis_app_synchronous binary (default: ./build)
#   OKVIS_DATASETS  root holding the dataset folders                      (default: ~/Datasets)
#   OKVIS_OUTPUT    where results are written                             (default: ./output)
#   SEQUENCE        sequence subfolder under $OKVIS_DATASETS/forestfire
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OKVIS_BUILD="${OKVIS_BUILD:-$REPO_ROOT/build}"
OKVIS_DATASETS="${OKVIS_DATASETS:-$HOME/Datasets}"
OKVIS_OUTPUT="${OKVIS_OUTPUT:-$REPO_ROOT/output}"
SEQUENCE="${SEQUENCE:-2025-06-17_02-43-11_0/0617_024311}"

mkdir -p "$OKVIS_OUTPUT/wildfire"

GLOG_logtostderr=1 "$OKVIS_BUILD/okvis_app_synchronous" \
  "$REPO_ROOT/config/wildfire/okvis2.yaml" \
  "$OKVIS_DATASETS/forestfire/$SEQUENCE" \
  "$OKVIS_OUTPUT/wildfire"
