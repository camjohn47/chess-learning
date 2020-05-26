#!/usr/bin/env bash
set -eu

black .
isort --recursive .
