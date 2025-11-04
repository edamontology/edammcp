#!/usr/bin/env bash

# install uv
pip install uv

# install dev dependencies
uv sync --dev
