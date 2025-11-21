#!/bin/bash
docker build \
  -t neurosim:latest \
  -f docker/Dockerfile \
  .
