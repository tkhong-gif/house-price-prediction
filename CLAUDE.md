# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## AI usage

The team used Claude (Anthropic) as a coding assistant. Claude drafted the analysis stage, the pipeline runner, the county data download and the tests, cleaned up unused files, and helped write documentation. The data pipeline, features and models are the team's own work, and every AI-assisted change was reviewed and tested by the authors.

## Responsibilities

Claude suggests and drafts changes; the authors decide what is kept and are responsible for the results. Claude must never commit, upload or print the county data's owner names and addresses, and pushes to GitHub only when asked. Long runs (training about 2¼ hours) are started by the authors, and results are reported as measured.

## Working in this repository

Run commands from this folder with `uv run python -m ...`, and run `uv run pytest` after changes. The project must end up running serverless on AWS (S3, Athena, SageMaker) for grading; an API or Dockerfile is not needed. Features built from sale prices must be computed from training houses only, and models must be retrained whenever the feature table changes.
