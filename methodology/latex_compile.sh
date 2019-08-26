#!/usr/bin/env bash

grep -l '\\documentclass' *tex | xargs latexmk -pdf -pvc -silent