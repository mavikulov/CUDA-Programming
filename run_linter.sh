#!/bin/bash

# Run from build directory

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 task-name"
    exit 1
fi

TASK_PATH=../$1

if [ ! -f compile_commands.json ]; then
    echo "Run this script from the build directory"
    exit 1
fi

../run-clang-format.py --clang-format-executable clang-format-20 -r $TASK_PATH
