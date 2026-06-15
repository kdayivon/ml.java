#!/bin/bash
set -e

SRC_DIR="src/header"
MAIN="src/Main.java"
OUT_DIR="out" 

javac -d "$OUT_DIR" "$SRC_DIR"/*.java "$MAIN" 

java -cp "$OUT_DIR" Main

