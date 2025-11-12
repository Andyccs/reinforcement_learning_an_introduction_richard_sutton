# Makefile for LaTeX project

# Compiler
LATEXMK = latexmk -pdf

# Main file
MAIN = main.tex

# Build directory
BUILD = build

# Default target
all:
	mkdir -p $(BUILD)
	cd srcs && $(LATEXMK) -outdir=../$(BUILD) $(MAIN)

# Clean up auxiliary files
clean:
	rm -rf $(BUILD)

.PHONY: all clean
