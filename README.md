# IEM Explorer – Electromigration Analysis Tool

## Overview

IEM Explorer is a Python-based graphical tool developed to analyze electromigration experiments in microstructured metallic devices.

The software processes pulse-based resistance data (Rmin) and automatically detects the electromigration onset current (I_EM) using a combined derivative and plateau-based criterion.

## Key Features

* Graphical user interface (Tkinter-based)
* Multi-file CSV import
* Automatic detection of:

  * Electromigration onset current (I_EM)
  * Plateau regions in resistance evolution
* Data smoothing using moving average
* Derivative-based analysis (dR/dI)
* Reverse-search algorithm for robust onset detection
* Interactive plots:

  * Raw and smoothed Rmin
  * dR/dI vs current
  * Reverse-search visualization
  * Candidate window inspection
* Export of summary results to CSV

## Physical Method

The detection of I_EM is based on:

1. Smoothing of Rmin(I)
2. Calculation of dR/dI
3. Reverse traversal of the dataset (high → low current)
4. Identification of the first window satisfying:

   * Low derivative magnitude (|dR/dI| < threshold)
   * Low resistance variation (ΔR < k·σ)

This ensures robust identification of plateau-like behavior associated with electromigration onset.

## Requirements

* Python 3.x
* numpy
* pandas
* matplotlib

Install dependencies using:

pip install -r requirements.txt

## Usage

Run the application:

python iem_explorer.py

Then:

1. Load CSV files
2. Select current and Rmin columns
3. Set analysis parameters
4. Click “Analyze”

## Example Data

A sample dataset is provided in:

example_data/sample_em_data.csv

You can use this file to test the tool and reproduce the analysis workflow.

## Example Output

The tool generates multiple plots including:

* Rmin vs current
* dR/dI vs current
* Reverse search visualization
* Candidate plateau windows

## Author

Elijah – Experimental physicist working on electromigration and thin-film systems.
