import marimo

__generated_with = "0.14.10"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import os 
    from pathlib import Path

    base_folder = Path('/Users/thomasbush/Documents/ML/alphafold-decoded/tutorials')
    attention = base_folder / 'attention'
    control_folder = attention / 'control_values'
    assert os.path.isdir(control_folder), 'Folder "control values" not found'
    return


@app.cell
def _():
    import math
    import torch
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
