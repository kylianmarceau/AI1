CSC2042S 2025 Assignment 1 — Unsupervised Learning

- Summary: Implements the assignment pipeline in a single Jupyter notebook, including data preprocessing, custom K-means (with random and k-means++ initializations), convergence experiments, K selection, PCA analysis, and visualizations. All plots now render inline under their producing cells, while also saving copies to `figures/`.

Setup
- Python: 3.9+ (tested with 3.10/3.11)
- Install dependencies:
  - `pip install numpy pandas matplotlib scikit-learn scipy seaborn`
- Data: Place the provided WDI data in `CSC2042S-Assignment1-Data/WDICSV.csv`.
  - Do not submit any data files as part of your tarball.

How To Run
- Open `ASSignment.ipynb` in Jupyter (VS Code, JupyterLab, or `jupyter notebook`).
- Ensure the kernel has the packages listed above.
- Run all cells in order. The notebook reads from `CSC2042S-Assignment1-Data/WDICSV.csv`.
- Plots: Every plotting cell calls `plt.show()` so figures appear inline; copies are still saved under `figures/`.

Files
- `ASSignment.ipynb`: Main analysis notebook (preprocessing, K-means implementation and experiments, PCA, visualizations).
- `figures/`: Generated images saved by the notebook (plots are also shown inline).
- `CSC2042S-Assignment1-Data/`: Expected location of dataset (not included in submission).
- `assignment_text.txt`: Text copy of the official assignment brief (for reference only).
- `tools/clean_notebook.py`: Helper used to strip code comments and ensure inline plot rendering.

Notes and Assumptions
- K-means is implemented in pure NumPy per the brief; scikit-learn is used only for tasks allowed by the assignment (e.g., t-SNE/PCA).
- Random seeds are set where applicable to assist reproducibility; results may still vary slightly across environments.
- If you re-run the notebook from a clean kernel, all graphs will be visible directly under the producing cells and also be written to `figures/`.

Submission Guidance (from brief)
- Submit a single compressed archive named as your student number, e.g. `GWRBRA001.tar.xz`.
- Include: `ASSignment.ipynb`, `README.md`, and any supporting code you wrote (e.g., under `tools/`) and optionally `figures/` if you want to include static outputs.
- Do not include any data files.
- Also submit a separate PDF report (<= 6 pages) discussing design decisions and results.

Creating the Tarball (example)
- From the assignment folder (excluding data):
  - `tar -cJf GWRBRA001.tar.xz ASSignment.ipynb README.md figures tools assignment_text.txt`
- Verify the archive by extracting it and opening the notebook before submitting.

