# Setup

These instructions use `uv` to create and manage the project environment on Windows.

## 1. Install `uv`

Install `uv` if it is not already available on your machine. Then open a terminal in the repository root.

## 2. Create the project environment

Create the local environment in `.venv`:

```powershell
uv venv --python 3.12
```

## 3. Install dependencies

Install the packages listed in `requirements.txt`:

```powershell
uv pip install -r requirements.txt
```

If kernel registration fails, install `ipykernel` explicitly:

```powershell
uv pip install ipykernel
```

## 4. Register the Jupyter kernel

Register the environment so VS Code and Jupyter can use it:

```powershell
.venv\Scripts\python.exe -m ipykernel install --user --name barren-plateaus --display-name "Barren Plateaus (.venv)"
```

## 5. Select the kernel in VS Code

Open [assets/code/QNSPSA-vs-COBYLA.ipynb](assets/code/QNSPSA-vs-COBYLA.ipynb) and choose the kernel named `Barren Plateaus (.venv)` from the kernel picker.