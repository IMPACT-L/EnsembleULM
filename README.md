
# Ensemble Learning for ULM

> **Adaptive fusion of multiple models for accurate microbubble localization in super‑resolution ultrasound imaging**

![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg) ![Python](https://img.shields.io/badge/Python-3.9%20%7C%203.10-blue)  
[![DOI](https://zenodo.org/badge/DOI/10.1109/ISBI60581.2025.10980786.svg)](https://doi.org/10.1109/ISBI60581.2025.10980786)

----------

## ✨ Key Features

-   **Model‑aware fusion:** Optional adaptive IoU thresholds and per‑model weights improve agreement across all detectors.
    
-   **Dynamic thresholding:** Optional per‑model score optimisation using ground‑truth to maximise AUC.
    
-   **Plug‑and‑play post‑processors:** Classic NMS, Soft‑NMS, Non‑Maximum Weighted, and Weighted Boxes Fusion—all optionally density‑adaptive.
    
-   **Real‑world co‑ordinates:** Automatic conversion from normalised box space to millimetres using simulation metadata.
    
-   **Reproducible workflow:** Single YAML config controls data paths, hyper‑parameters, and toggles for simulation/in‑vivo modes.
    

----------

## 🏗️ Project Structure

```
.
├── src/                    # Core Python code (this repo)
│   ├── utils/              # Helper modules (thresholding, JSON → boxes, …)
│   └── ensemble_ulm.py     # Main script with adaptive fusion pipeline
├── data/
│   ├── Simulation Predictions/        # JSON outputs of individual models
|	├── Invivo Predictions/        # JSON outputs of individual models
|	├── outputs/        # JSON outputs of individual models
│   └── metadata/           # .mat or .h5 files with simulation grid info
├── config_invivo.yaml      # Central configuration file for invivo data
├── config_simulation.yaml      # Central configuration file for simulation data
├── requirements.txt
└── LICENSE                 # MIT licence text

```

----------

## 🚀 Quick Start

### 1. Clone & install

```bash
 git clone https://github.com/<your‑user>/ensemble‑ulm.git
 cd ensemble‑ulm
 python -m venv .venv && source .venv/bin/activate
 pip install -r requirements.txt

```

### 2. Prepare inputs

-   Place per‑frame **prediction JSONs** from each model (Deformable DETR in our case) in `data/Simulation predictions/`.
    
-   Put the **simulation metadata** file (`metadata.mat` or `.h5`) in `data/metadata/`.
    
-   Adjust paths in `config_invivo.yaml` if needed.
    

### 3. Run the pipeline

```bash
python ensemble_ulm.py --config config_invivo.yaml
or
python ensemble_ulm.py --config config_simulation.yaml

```

Fusion results for every method are written to `data/output/boxes_<method>.txt`.

----------


## 📊 Citation

If you use _any part_ of this code, **please cite both the paper and the repository**:

```bibtex
@inproceedings{Gharamaleki2025ULMEnsemble,
  author    = {S. K. Gharamaleki and B. Helfeld and H. Rivaz},
  title     = {Ensemble Learning for Microbubble Localization in Super‑Resolution Ultrasound},
  booktitle = {Proc. 2025 IEEE 22nd Int. Symp. Biomed. Imaging (ISBI)},
  address   = {Houston, TX, USA},
  pages     = {—},
  month     = apr,
  year      = {2025},
  doi       = {10.1109/ISBI60581.2025.10980786}
}

```

```text
Gharamaleki, S. K. **Ensemble‑ULM**  GitHub repository, 2025.  
URL: https://github.com/sepidehkhakzad/EnsembleULM

```

----------

## 🔖 License & Disclaimer

This project is licensed under the MIT licence (see `LICENSE`).  
The software is provided **“AS IS”** without warranty of any kind. Use at your own risk.

----------

## 🤝 Contributing

Pull‑requests are welcome! Please:

1.  Fork the repo & create a new branch.
    
2.  Run _black_ / _ruff_ before committing.
    
3.  Add tests for new features where practical.
    

----------

## 🙏 Acknowledgements

-   **ensemble‑boxes** by @ZFTurbo: https://github.com/ZFTurbo/Weighted-Boxes-Fusion —core post‑processing algorithms.
- Ultra-SR IEEE IUS 2022 Challenge for providing the dataset. 
----------