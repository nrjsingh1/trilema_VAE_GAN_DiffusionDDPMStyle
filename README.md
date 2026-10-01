# CSIT 595 Assignment 1

The assignment implementation is in [`codes/assignment1_experiments.ipynb`](codes/assignment1_experiments.ipynb).

## Run

1. Install dependencies: `python -m pip install -r requirements.txt`
2. Open the notebook in Jupyter or Google Colab.
3. Run the smoke-test configuration first. Set `CONFIG["full_run"] = True` for the final Colab Pro run.
4. Save the generated figures and metrics into the report section at the end of the notebook.

The notebook downloads Fashion-MNIST through `torchvision`, converts images to `1 x 32 x 32`, and normalizes them to `[-1, 1]`. It trains an evaluator CNN and unconditional VAE, DCGAN, and DDPM-style models, then computes cFID, predicted-label entropy, KL divergence to the real label distribution, sample speed, and the 5,000-image generation time.

The experiment is intentionally unconditional: labels are used only by the evaluator after generation and never by a generator or denoiser.
