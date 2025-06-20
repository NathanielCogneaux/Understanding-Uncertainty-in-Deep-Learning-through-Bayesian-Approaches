# Uncertainty Quantification in Deep Learning

This repository serves as a personal workspace to collect and organize thoughts on **Uncertainty Quantification (UQ)** in deep learning. If you're interested in UQ, you may find the resources here useful.

## 📄 PDF Notes

You'll find a PDF titled **"Understanding Uncertainty in Deep Learning through Bayesian Approaches"**, which introduces the basic theoretical foundations of UQ in deep learning. It covers:

* The motivation behind incorporating uncertainty in deep learning
* The mathematical framework of Bayesian approaches
* Key distinctions and categories of uncertainty

## 🧪 Notebooks

Accompanying the PDF are a set of **Jupyter notebooks**, which are fully usable. These notebooks reproduce key images, experiments, and results discussed in the notes. You can run them to better understand the concepts and visualizations presented.

All dependencies can be installed at once using:
   ```bash
   pip install -r requirements.txt
   ```

## 🎤 Presentation Slides & Paper Reviews

In the folder **`related_docs/`**, you’ll find several documents related to practical applications of **Uncertainty Quantification (UQ)** in deep learning:

1. 📄 **Slides** from a presentation on the paper:
   **"Decomposition of Uncertainty in Bayesian Deep Learning for Efficient and Risk-sensitive Learning"**
   by *Stefan Depeweg, José Miguel Hernández-Lobato, Finale Doshi-Velez, and Steffen Udluft*
   → [arXiv:1710.07283](https://arxiv.org/abs/1710.07283)
   This work focuses on the role of UQ in **reinforcement learning**, and how decomposing uncertainty can improve learning efficiency and safety.

2. 📄 **Slides and a review** of the paper:
   **"What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?"**
   by *Alex Kendall and Yarin Gal*
   → [arXiv:1703.04977](https://arxiv.org/abs/1703.04977)
   This paper addresses how **epistemic and aleatoric uncertainties** contribute to performance in **computer vision** tasks, particularly in regression and segmentation problems.

These materials highlight how different forms of uncertainty play critical roles in both **reinforcement learning** and **computer vision**.