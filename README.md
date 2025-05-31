\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage[hidelinks]{hyperref}
\usepackage{listings}
\usepackage{xcolor}
\usepackage{enumitem}
\usepackage{titling}

\definecolor{codegray}{gray}{0.95}
\lstset{
  backgroundcolor=\color{codegray},
  basicstyle=\ttfamily\small,
  breaklines=true,
  frame=single,
  columns=fullflexible,
  keepspaces=true
}

\pretitle{\begin{center}\LARGE\bfseries}
\posttitle{\end{center}\vspace{1em}}
\title{DP-Project-NeLo-Mapper \\ \large\texttt{README}}
\author{Blaž Bulić}
\date{}

\begin{document}
\maketitle

\section*{Overview}
This repository integrates \textbf{NeLo} (Neural Laplacian Operators) with \textbf{Mapper} for 3D point-cloud segmentation. It is built upon the original NeLo codebase by Pang \emph{et al.} (\href{https://github.com/IntelligentGeometry/NeLo}{IntelligentGeometry/NeLo}). We provide three pipelines:
\begin{enumerate}[nosep]
  \item Spectral + KMeans  
  \item Mapper using NeLo‐learned filters  
  \item Mapper using manually computed Gaussian–Laplacian filters  
\end{enumerate}

\section{Requirements}
\begin{itemize}[nosep]
  \item \textbf{OS:} Ubuntu 22.04  
  \item \textbf{Python:} 3.10  
  \item \textbf{Core packages:}
    \begin{itemize}[nosep]
      \item \texttt{torch\ge2.3}, \texttt{pytorch-lightning}, \texttt{torch-geometric}  
      \item \texttt{numpy}, \texttt{scipy}, \texttt{trimesh}, \texttt{pyembree}  
      \item \texttt{pymeshlab}, \texttt{joblib}, \texttt{kmapper}  
      \item \texttt{scikit-learn}, \texttt{matplotlib}, \texttt{rich}, \texttt{tqdm}
    \end{itemize}
\end{itemize}

\paragraph{Conda environment setup}
\begin{lstlisting}
conda create -n nelo-mapper python=3.10
conda activate nelo-mapper

# PyTorch + CUDA
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 \
    -c pytorch -c nvidia

# PyG
conda install pyg -c pyg

# remaining deps
pip install numpy scipy trimesh pyembree pymeshlab \
    joblib kmapper scikit-learn matplotlib rich tqdm
\end{lstlisting}

\section{Data Preparation}
\begin{enumerate}[nosep]
  \item Place raw meshes in \texttt{data/raw\_data/\<category\>/\*.obj}.  
  \item Run processing script:
  \begin{lstlisting}
cd src/data_prepare
python generate_processed_mesh.py
  \end{lstlisting}
  Processed meshes will appear under \texttt{data/processed\_data/\<category\>/}.
\end{enumerate}

\section{Extracting the Learned Laplacian}
\begin{enumerate}[nosep]
  \item Copy pretrained checkpoint (\texttt{.ckpt}) into \texttt{out/checkpoints/}.  
  \item Run:
  \begin{lstlisting}
python main.py --config config/chair_cache.py
python main.py --config config/chair_test.py
  \end{lstlisting}
  \item Learned Laplacian matrices (\texttt{.npz}) appear in \texttt{out/predicted\_L/}.
\end{enumerate}

\section{Segmentation \& Evaluation}
Use the segmentation script to run all three methods, compute  
\emph{Silhouette}, \emph{Davies–Bouldin}, and \emph{Calinski–Harabasz} scores,  
and visualize 3D results:
\begin{lstlisting}
python blaz_segmentation.py \
  data/chair/test_meshes/<mesh>.obj \
  out/predicted_L/<mesh>.npz
\end{lstlisting}

Output:
\begin{itemize}[nosep]
  \item Console table of metrics  
  \item Bar charts of each metric  
  \item 3‐panel 3D scatter (side‐by‐side)
\end{itemize}

\section{Repository Structure}
\begin{verbatim}
.
├── data
│   ├── raw_data/
│   └── processed_data/
│       └── chair/
│           └── test_meshes/
├── out
│   ├── checkpoints/
│   └── predicted_L/
├── src
│   ├── data_prepare/
│   │   └── generate_processed_mesh.py
│   ├── modules/
│   ├── models/
│   ├── pipeline.py
│   └── my_dataset.py
├── config
│   ├── global_config.py
│   ├── chair_cache.py
│   └── chair_test.py
├── blaz_segmentation.py
├── main.py
├── requirements.txt
└── README.tex
\end{verbatim}

\section{Citation}
If using this work, please cite:
\begin{lstlisting}
@article{pang2024neural,
  title={Neural Laplacian Operator for 3D Point Clouds},
  author={Pang, Bo and Zheng, Zhongtian and Li, Yilong and Wang, Guoping and Wang, Peng-Shuai},
  journal={ACM Trans. on Graphics (SIGGRAPH Asia)},
  year={2024}
}
\end{lstlisting}

\end{document}
