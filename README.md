# DP-Project-NeLo-Mapper

This reposirety integrates [**NeLo (Neural Laplacian operators)**](https://arxiv.org/abs/2409.06506)  with [**Mapper**](https://research.math.osu.edu/tgda/mapperPBG.pdf) for 3D point-cloud segmentation. Its built on top of the original Nelo codebase by Pang *et al.* (https://github.com/IntelligentGeometry/NeLo).

## Requirements

As previously mentioned this is built on top of the original NeLo codebase. To install the required packages and libraries follow the instructions on https://github.com/IntelligentGeometry/NeLo. <br/>
Additional requirements are the python libraries: ebreex, matplotlib, mpl_toolkits, kmapper, scipy and sklearn.

## Extraction the Learned Laplacian

Once you run the NeLo's data preparation process you will find your processed objects in `data/processed_data` folder. Next place the `.obj` file of your object/s in the `data/plane/test_meshes` folder and run:

```bash
bash script/test_on_plane.sh
```

This will save the learned Laplacian matrix in the `out/predicted_L` folder.

## Segmentation and Evaluation

To produce the segmentations and evaluate them run: 

```bash
python b_segmentation.py "path_to_your_object" "path_to_the_objects_laplacian_matrix"
```

example:

```bash
python b_segmentation.py data/plane/test_meshes/plane.obj out/predicted_L/1db7bca33ba446aba5cac89017eae8d1.npz
```

Additionaly if you want to visualize your original object run: 

```bash
python b_visualize_obj.py "path_to_your_oject"
```

example:

```bash
python b_visualize_obj.py data/plane/test_meshes/plane.obj
```

For additional information check out the supplemental video: `Supplemental_video.mp4` and the projects report: `Project_Report.pdf`.


If you have any questions, please feel free to contact me at bb53717@student.uni-lj.si
