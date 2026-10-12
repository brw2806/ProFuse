# Code import

This repository includes source files imported from `EPNet (copy)-thiscpu.rar`.
The original EPNet / PointRCNN documentation and licenses are retained.

Included: Python and C++/CUDA source, configuration files, build scripts,
KITTI split lists, demonstration images, and the original source drafts.

Excluded: the archive's Git history, virtual environments, IDE metadata,
Python caches, compiled extensions, build outputs, KITTI sample data,
training/evaluation logs, model checkpoints, generated NumPy arrays,
and generated PDF figures. These remain in the original archive.

Install dependencies and build extensions following the original README.
Dataset files and checkpoints must be supplied separately. Demonstrations
that load generated NumPy arrays require those arrays from the original
archive or regeneration.

Import verification: all 63 Python source files parsed successfully.
Training, inference, extension compilation, and CPU compatibility have not
been validated. The archive name alone does not establish CPU compatibility;
the imported project retains CUDA extension source and its original
dependency versions.
