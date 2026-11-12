# MultiSG

## Object fingerprinting (single RGB image)

`src/fingerprint/fingerprint_image.py` runs DovSG's 2D perception pipeline on one image: RAM tags, Grounding DINO boxes, NMS, SAM2 masks, and CLIP ViT-H-14 features. DovSG is a git submodule in `third_party/DovSG` and is not modified.

For each detected object it saves a label, box, confidence, mask, a 1024-d CLIP image feature (masked crop and plain crop weighted 0.75/0.25) and a CLIP text feature of the label.

### Requirements

- Linux with an NVIDIA GPU. All four models load onto one GPU and their weights alone are about 7 to 8 GB, so use a card with clearly more than 8 GB (16 GB or more is safe). `--device cpu` works but is slow and needs about 10 GB of RAM.
- Python 3.10 or 3.11 (not 3.12: `supervision==0.14.0` does not support it).
- About 18 GB of disk: 11.5 GB of checkpoints plus the Python environment.

### 1. Get the code

Only the three submodules the perception path needs are fetched, at the commits DovSG pins.

```bash
git clone <this repo> multi_sg
cd multi_sg
git submodule update --init
git -C third_party/DovSG submodule update --init third_party/segment-anything-2 third_party/GroundingDINO third_party/recognize-anything
```

### 2. Environment

If a CUDA toolkit is installed, set `CUDA_HOME` first (for example `export CUDA_HOME=/usr/local/cuda`) so GroundingDINO compiles its CUDA op.

```bash
python3 -m venv .venv-dovsg
source .venv-dovsg/bin/activate
python3 -m pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
cd third_party/DovSG/third_party
python3 -m pip install -e segment-anything-2
python3 -m pip install -e GroundingDINO
python3 -m pip install -r recognize-anything/requirements.txt
python3 -m pip install -e recognize-anything
python3 -m pip install numpy==1.26.4 supervision==0.14.0 open_clip_torch==2.31.0 transformers==4.50.3 open3d scipy matplotlib opencv-python tqdm
cd ../../..
```

The last install line must run last. pip will warn that GroundingDINO wants `supervision>=0.22`; that is expected, DovSG's code uses the 0.14 API.

### 3. Checkpoints

DovSG reads these from fixed paths under `third_party/DovSG/checkpoints`.

```bash
cd third_party/DovSG
mkdir -p checkpoints/GroundingDINO checkpoints/segment-anything-2
huggingface-cli download google-bert/bert-base-uncased config.json model.safetensors tokenizer_config.json tokenizer.json vocab.txt --local-dir checkpoints/bert-base-uncased
huggingface-cli download laion/CLIP-ViT-H-14-laion2B-s32B-b79K open_clip_pytorch_model.bin --local-dir checkpoints/CLIP-ViT-H-14-laion2B-s32B-b79K
huggingface-cli download xinyu1205/Recognize_Anything-Tag2Text ram_swin_large_14m.pth --repo-type space --local-dir checkpoints/recognize_anything
wget -P checkpoints/GroundingDINO https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
cp third_party/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py checkpoints/GroundingDINO/
wget -P checkpoints/segment-anything-2 https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt
cd ../..
```

If `ram_swin_large_14m.pth` is no longer in that Space, download it from the `xinyu1205/recognize_anything_model` model repo instead.

### 4. Run

```bash
source .venv-dovsg/bin/activate
python3 -c "import torch, sam2, groundingdino, ram, open_clip, supervision, open3d; print(torch.__version__, torch.cuda.is_available(), supervision.__version__)"
python3 src/fingerprint/fingerprint_image.py path/to/image.jpg
```

The first command should print `2.3.1+cu121 True 0.14.0`. The script can be run from any directory and writes two files to `fingerprints/` (change with `--out`):

- `image.jpg`: the input with boxes, masks and labels drawn on it.
- `image.pkl`: DovSG's `det_res` dict with keys `xyxy`, `confidence`, `class_id`, `mask`, `classes`, `image_feats`, `text_feats`. Row `i` of each array is object `i`; its label is `classes[class_id[i]]`.

```bash
python3 -c "import pickle; d = pickle.load(open('fingerprints/image.pkl', 'rb')); print([d['classes'][i] for i in d['class_id']], d['image_feats'].shape)"
```

Options: `--device cpu`, and `--add_classes "red mug" "stapler"` to add labels that Grounding DINO should always look for (DovSG's own lab-specific list is cleared by default).

## Acknowledgement

The perception pipeline is [DovSG](https://github.com/BJHYZJ/DovSG), used unmodified:

```
@ARTICLE{10909193,
  author={Yan, Zhijie and Li, Shufei and Wang, Zuoxu and Wu, Lixiu and Wang, Han and Zhu, Jun and Chen, Lijiang and Liu, Jihong},
  journal={IEEE Robotics and Automation Letters},
  title={Dynamic Open-Vocabulary 3D Scene Graphs for Long-Term Language-Guided Mobile Manipulation},
  year={2025},
  volume={10},
  number={5},
  pages={4252-4259},
  doi={10.1109/LRA.2025.3547643}}
```
