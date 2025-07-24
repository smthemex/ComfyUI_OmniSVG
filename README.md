# ComfyUI_OmniSVG
[OmniSVG](https://github.com/OmniSVG/OmniSVG): A Unified Scalable Vector Graphics Generation Model,you  can try it in ComfyUI.


---

# 1 . Installation /安装

In the ./ComfyUI /custom_node directory, run the following:   
```
git clone https://github.com/smthemex/ComfyUI_OmniSVG.git
```


# 2 . Requirements  
* if windows ,cairo is difficult to install . use [links](https://github.com/tschoonj/GTK-for-Windows-Runtime-Environment-Installer/releases/download/2022-01-04/gtk3-runtime-3.24.31-2022-01-04-ts-win64.exe)   install / 如果cairo安装不上，用连接的exe安装

```
pip install -r requirements.txt
```

# 3. Model
* [OmniSVG/OmniSVG](https://huggingface.co/OmniSVG/OmniSVG/tree/main) download/下载pytorch_model.bin
* [Qwen/Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/tree/main) all files /所有文件
  
  ```
  |   ├──you comfyui/models/diffusion_models/
  |       ├──pytorch_model.bin
  |   ├──any path/Qwen/Qwen2.5-VL-3B-Instruct
  |       ├──all files # 所有文件
  
  ```

# 4. Tips
* 文生矢量和图生矢量，连图则图，无图则文。

# 5 .Example
* ![](https://github.com/smthemex/ComfyUI_OmniSVG/blob/main/example_workflows/example.png)
* ![](https://github.com/smthemex/ComfyUI_OmniSVG/blob/main/example_workflows/example_.png)


# 6. Citation
```
@article{yang2025omnisvg,
  title={OmniSVG: A Unified Scalable Vector Graphics Generation Model}, 
  author={Yiying Yang and Wei Cheng and Sijin Chen and Xianfang Zeng and Jiaxu Zhang and Liao Wang and Gang Yu and Xinjun Ma and Yu-Gang Jiang},
  journal={arXiv preprint arxiv:2504.06263},
  year={2025}
}

```
