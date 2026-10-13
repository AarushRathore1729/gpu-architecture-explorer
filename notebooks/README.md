# Colab runner

`GPU_Architecture_Explorer_Colab.ipynb` embeds this source tree. Upload it directly in Colab, select A100, and run all cells. No GitHub clone or source archive upload is needed.

Rebuild it after source edits with:

```bash
python scripts/build_colab.py
```

Generated notebooks do not recursively embed themselves. The notebook exports a full repository and measured result bundle after its GPU run.
