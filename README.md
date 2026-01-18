VQA From Scratch
===================

Grabbing MSCoCo (VAE training, VQA visual component)
----------------------------------------------------
```bash
> mkdir Images && cd Images
> curl -OJL images.cocodataset.org/zips/train2014.zip
> curl -OJL images.cocodataset.org/zips/val2014.zip
> curl -OJL images.cocodataset.org/zips/test2015.zip
> unzip train2014.zip
> unzip val2014.zip
> unzip test2015.zip
> rm *.zip

(repeat similar process with Annotations)
```

Training
--------
```bash
(one time)
> pyenv virtualenv 3.10.14 vqa
> pyenv activate vqa
> python3 -m pip install requirements.txt

(each working session)
> pyenv activate vqa

> ./run.sh <run_id>
> ./run.sh <run_id> (<checkpoint step to begin from>)
```

Running loss logging web app
----------------------------
To visualize the training process a bit better, codex wrote a nice little
bun web app for us to track experiments in both real time and reload old ones.

```bash
> cd tracker/ && bun run trackerapp.ts -f ../logs/<run_id> -p 3000
```
and navigate to `localhost:3000` in your browser. Multiple instances can be run for tab-by-tab comparisons.

Create mp4 of step_nnn.png's 
----------------------------
This was cooler when my goal was focused on pretty reconstructions.

```bash
> cd logs/<run_id>/
> ls step_*.png | sort -V | sed "s/^/file '/; s/$/'/" > frames.txt && \
ffmpeg -y -r 30 -f concat -safe 0 -i frames.txt \
  -c:v libx264 -pix_fmt yuv420p -crf 18 out.mp4
```
