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


Probing
-------
Linear probes on mu are used to test downstream task efficiency. Multiple probes may be run in parallel and share the same batch, making them almost 2x as fast when running 3 in parallel, relative to 3 sequential runs.

```bash
> python3 -m evals.probe --ckpt logs/sl_d2_b01/step_10001.tar --use_mu
> python3 -m evals.probe --ckpts logs/model1/step_10001.tar logs/model2/step_10001.tar --use_mu --multi_mode=lockstep
```


Create mp4 of step_nnn.png's 
----------------------------
This was cooler when my goal was focused on pretty reconstructions.

```bash
> cd logs/<run_id>/
> ls step_*.png | sort -V | sed "s/^/file '/; s/$/'/" > frames.txt && \
ffmpeg -y -r 30 -f concat -safe 0 -i frames.txt \
  -c:v libx264 -pix_fmt yuv420p -crf 18 out.mp4
```

Gaussian Visualizaton app
-------------------------
Go to Chrome and use 'file:///' in the search bar to pull up the 
file search functionality, and navigate to <project>/gaus/index.html.

Super handy for getting simple 2d visualizations of how gaussians move
under different pressures (loss functions).
