VQA From Scratch
===================

Grabbing MSCoCo (VAE training, VQA visual component)

> mkdir Images && cd Images
> curl -OJL images.cocodataset.org/zips/train2014.zip
> curl -OJL images.cocodataset.org/zips/val2014.zip
> curl -OJL images.cocodataset.org/zips/test2015.zip
> unzip train2014.zip
> unzip val2014.zip
> unzip test2015.zip
> rm *.zip
