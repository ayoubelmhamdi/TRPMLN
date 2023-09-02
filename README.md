# Repository for Research about lung nodules detection using AI.
----------------------------------------------------------------

## Create TRPMLN Dataset for training to detect cancer/normal nodules.

We should have a new empty directory to store tiff images by using something like `mkdir ROI/` in UNIX, and we should have the `LIDC-LDRI` Dataset.
### Installation and initialisation.
```console
$ python3 -m venv venv
$ source venv/bin/activate
$ pip3 install -r requirements.txt
```

### Execution.
```console
$ python3 create_trpmln_dataset.py --dataset DATASET/ --roi ROI --csv output.csv
```
The `execution` should give one `csv` file and a directory of `ttif` images.
The `csv` File contains the nodule name, and if it's cancer or not, based on the score of three experts that diagnosed each nodule as malignant or benign.
