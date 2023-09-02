# Repository for Research about lung nodules detection using AI.
----------------------------------------------------------------

## Create a TRPMLN dataset to train AI models that may be able to detect normal cancer/nodules.

We should have a new empty directory to store `tiff` images by using something like `mkdir ROI/` in UNIX, and we should have the `LIDC-LDRI` Dataset.
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

### USAGE the TRPMLN as a Dataset to Train a model.

The main goal of our dataset `TRPMLN` is to simply get these values `X_train`, `X_test`, `y_train`, and `y_test` by applying some preprocessing steps to the `LIDC-IDRI` dataset to make it suitable for our CNN model.

```py
df = pd.read_csv("output.csv")
filenames = df["roi_name"].tolist()
labels = df["cancer"].tolist()
# read_image(filename) function should retuen the img array using PIL module
images = [read_image('ROI/' + filename) for filename in filenames]
X = np.array(images)
# Reshape to add channel dimension for grayscale images
X = X.reshape(-1, 64, 64, 1)
y = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split( X, y, random_state=42, test_size=0.2)
```
