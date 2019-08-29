# AutoML Ludwig

Simple introductory project to automatic machine learning with Ludwig.

## Installation

I recommend you use `virtualenv`. If you have it installed, create a new environment and activate it:

```bash
$> virtualenv -p python3 venv
$> source venv/bin/activate
```

After that, just run the following commands:

```bash
$> sudo apt-get install -y libgmp-dev libmpfr-dev libmpc-dev
$> pip install -r requirements.txt
```

## Try It

First, you need to build the dataset with:

```bash
python build_fashion_mnist.py
```

Then, just run the following command to fire up an automatic search:

```bash
ludwig train --data_train_csv fashion_mnist_train.csv --data_test_csv fashion_mnist_test.csv --model_definition_file model_definition.yml > output.txt```
```

Keep in mind this is a heavy and long running process. You'll see the results in `output.txt` after a few hours.


## NOTE

For some reason, I could not make Ludwig work with `tensorflow-gpu`, so because it is using your CPU, it'll take a really long time to train.
