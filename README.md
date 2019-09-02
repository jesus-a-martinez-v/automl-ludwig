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
ludwig train --data_train_csv fashion_mnist_train.csv --data_test_csv fashion_mnist_test.csv --model_definition_file model_definition.yml```
```

Finally, evaluate the model:

```bash
ludwig test --data_csv fashion_mnist_test.csv --model_path results/experiment_run_[i]/model/
```

Where `i` is the number of the experiment that produced the desired model.


## NOTE

For some reason, I could not make Ludwig work with `tensorflow-gpu`, so because it is using your CPU, it'll take a really long time to train.
