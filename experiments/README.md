In our paper, we ran experiments five times using different random seeds and reported means and standard deviations. We shared the model checkpoints and example outputs for a single run at https://zenodo.org/record/7259757.

In the following sections, we describe how to reproduce the results of sequential training **MNLI⇒FEVER⇒VITC** with **AEWC**.

## Step 1: Prepare data

```bash
cd data
sh 01_prepare.sh
```

We download the datasets, uncompress them, and create balanced development sets. If everything works properly, we should see:
```bash
+ wc -l mnli/train.jsonl mnli/dev.jsonl mnli/test.jsonl fever/train.jsonl fever/dev.jsonl fever/test.jsonl vitc/train.jsonl vitc/dev.jsonl vitc/test.jsonl
  383702 mnli/train.jsonl
   9000 mnli/dev.jsonl
   9832 mnli/test.jsonl
  178059 fever/train.jsonl
   9000 fever/dev.jsonl
  11710 fever/test.jsonl
  248953 vitc/train.jsonl
   9000 vitc/dev.jsonl
  34481 vitc/test.jsonl
  893737 total
+ wc -l adversarial/test.jsonl symmetric/test.jsonl triggers/test.jsonl
  766 adversarial/test.jsonl
  712 symmetric/test.jsonl
  186 triggers/test.jsonl
 1664 total
```

Then, we create various mixed training and development sets for the mix-and-train and sequential training methods:
```bash
sh 02_create_mixed_data.sh
cd ..
```
The above script should print out the created data statistics.


## Step 2: Train a base model

```bash
cd base/mnli
sbatch -p qgpu 01_train.sh
```

We wrote this script to use `sbatch` on our SLURM GPU cluster. Thus, it contains SLURM directives (e.g., `#SBATCH`) and a command to activate the environment `ewc`. It should work with the shell command `sh` with minimal (or no) modification.

The script contains the argument:
```bash
 --cache_dir "/local/$(whoami)" \
```
to specify the cache directory. You may modify the location to be suitable for your machine. It also contains environment variables:
```bash
HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
```
to force the script to use the local dataset/model files. If you have not used HuggingFace's Transformers before, comment out the above line.

Once the training is done, we should see something like:
```bash
...
Epoch 2: 100%|██████████| 11991/11991 [35:23<00:00, 5.65it/s, loss=0.402, v_num=276608]
Training took '0:36:42.630888'
```

The model checkpoint is kept in `bert-base-uncased-128-mod`.

Then, run:
```bash
sbatch -p qgpu 02_test.sh
```

Again, the script should work with `sh`. The output probabilities and evaluation results are kept in `bert-base-uncased-128-out`.

For example, the output for the MNLI test set should be:
```bash
cat bert-base-uncased-128-out/eval.mnli.test.txt
Confusion Matrix:
   S   R   N
S 2978  133  352
R  151 2779  310
N  287  341 2501

      S   R   N
Prec: 87.18 85.43 79.07
Rec:  85.99 85.77 79.93
F1:  86.58 85.60 79.50

Acc: 83.99
```

## Step 3: Run the 1st sequential training

```bash
cd ../../aewc/mnli_1+fever
sbatch -p qgpu 01_train.sh
```

The above script runs five jobs simultaneously using SLURM's job array. Once all the jobs are done, run:
```bash
sbatch -p qgpu 02_dev.sh
```
to find the best lambda on the development set.

For example, we can check the results by:
```bash
tail -n 1 bert-base-uncased-128-*-out/eval.dev.txt
==> bert-base-uncased-128-1e+0-out/eval.dev.txt <==
Acc: 82.87

==> bert-base-uncased-128-1e-1-out/eval.dev.txt <==
Acc: 81.82

==> bert-base-uncased-128-1e+1-out/eval.dev.txt <==
Acc: 82.74

==> bert-base-uncased-128-1e-2-out/eval.dev.txt <==
Acc: 81.7

==> bert-base-uncased-128-1e+2-out/eval.dev.txt <==
Acc: 80.39
```

After knowing the lambda value that yields the best accuracy on the development set (`1e+0` in this experiment), write it to `lambda_best.txt` and run:
```bash
cat lambda_best.txt
1e+0
sbatch -p qgpu 03_test.sh
```
to get the final outputs and evaluation results.

For example:
```bash
tail -n 1 bert-base-uncased-128-1e+0-out/eval.*.test.txt
==> bert-base-uncased-128-1e+0-out/eval.adversarial.test.txt <==
Acc: 53.66

==> bert-base-uncased-128-1e+0-out/eval.fever.test.txt <==
Acc: 87.36

==> bert-base-uncased-128-1e+0-out/eval.mnli.test.txt <==
Acc: 78.75

==> bert-base-uncased-128-1e+0-out/eval.symmetric.test.txt <==
Acc: 81.32

==> bert-base-uncased-128-1e+0-out/eval.triggers.test.txt <==
Acc: 70.43

==> bert-base-uncased-128-1e+0-out/eval.vitc.test.txt <==
Acc: 62.4
```

## Step 4: Run the 2nd sequential training

The process is the same as the 1st one:
```bash
cd ../mnli_1+fever_1+vitc/
sbatch -p qgpu 01_train.sh
sbatch -p qgpu 02_dev.sh
```

After knowing the best lambda value (`1e+0` in this experiment), write it to `lambda_best.txt` and run:
```bash
cat lambda_best.txt
1e+0
sbatch -p qgpu 03_test.sh
```

We should see something like:
```bash
tail -n 1 bert-base-uncased-128-1e+0-out/eval.*.test.txt
==> bert-base-uncased-128-1e+0-out/eval.adversarial.test.txt <==
Acc: 46.08

==> bert-base-uncased-128-1e+0-out/eval.fever.test.txt <==
Acc: 82.6

==> bert-base-uncased-128-1e+0-out/eval.mnli.test.txt <==
Acc: 78.45

==> bert-base-uncased-128-1e+0-out/eval.symmetric.test.txt <==
Acc: 76.97

==> bert-base-uncased-128-1e+0-out/eval.triggers.test.txt <==
Acc: 75.81

==> bert-base-uncased-128-1e+0-out/eval.vitc.test.txt <==
Acc: 80.27
```
