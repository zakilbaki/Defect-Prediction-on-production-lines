# Dataset setup

Download the files from
[Valeo Challenge Data #36](https://challengedata.ens.fr/challenges/36) after
registering for access, then place them in this directory:

```text
data/
├── traininginputs.csv
├── trainingoutput.csv
└── testinputs.csv
```

The CSV files are intentionally ignored by Git. The notebook expects all three files;
the reusable training command uses the two labeled training files.
