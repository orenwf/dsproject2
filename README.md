# Big Data Project 2 - Calculating Similarity Scores

## Setup

- assumes Ubuntu 18.04 LTS or newer
- `cd dsproject2`
- `chmod +x scripts/*`
- setup current machine as master: `./scripts/setup_spark.sh`
- WARNING: THIS WILL DOWNLOAD AND INSTALL SPARK FROM APACHE AT `/usr/local/spark`
  - skip this if you have already set up spark
- For any workers to be set up:
  - must have master ssh key authorized
  - `./scripts/setup_worker.sh` $(worker IP address)

## Running Jobs

- `/usr/local/bin/spark-submit similarityscore.py` name-of-input-file term1 term2 ...
  - each term will be searched for against the corpus for similarity score and up to the top 5 most similar temrs will be ranked
    - for up to top n similar terms use flag `--max` n
  - this will run the script and output a text file with the results in the format of `mapreduce.result.`name-of-text-file`.txt` with an integer concatenated to the end to differentiate
- the tool automatically filters for terms matching `dis_`X`_dis` and `gene_`X`_gene` but can be run with no filtering using the `--nofilter` flag
- limit the results rankings to only terms that match `--search` some-python-regex
- `/usr/local/bin/spark-submit similarityscore.py --help` for info on more flags and options

## Running Tests

- `usr/local/bin/spark-submit similarityscoretests.py` name-of-input-file term1 term2
  - the same flags can be used to tests
- this runs the script against a single threaded algorithm to check the output of spark is the same - avoid giving it large input
