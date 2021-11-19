# PLELog 
 
 [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4470181.svg)](https://doi.org/10.5281/zenodo.4470181)
 
 
**Please be NOTED that we are going on some major improve work on PLELog, the current version is useable but not as well as I was expected. We'll keep updating codes and make sure it is easy for anyone who are interested in our research. Sorry for the inconvenience during this time. New PLELog will see you soon!**

This is the basic implementation of our submission in ICSE 2021: **Semi-supervised Log-based Anomaly Detection via Probabilistic Label Estimation**.
- [PLELog](#plelog)
  * [Description](#description)
  * [Project Structure](#project-structure)
  * [Datasets](#datasets)
  * [Reproducibility:](#reproducibility-)
    + [Environment:](#environment-)
    + [Preparation](#preparation)
  * [Anomaly detection](#anomaly-detection)
  * [Contact](#contact)

## Description

`PLELog` is a novel approach for log-based anomaly detection via probabilistic label estimation. 
It is designed to effectively detect anomalies in unlabeled logs and meanwhile avoid the manual labeling effort for training data generation.
We use semantic information within log events as fixed-length vectors and apply `HDBSCAN` to automatically clustering log sequences. 
After that, we also propose a Probabilistic Label Estimation approach to reduce the noises introduced by error labeling and put "labeled" instances into `attention-based GRU network` for training. 
We conducted an empirical study to evaluate the effectiveness of `PLELog` on two open-source log data (i.e., HDFS and BGL). The results demonstrate the effectiveness of `PLELog`. 
In particular, `PLELog` has been applied to two real-world systems from a university and a large corporation, further demonstrating its practicability.

## Project Structure

```
├─approaches  #HDBSCAN & RNN approaches here, including training, validating, and testing.
├─config      
├─data        #Code for data processing.
├─utils
├─dataset
│  ├─BGL      #Sample data for BGL (Quick start)
├─model       #RNN models.
├─module      #Anomaly detection modules, including classifier, Attention, etc.
├─outmodel    #Model parameters for trained models, detailed save path is set in config files.
├─logs       
├─output_res  #Output result of Attention-Based GRU classification model.
├─pipeline.py #Main entrance code.
└─test.py     #Quick start for PLELog
```

## Datasets

We used `2` open-source log datasets, HDFS and BGL. 
In the future, we are planning on testing `PLELog` on more log data.

| Software System | Description                        | Time Span  | # Messages | Data Size | Link                                                      |
|       ---       |           ----                     |    ----    |    ----    |  ----     |                ---                                        |
| HDFS            | Hadoop distributed file system log | 38.7 hours | 11,175,629 | 1.47 GB   | [LogHub](https://github.com/logpai/loghub)                |
| BGL             | Blue Gene/L supercomputer log      | 214.7 days | 4,747,963  | 708.76MB  | [Usenix-CFDR Data](https://www.usenix.org/cfdr-data#hpc4) |

## Reproducibility

### Environment

**Note:** We attach great importance to the reproducibility of `PLELog`. To run and reproduce our results, please try to install the suggested version of the key packages.

**Key Packages:**

```
PyTorch v1.5.1
python v3.8.3
hdbscan v0.8.26
overrides v3.1.0
```

The mainly required python packages including PyTorch, overrides, hdbscan, scikit-learn.
`Anaconda` is recommended to manage those packages and their versions.
hdbscan and overrides are not available while using anaconda, try using pip.

### Preparation

You need to follow these steps to **completely** run `PLELog`.
- **Step 1:** To run `PLELog` on different log data, create a directory under `dataset` folder **using unique and memorable name**(e.g. HDFS and BGL). `PLELog` will try to find the related files and create logs and results according to this name.
- **Step 2:** Move target log file (plain text, each raw contains one log message) into the folder of step 1.
- **Step 3:** Run `utils/Drain.py` (make sure it has proper parameters) to finish log parsing and extract log templates. You can find the details about Drain parser from [IBM](https://github.com/IBM/Drain3).
- **Step 4:** Download [Stanford NLP word embeddings](https://nlp.stanford.edu/projects/glove/), rename as `nlp-word.vec` and put it under `dataset` folder.

**Note:** Since log can be very different, here in this repository, we only provide the processing approach of HDFS and BGL w.r.t our experimental setting.
If you want to apply `PLELog` on new log data, please refer to the `prepare_data` method in `pipeline.py` to add new pre-process methods.

## Anomaly Detection

- **Complete:** You can run `PLELog` from the ground up by running `pipeline.py` after the preparation. The results will be shown in the `logs` folder named after detailed settings. And the classification results are saved in the `output_res` folder for further analysis.
- **Quick Start:** Since `HDBSCAN` may need hours to finish, we provide a trained model (on `BGL` dataset) and a test input as a quick start for `PLELog`, just run `test.py` under the correct environment.
  Logs will be written in `log/test.log`, you can find the results at the end of the file.
Feel free to play with `PLELog` through the command parameters below: (The results of different settings should be separated, don't worry! :P)

```
usage: pipeline.py [-h] [--config_file CONFIG_FILE] [--gpu GPU] [--hdbscan_option HDBSCAN_OPTION]
                   [--dataset DATASET] [--train_ratio TRAIN_RATIO] [--dev_ratio DEV_RATIO]
                   [--test_ratio TEST_RATIO] [--min_cluster_size MIN_CLUSTER_SIZE]
                   [--min_samples MIN_SAMPLES] [--reduce_dim REDUCE_DIM]
optional arguments:
  -h, --help            show this help message and exit
  --config_file CONFIG_FILE
                        Configuration file for Attention-Based GRU Network.
  --gpu GPU             GPU ID if using cuda, -1 if cpu.
  --hdbscan_option HDBSCAN_OPTION
                        Different strategies of HDBSCAN clustering. 0 for PLELog_noP, 1 for PLELog, -1 for upperbound.
  --dataset DATASET     
                        Choose dataset, HDFS or BGL.
  --train_ratio TRAIN_RATIO
                        Ratio of train data. Default 6.
  --dev_ratio DEV_RATIO
                        Ratio of dev data. Default 1.
  --test_ratio TEST_RATIO
                        Ratio of test data. Default 3.
  --min_cluster_size MIN_CLUSTER_SIZE
                        Minimum cluster size, a parameter of HDBSCAN.
  --min_samples MIN_SAMPLES
                        Minimum samples, a parameter of HDBSCAN.
  --reduce_dim REDUCE_DIM
                        Target dimension of FastICA.
  --thredshold THRESHOLD
                        Threshold for final classification, any instance with "anomalous score" higher than this threshold will be regarded as anomaly.
```

## Contact

We are happy to see `PLELog` being applied in the real world and willing to contribute to the community. Feel free to contact us if you have any question!
Authors information:

| Name          | Email Address          | 
| ------------- | ---------------------- | 
| Lin Yang      | linyang@tju.edu.cn     |
| Junjie Chen * | junjiechen@tju.edu.cn  |
| Weijing Wang  | wangweijing@tju.edu.cn |

\* *corresponding author*
