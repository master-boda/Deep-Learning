- Run `/src/setup_data.py` to set up the data split and update the image metadata csv (after downloading the dataset from Moodle).
- You will need to specify your own path to the downloaded dataset in the `setup_data.py` file.

- Due to TensorFlow discontinuing support for CUDA in Native Windows, we are using an outdated version (2.10).
- Consequently, we are also using outdated versions of other packages (such as numpy and pandas).