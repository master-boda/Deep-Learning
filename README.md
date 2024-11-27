## Setup Instructions

1. **Set Up the Images**
   - Run `/src/setup_data.py` to set up the images for further development (after downloading the dataset from Moodle).
   - You will need to specify the path to the dataset in the `setup_data.py` file.

2. **Run Scripts**
   - Run `.py` files from the root directory of the project.

3. **Install Packages**
   - Install the packages from the `requirements.txt` file.
   - Guys, do yourselves a favor and create a virtual environment for this repo.
   - Due to TensorFlow discontinuing support for CUDA in Native Windows, we are using an outdated version (2.10).
   - Consequently, we are also using outdated versions of other packages (such as numpy and pandas).