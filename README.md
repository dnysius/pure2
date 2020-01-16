# pure
A collaborative project with pure technologies.

### Accessing the data
The folders containing the data are stored in /pure2/data.zip. Extract this folder and place the folders to the repository's parent directory. For example, if the repo is located in .../GitHub/pure2/, move the folders to /GitHub/.

### Scan-specific configuration file
The file `conf.txt` must be created for each scan folder to setup the configuration for plotting the data.

### Plotting
Use `1d_bscan.py` to create a B-scan on the acquired data. Change the string contained in `DATA_FOLDER` to the folder name of the scan you are working with.