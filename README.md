# pure
A collaborative project with pure technologies.

### angle dependence
`class_analysis.py` contains two classes, Signal and Transducer. `Signal` is a class that represents a single waveform produced by a given transducer. `Transducer` is a class that contains the data and results for the angle-dependence experiment for a given Transducer.

`class_angle.py` contains the class Micrometer which represents the calibrated function model for the rotating metal sample.

`clean_csv.py` contains a function which removes the first N rows in all spreadsheets in a given directory. This is useful because the spreadsheets saved from the oscilloscope contain 2 row headers which are strings. The function writes the new arrays into `.csv` files into a subfolder named `clean` within the given working directory.
