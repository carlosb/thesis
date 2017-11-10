# Description
Encodes the last column of the given dataset between 0 and (n_classes - 1)
by specfying a separator. If no separator is specified it will attempt to
automatically detect the separator. It will overwrite or create a new file
with the extension: `.csv.encoded`
*Note*: This is really just a wrapper around the pandas and sklearn functions

# Help
To output the help please run this command:
````
python labelencoder.py -h
````
To get the output:
````
usage: 
    Encodes the last column of the given dataset between 0 and (n_classes - 1)
    by specfying a separator. If no separator is specified it will attempt to
    automatically detect the separator. It will overwrite or create a new file
    with the extension: .csv.encoded

       [-h] [-s SEP] [-cs CHUNKSIZE] [-cl CLASSES] filename [filename ...]

positional arguments:
  filename              Name of file to encode

optional arguments:
  -h, --help            show this help message and exit
  -s SEP, --sep SEP     Delimiter to use. If no sep is specified, the C engine
                        cannot automatically detect the separator, but the
                        Python parsing engine can, meaning the latter will be
                        used and automatically detect the separator by
                        Python's builtin sniffer tool, csv.Sniffer. In
                        addition, separators longer than 1 character and
                        different from '\s+' will be interpreted as regular
                        expressions and will also force the use of the Python
                        parsing engine. Note that regex delimiters are prone
                        to ignoring quoted data. Regex example: ' '
  -cs CHUNKSIZE, --chunksize CHUNKSIZE
                        Number of bytes to read and write at a time. You must
                        pass the actual labels of the classes using --classes.
  -cl CLASSES, --classes CLASSES
                        The labels of the classes.
````
# Example
````
python labelencoder.py *.txt
````

You should get the following output:

````
Encoding:
... australian.txt
... blood_transfusion.txt
... breast_cancer.txt
... bupa.txt
... german.txt
... haberman.txt
... heart.txt
... planning_relax.txt
... sonar.txt
... vertebral_column.txt

These files were succesful: 
... australian.txt
... blood_transfusion.txt
... breast_cancer.txt
... bupa.txt
... haberman.txt
... heart.txt
... planning_relax.txt
... sonar.txt
... vertebral_column.txt

These files could not be encoded: 
... german.txt
````

In order to encode `german.txt` we have to note that it has some particular
separation that the parser can't automatically detect. Specifically, each entry
is separated by a variable number of white spaces. So in this case we will have to
specify the regex '[ ]+'. Using the command this looks like:
````
python labelencoder.py --sep='[ ]+' german.txt
````
The result should look like:
````
Encoding:
... german.txt

These files were succesful: 
... german.txt
All files encoded!
````


