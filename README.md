# Automatic identification of verbal multiword expressions
Tis is the source code of the system that participated to the 2017 shared task on automatic identification of verbal multiword expressions (VMWEs). It is called MUMULS (MUltilingual MULtiword Sequences) and was a winner only for Romanian among 18 languages.
However, it was one of few systems that could categorize VMWEs type.
[Shared task web](http://multiword.sourceforge.net/PHITE.php?sitesig=CONF&page=CONF_05_MWE_2017___lb__EACL__rb__&subpage=CONF_40_Shared_Task)
[Results](http://multiword.sourceforge.net/PHITE.php?sitesig=CONF&page=CONF_05_MWE_2017___lb__EACL__rb__&subpage=CONF_50_Shared_Task_Results)

## Scripts for MWE Shared task:

The system was implemented using a supervised approach based on recurrent neural networks using the open source library TensorFlow. It is designed to predict one of five tags given a file in .conllu format. The tags are: IReflV(inherently reflexive verb), LVC (light verb construction), VPC (Verb-particle construction), ID (idiomatic expression) and OTH - other types. 

====Tensorflow:=====

mwe_dataset.py, mwe_tagger.py

Both cloned from https://github.com/ufal/npfl114/tree/master/labs08 it was a homework for a Deep Learning course by Milan Straka, and adjusted to the shared task.

### Shared task data
I used the experimental data set of the shared task which included training and evaluation data sets, available in 18 languages. The corpus was manually annotated with VMWE tags. So the two files were provided: .conllu with morphosyntactic annotation and .parsemetsv with the annotation of MWE itself. Short snippets:
```conllu
1       Zdá     zdát    VERB    VB      Mood=Ind|Negative=Pos|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin|Voice=Act    _       _       _       _
2       se      se      PRON    P7      Case=Acc|PronType=Prs|Reflex=Yes|Variant=Short  _       _       _       _
3       vám     ty      PRON    PP      Case=Dat|Number=Plur|Person=2|PronType=Prs      _       _       _       _
4       ,       ,       PUNCT   Z:      _       _       _       _       _
5       že      že      SCONJ   J,      _       _       _       _       _
6       finanční        finanční        ADJ     AA      Animacy=Inan|Case=Nom|Degree=Pos|Gender=Masc|Negative=Pos|Number=Sing   _       _       _       _
7       úřad    úřad    NOUN    NN      Animacy=Inan|Case=Nom|Gender=Masc|Negative=Pos|Number=Sing      _       _       _       _
8       nepostupoval    postupovat      VERB    Vp      Gender=Masc|Negative=Neg|Number=Sing|Tense=Past|VerbForm=Part|Voice=Act _       _       _       _
9       správně správně ADV     Dg      Degree=Pos|Negative=Pos _       _       _       _
10      ?       ?       PUNCT   Z:      _       _       _       _       _

```
```parsemetsv
1       Zdá     _       1:IReflV
2       se      _       1
3       vám     nsp     _
4       ,       _       _
5       že      _       _
6       finanční        _       _
7       úřad    _       _
8       nepostupoval    _       _
9       správně nsp     _
10      ?       _       _
```

### Data preparation
You should divide your data into train, dev and test set (there is an option to run training on blind test file).
The format of the data that you should feed to a script:
```
Zdá     zdát    VERB    IReflV
se      se      PRON    CONT
vám     ty      PRON    _
,       ,       PUNCT   _
že      že      SCONJ   _
finanční        finanční        ADJ     _
úřad    úřad    NOUN    _
nepostupoval    postupovat      VERB    _
správně správně ADV     _
?       ?       PUNCT   _
```
Where *LVC* is an MWE tag, and *CONT* is the continuation of the MWE.
If you use the data of the shared task, you can try running the transformation scripts that I wrote as follows:
```
convert_formats.py
```
It browses the language directories, takes .parsemetsv and .conllu files for each language and generates two files: one with form+lemma+pos tokens, the other the respective MWE annotation column.
```
process_folders.sh
```
Pastes the two files generated from the previous script, divides the whole file into dev (first 10%), train (next 80) and test (last 10). It also strips the id column of conllu as those factors are not needed for training, and stors this column for test . Maybe one day I will write something more sophisticated for cross-validation. 



### Run the script:
python mwe_tagger.py --data_train="train" --data_dev="dev" --data_test="test"
If run without those data parameters, it will take a small subset of the Czech training data (50000cs-train.txt, cs-dev.txt, cs-test.txt).
It will output the file with predictions almost in parsemetsv format. 

### postprocessing
```
evaluate_all.sh 
```
In order to satisfy the evaluate.py conditions, we substitute MWE tag with order_of_the_MWE+MWEtag, and the symbol CONT with respective number, just as it appeared in raw training data (tf2parseme.py).
The script then evaluates all the outputs for all languages that are located in the language folders and writes a log using a script bin/evaluate.py from the shared task.


# Refactoring experiments: Character-level NN
