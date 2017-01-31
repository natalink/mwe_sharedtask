#!/usr/bin/python
# -*- coding: utf-8 -*-

import xml.etree.ElementTree as et
import sys
import re
from os.path import basename
import os
import sys
import re

rootdir = "data_mwe"


def formatting(input_file, filename):
    file1 = open(filename, 'w')
    if not os.path.exists(input_file):
        print ("input_file file does not exist", input_file)
        return 0
    print ("Exists file ", input_file)
    newattributes = []
    counts = 1
    stack_of_mwes = []
    mwe_tagnames = []
    for line in open(input_file):
        if line not in ['\n', '\n\r']:
            #print "parsemetsv:" + line
            line.rstrip("\n")
            attributes = line.split("\t")
            attributes[2].rstrip("\n")
            mwe_tag = re.compile("(\w+)]")
            if (attributes[2] == '\n'):
                print ("THIS IS EMPTY")
                newattributes.append(attributes[0] + '\t' + '_' + '\t' + '_\n')
            elif not attributes[2].startswith("_") and not attributes[2].startswith("CONT"):
                #print ("attributes[2] ", attributes[2])
            #if mwe_tag.match(attributes[2]):
                #print ("MATCH: ", attributes[2])
                # first occurence of an mwe tag
                stack_of_mwes.append(attributes[2])
                mwe = "{}:{}".format(str(counts), attributes[2])
                newattributes.append(attributes[0] + '\t' + '_' + '\t' + mwe)
                counts += 1

            elif (attributes[2].startswith("CONT")):
                count = len(stack_of_mwes)
                if count ==0:
                    count +=1
                newattributes.append(attributes[0] + '\t' + '_' + '\t' + str(count) +"\n")

            else:
                newattributes.append(attributes[0] + '\t' + '_' + '\t' + attributes[2])
        else:
            #print(newattributes)
            #print("\n")
            file1.write("".join(newattributes))
            file1.write("\n")
            newattributes = []
            counts=1
            stack_of_mwes = []

    file1.close()

def main():
    import argparse
    import subprocess
    parser = argparse.ArgumentParser()

    parser.add_argument("--tf_output", type=str, help="process columns from TF output")
    #parser.add_argument
#    parser.add_argument("--typefile", type=str, help="")
    args = parser.parse_args()
    from subprocess import call
    argv = sys.argv
    #print ("I am here")
    if len(argv) !=2:
        print 'USAGE: {} <file with predictions from tf>'.format(argv[0])
        sys.exit(0)
    datafile = args.tf_output

    import os.path
    langpath = datafile.split("/")[1]
    experiment= datafile.split("/")[2]
    outfile= "sharedtask-data/" + langpath + "/" + "predictions_raw_{}".format(experiment)
    #print (datafile, langpath)
    formatting(datafile, outfile)


if __name__ == "__main__":
    main()
