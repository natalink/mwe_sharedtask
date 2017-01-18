#!/usr/bin/python
# -*- coding: utf-8 -*-

import xml.etree.ElementTree as et
import sys
import re
from os.path import basename
import os
import sys
import re
pattern = re.compile("^FR$")
rootdir = "data_mwe"



def formatting(folder, input_file, dataformat):
    outfile = rootdir + "/" + "/{}_{}".format(folder, dataformat)
    outtags = rootdir + "/" + "/MWETAGS_{}_{}".format(folder, dataformat)
    file1 = open(outfile, 'w')
    #file2 = open(outtags, 'w')
    if not os.path.exists(input_file):
        print "input_file file does not exist" + input_file
        return 0
    print "Exists file " + input_file
    newattributes = []
    mwetags = []
    for line in open(input_file):

        if line.startswith("#"):
            continue

        if line not in ['\n', '\n\r']:
            print "parsemetsv:" + line
            attributes = line.split("\t")
            if dataformat == 'forms':
                newattributes.append(attributes[1] + "_" + attributes[3])
            elif dataformat =='lemma_tag':
                newattributes.append(attributes[2] + "_" + attributes[3])
            elif dataformat == 'for_TF' and input_file.endswith('.conllu'):
                newattributes.append(attributes[2] + "_" + attributes[3])
            elif dataformat == 'for_TF' and input_file.endswith('.parsemetsv'):
                mwetags.append(attributes[3])
            else:
                pass

        else:
            print(newattributes)
            #print(mwetags)
            print("\n")
            file1.write("\n".join(newattributes))
            file1.write("\n")
            #file2.write("".join(mwetags))
            #file2.write("\n")
            newattributes = []
            mwetags = []


    file1.close()
    #file2.close()



def main():
    argv = sys.argv
    if len(argv) !=2:
        print 'USAGE: {} <format: forms, lemmas_tags, for_TF>'.format(argv[0])
        sys.exit(0)
    dataformat = argv[1]
    for root, subFolders, files in os.walk(rootdir):
        for folder in subFolders:
        #print folder
            if pattern.match(folder): #only language folders
            #print folder
                input_file = rootdir + "/" + folder + "/train.conllu"
                parseme = rootdir + "/" + folder + "/train.parsemetsv"
                if formatting(folder, input_file, dataformat) == 0:
                        continue
                elif dataformat =='for_TF':
                    formatting(folder,input_file,dataformat)
                    #formatting(folder,parseme,dataformat)

                else:
                    formatting(folder, input_file, dataformat)



if __name__ == "__main__":
    main()
