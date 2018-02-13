#!/usr/bin/python
# -*- coding: utf-8 -*-

import re
import os
import argparse
from itertools import izip
from __builtin__ import str
import subprocess


pattern = re.compile("^CS$")
rootdir = "sharedtask_data"

def num_there(s):
    return any(i.isdigit() for i in s)

def tags2nums(line): #_ _ _ _ _ _ _ _ LVC:FIRST LVC:CONT  -------> _ _ _ _ _ _ _ _ 1:LVC 1
    new_tags = []
    stack_of_mwes = []
    mwe_tags = line.split()
    counts = 1

    for tag in mwe_tags:

        if "_" in tag:
            new_tags.append(tag)

        elif "FIRST" in tag:

            stack_of_mwes.append(tag)
            print "COUNT in FIST: ", counts
            tag = tag.replace(':FIRST', '')
            tag = str(counts) + ":" + tag
            new_tags.append(tag)
            counts += 1
            print "COUNTS AFTER+: ", counts

        elif "CONT" in tag:
            count = len(stack_of_mwes)
            if count ==0:
                count +=1
                new_tags.append(str(count))

    return "\t".join(new_tags)


def sent2line(input_format):
    with open(input_format) as file_input_format:
        sentences = file_input_format.readlines()
        transformed = []
        for sentence in sentences:
            words = sentence.split(' ')
            for word in words:
                if "\n" not in word:
                    transformed.append(word)
                else:
                    word = word.rstrip('\n')
                    transformed.append(word)
                    transformed.append("\n")
    return transformed

def read_lines(input_format, test_blind_parsemetsv, transformed):
    with open(test_blind_parsemetsv) as file_parsemetsv:

        for input_line, line_parsemetsv in izip(input_format, file_parsemetsv):

            if input_line.startswith('#'):
                continue

            input_line = input_line.strip()
            input_line = tags2nums(input_line)
            line_parsemetsv = line_parsemetsv.strip()
            #print "line_parsemetsv :", line_parsemetsv
            line_parsemetsv = "\t".join(line_parsemetsv.split('\t')[:-1])
            #print "line_parsemetsv AFTER :", line_parsemetsv
            print("{1}\t{0}".format(input_line, line_parsemetsv))


def main():
    
    parser = argparse.ArgumentParser()
    for root, subFolders, files in os.walk(rootdir):
        for lang_folder in subFolders:            
            if "BG" in lang_folder or "HE" in lang_folder or "LT" in lang_folder: #no conllu for those files; skip
                continue
            elif pattern.match(lang_folder): #only language folders
                print "processing FOLDER: " + lang_folder
                input_format = rootdir + "/" + lang_folder + "/val_test.tags" #_ _ _ _ _ _ _ _ LVC:FIRST LVC:CONT IReflV:FIRST _ IReflV:CONT
                test_blind_parsemetsv = rootdir + "/" + lang_folder + "/test.blind.parsemetsv"
                transformed = rootdir + "/" + lang_folder + "/transformed.parsemetsv"
                file_transformed = open(transformed, 'w')
                vertical_input = sent2line(input_format)
                read_lines(vertical_input, test_blind_parsemetsv, file_transformed)


if __name__ == "__main__":
    
    main()
    
