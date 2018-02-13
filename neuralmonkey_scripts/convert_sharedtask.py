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

def process_block(block): 
    mwe_list = []
    lemma_list = []
    word_list = []
    line_count = 0
    for index_line, line in enumerate(block):
        if num_there(line):
            attributes = line.split()
            mwe_list.append(attributes[-1]) #last element is mwe tag from parsemetsv
            lemma_list.append(attributes[1]) #third attribute from conllu is lemma 
            word_list.append(attributes[0])
    
    dict_vmwe = {}
    dict_lemma = {}    
    dict_vmwe, dict_lemma = stack_vmwe_positions(mwe_list, lemma_list, dict_vmwe, dict_lemma)
    length_of_sentence = len(mwe_list)
    out_vmwe_list, out_lemma_list, out_vmwe_dependency_list, out_vmwe_number_list = form_columns(dict_vmwe, dict_lemma, length_of_sentence)
    newblock = []

    for line in block:
        if num_there(line):
            attrs = line.split()
            del attrs[-4:]
            # done earlier attrs[1], attrs[0] = attrs[0], attrs[1] #swap ID and wordform (While compiling, manatee script will treat first column as word forms)
            newblock.append("\t".join(attrs))
            line_count+=1
    
    outblock = word_list
    mweblock = mwe_list
    print outblock
    #outblock = [ "{}\t{}\t{}\t{}\t{}".format(a, b, c,d,f) for a, b, c, d, f in  zip(newblock, out_vmwe_list, out_vmwe_dependency_list, out_vmwe_number_list, out_lemma_list ) ]
    return outblock, out_vmwe_list, line_count
    
def form_columns (dict_vmwe, dict_lemma, length_of_sentence):
    out_vmwe_list = ['_'] * length_of_sentence # type
    out_vmwe_number_list = ['_'] * length_of_sentence #
    out_lemma_list = ['_'] * length_of_sentence
    out_vmwe_dependency_list = ['_'] * length_of_sentence
    
    for mwe_number, vmwe_tags in dict_vmwe.iteritems():
        type = vmwe_tags.split("_")[0]
        positions = vmwe_tags.split("_")[1:]
        lemma = dict_lemma[mwe_number]

        
        for ind, pos in enumerate(positions): # ugly hack!!!
            if ind == 0 and out_vmwe_list[int(pos)] == '_': # :head, single
                out_vmwe_list[int(pos)] = type + ":FIRST"
                out_vmwe_dependency_list[int(pos)] = "head"
                out_lemma_list[int(pos)] = lemma
                out_vmwe_number_list[int(pos)] = mwe_number
                
            elif ind == 0 and out_vmwe_list[int(pos)] != '_':
                
                out_vmwe_list[int(pos)] =  out_vmwe_list[int(pos)] + ";" + type + ':FIRST'# + ":head"
                out_vmwe_dependency_list[int(pos)] = out_vmwe_dependency_list[int(pos)] + ";" + "head"
                out_lemma_list[int(pos)] =  out_lemma_list[int(pos)] + ";" + lemma
                out_vmwe_number_list[int(pos)] = out_vmwe_number_list[int(pos)] + ";" + mwe_number
                
            elif ind != 0 and out_vmwe_list[int(pos)] != '_':# ['_', '1:LVC', '_', '1;2:LVC', '_', '_', '_', '_', '2', '_']
                # this may have some bugs, test it on more use cases!!!   
                out_vmwe_list[int(pos)] =  out_vmwe_list[int(pos)] + ";" + type+ ':CONT'# + ":head"
                out_vmwe_dependency_list[int(pos)] = out_vmwe_dependency_list[int(pos)] + ";" + "child"
                out_lemma_list[int(pos)] =  out_lemma_list[int(pos)] + ";" + lemma
                out_vmwe_number_list[int(pos)] = out_vmwe_number_list[int(pos)] + ";" + mwe_number

            else:
                out_vmwe_list[int(pos)] = type+':CONT'# + ":child"
                out_vmwe_dependency_list[int(pos)] = "child"
                out_lemma_list[int(pos)] = lemma
                out_vmwe_number_list[int(pos)] = mwe_number
                
    print out_vmwe_list, out_lemma_list, out_vmwe_dependency_list, out_vmwe_number_list
    return out_vmwe_list, out_lemma_list, out_vmwe_dependency_list, out_vmwe_number_list
            
           
#########Given mwe and lemma attribute lists returns the positions. Output: mwe_list: {'1': 'LVC_3_5'} lemma_list:{'1': 'il_de'} ############
def stack_vmwe_positions(mwe_list, lemma_list, dict_vmwe, dict_lemma):
   
    for ind, mwe in enumerate(mwe_list):
        if ";" in mwe:
            multiple_mwes = mwe.split(";")
            for mul_mwe in multiple_mwes:
                if ":" in mul_mwe:
                    number_mwe, type_mwe = mul_mwe.split(":")
                    position_type = type_mwe + "_" + str(ind)
                    dict_vmwe[number_mwe] = position_type
                    lemma = lemma_list[ind]
                    dict_lemma[number_mwe] = lemma
                else: # ['_', '1:LVC', '_', '1;2:LVC', '_', '_', '_', '_', '2', '_'] quick_ugly_hack!! need to make some recursion instead.
                    if mul_mwe in dict_vmwe:
                        current_value = dict_vmwe.get(mul_mwe)
                        new_value = current_value + "_" + str(ind)
                        dict_vmwe.update({mul_mwe: new_value})
                        current_lemma = dict_lemma.get(mul_mwe)
                        new_lemma = current_lemma + " " + lemma_list[ind] # space iof _
                        dict_lemma.update({mul_mwe: new_lemma})
                
        elif ":" in mwe and not ";" in mwe:
            number_mwe, type_mwe = mwe.split(":")
            position_type = type_mwe + "_" + str(ind)
            dict_vmwe[number_mwe] = position_type
            lemma = lemma_list[ind]
            dict_lemma[number_mwe] = lemma

                                        
        elif mwe.isdigit():
            if mwe in dict_vmwe:
                current_value = dict_vmwe.get(mwe)
                new_value = current_value + "_" + str(ind)
                dict_vmwe.update({mwe: new_value})
                current_lemma = dict_lemma.get(mwe)
                new_lemma = current_lemma + " " + lemma_list[ind] # space iof _
                dict_lemma.update({mwe: new_lemma})
        
    return dict_vmwe, dict_lemma
        
   
def read_blocks(input_conllu, parsemetsv, file2, file3):   
   
    with open(input_conllu) as file_conllu, open(parsemetsv) as file_parsemetsv: 
        blocks = []
        for line, line_parsemetsv in izip(file_conllu, file_parsemetsv):
           
            if line.startswith('#'):
                continue
        
            if line not in ['\n', '\n\r', '\t\n']:
                
                line = line.strip()
                attrs = line.split("\t")
                attrs[0],attrs[1] = attrs[1], attrs[0]
                attrs[1],attrs[2] = attrs[2], attrs[1]
                swapped_line = "\t".join(attrs)
                line_parsemetsv = line_parsemetsv.strip()
                blocks.append(swapped_line+"\t"+line_parsemetsv)                              

            else:

                wordblock, mweblock, line_count = process_block(blocks)
                #file2.write("\n")
                file2.write(" ".join(wordblock))
                file2.write("\n")
                file3.write(" ".join(mweblock))
                file3.write("\n")
                empty_lines = 0
                blocks = []
            
                 
def file_len(fname):
    for i, l in enumerate(fname):
        pass
    return i + 1

            
def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, help="test.blind or train")
    for root, subFolders, files in os.walk(rootdir):
        for lang_folder in subFolders:            
            if "BG" in lang_folder or "HE" in lang_folder or "LT" in lang_folder: #no conllu for those files; skip
                continue
            elif pattern.match(lang_folder): #only language folders
                print "processing FOLDER: " + lang_folder
                conllu_train = rootdir + "/" + lang_folder + "/test.conllu"
                parsemetsv_train = rootdir + "/" + lang_folder + "/test.parsemetsv"
                conllu_test = rootdir + "/" + lang_folder + "/test.conllu"
                parsemetsv_test = rootdir + "/" + lang_folder + "/test.parsemetsv"
                word_seq_out = os.path.splitext(conllu_train)[0]+ '_test_' + lang_folder + '_word.seq'
                mwe_seq_out = os.path.splitext(conllu_train)[0]+ '_test_' + lang_folder + '_mwe.seq'
                print word_seq_out
                file2 = open(word_seq_out, 'w')
                file3 = open(mwe_seq_out, 'w')                
                read_blocks(conllu_train, parsemetsv_train, file2, file3)
                #file2_len = file_len(file2)
                #print file2_len
                #file2.write("<doc id=\"test2016\">\n")
                #read_blocks(conllu_test, parsemetsv_test, file2)
                #file2.write("</doc>")
                file2.close
                file3.close
                #file2_len = file_len(file2)

if __name__ == "__main__":
    
    main()
    
