#/bin/bash

DATA=sharedtask-data
#langs=FA
langs="BG CS DE EL ES FR HE HU IT LT MT PL PT RO SL SV TR"
annot="conllu parseme"
for lang in $langs
do
    rm $DATA/$lang/rmv*
    rm $DATA/$lang/remove_*
    for ann in $annot
    	do
    	echo "$DATA/$lang annotation: $annot  pasting this: $DATA/$lang/paste1_conllu_train_$lang"
   	paste $DATA/$lang/paste1_conllu_train_$lang $DATA/$lang/paste2_parseme_train_$lang > $DATA/$lang/train_ALL
        paste $DATA/$lang/paste1_conllu_test_$lang $DATA/$lang/paste2_parseme_test_$lang > $DATA/$lang/blindtest_ALL_$lang
########splitting into dev test train
    	done
	./split_conll.pl < $DATA/$lang/train_ALL -phead 10 $DATA/$lang/rmvdev_$lang $DATA/$lang/remove_rest
	./split_conll.pl < $DATA/$lang/remove_rest -phead 90 $DATA/$lang/rmvtrain_$lang $DATA/$lang/rmvtest_$lang

#rm $DATA/$lang/remove_*
########train, dev, test without IDS (easier to train), set aside IDS for test for evaluate.py
cat $DATA/$lang/rmvtrain_$lang | cut -f2,3,4,5 > $DATA/$lang/train_$lang
cat $DATA/$lang/rmvdev_$lang | cut -f2,3,4,5 > $DATA/$lang/dev_$lang
cat $DATA/$lang/rmvtest_$lang | cut -f2,3,4,5 > $DATA/$lang/test_$lang
cat $DATA/$lang/rmvtest_$lang | cut -f1 > $DATA/$lang/IDS_$lang
cat $DATA/$lang/blindtest_ALL_$lang | cut -f2,3,4,5 > $DATA/$lang/blindtest_$lang
cat $DATA/$lang/blindtest_ALL_$lang | cut -f1 > $DATA/$lang/IDS_blindtest_$lang


rm $DATA/$lang/rmv*
done
#echo "training for $DATA/$lang/train_$lang"
#python mwe_tagger.py --data_train='data_mwe/$lang/train_IT' --data_dev='data_mwe/$lang/dev_$lang' --data_test='$DATA/$lang/test_$lang'
#for dir in CS DE EL ES FR HE HU IT LT MT PL PT RO SL SV TR; do python mwe_tagger.py --data_train="data_mwe/$dir/train_$dir" --data_dev="data_mwe/$dir/dev_$dir" --data_test="data_mwe/$dir/test_$dir" ; done
#cut -f1,2,10 ../CS.dev | perl -ple '@parts=split/\t/;if ($parts[2] =~ /^[0-9]:([^;]*)/) {$parts[2] = $1;} $parts[2]="CONT" if       $parts[2] =~ /^\d+$/; $_="$parts[0]\t$parts[1]\t_\t$parts[2]\t_\t_\t_\t_\t_\t_"' >mwe.dev

#for dir in *; do paste $dir/train.conllu $dir/train.parsemetsv > $dir/train_all ; cat $dir/train_all | cut -f1,2,3,4,14 >  $dir/     train_$dir  ; done

#for dir in *; do ./../split_conll.pl < $dir/train_$dir -phead 10 $dir/dev-$dir $dir/rest_$dir; ./../split_conll.pl <       $dir/          rest_$dir -phead 90 $dir/train_$dir $dir/test_$dir ; done

