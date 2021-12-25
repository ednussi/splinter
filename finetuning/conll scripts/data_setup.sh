#!/bin/bash

dlx ( )  { 
  tar -xvzf $1 
  rm  $1 
}

#conll_url=http://conll.cemantix.org/2012/download
dlx conll-2012-train.v4.tar.gz
dlx conll-2012-development.v4.tar.gz
dlx conll-2012-test-key.tar.gz
dlx conll-2012-test-official.v9.tar.gz

dlx conll-2012-scripts.v3.tar.gz

dlx reference-coreference-scorers.v8.01.tar.gz
mv reference-coreference-scorers conll-2012/scorer

ontonotes_path =/Users/xuxiu/Downloads/e2e-coref-master/ontonotes-release-5.0
 bash conll-2012/v3/scripts/skeleton2conll.sh -D $ontonotes_path/data/files/data conll-2012

#----------------------The above process is the format conversion operation-------------------

#===========The following process is to format and preprocess the converted file==========


#Copy the final converted file out function compile_partition ( )  { 
    rm -f $2 . $5 . $3 $4 
    cat conll-2012/$3/data/$1/data/$5/annotations/*/*/*/*. $3 $4  > >  $2 . $5 . $3 $4 
}

function compile_language ( )  { 
    compile_partition development dev v4 _gold_conll $1 
    compile_partition train train v4 _gold_conll $1 
    compile_partition test  test v4 _gold_conll $1 
}

compile_language english #Copy the converted files from the English file. 
#compile_language chinese#If you don’t need to convert Chinese corpus, just comment out 
#compile_language arabic#If you don’t need to convert Arabic corpus, just comment out

python minimize.py 
#Convert to easy-to-understand and easy-to-read json file format python get_char_vocab.py #Get vocabulary dictionary (charater unit), char_vocab.english.txt

#python filter_embeddings.py glove.840B.300d.txt train.english.jsonlines dev.english.jsonlines 
#python cache_elmo.py train.english.jsonlines dev.english.jsonlines