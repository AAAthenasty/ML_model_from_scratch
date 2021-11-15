

for i in 10 100 1000 10000
do
  python learnhmm.py ${i} /Users/tianye/Desktop/CMU/10601_ML/hw7/handout/en_data/train.txt /Users/tianye/Desktop/CMU/10601_ML/hw7/handout/en_data/index_to_word.txt /Users/tianye/Desktop/CMU/10601_ML/hw7/handout/en_data/index_to_tag.txt /Users/tianye/Desktop/CMU/10601_ML/hw7/handout/en_data/hmminit.txt /Users/tianye/Desktop/CMU/10601_ML/hw7/handout/en_data/hmmemit.txt /Users/tianye/Desktop/CMU/10601_ML/hw7/handout/en_data/hmmtrans.txt
  python forwardbackward.py ${i} /Users/tianye/Desktop/CMU/10601_ML/hw7/handout/en_data/train.txt /Users/tianye/Desktop/CMU/10601_ML/hw7/handout/en_data/index_to_word.txt /Users/tianye/Desktop/CMU/10601_ML/hw7/handout/en_data/index_to_tag.txt /Users/tianye/Desktop/CMU/10601_ML/hw7/handout/en_data/hmminit.txt /Users/tianye/Desktop/CMU/10601_ML/hw7/handout/en_data/hmmemit.txt /Users/tianye/Desktop/CMU/10601_ML/hw7/handout/en_data/hmmtrans.txt /Users/tianye/Desktop/CMU/10601_ML/hw7/handout/en_data/predict.txt /Users/tianye/Desktop/CMU/10601_ML/hw7/handout/en_data/metric_${i}.txt
done
