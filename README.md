# lstmcrf
Name Entity Recogiration and Sentiment Analysis (Open Tagert ...) with blstm + crf model  
##pipeline model 
model_lstm.py: blstm + crf + dense feature  
pipe_dense_ner_train.py: train file for ner  
pipe_dense_sa_train.py: train file for sa  

model_lstm2.py: blstm + dense feature  
lstm_train.py: train file  

model_combine.py: blstm + crf + dense feature + sparse feature  
pipe_combine_ner_train.py: train file for ner  
pipe_combine_sa_train.py: train file for sa  

##joint model
model_joint_dense.py: blstm + crf + dense feature  
joint_dense_train.py: train file for ner and sa  

model_joint_combine.py: blstm + crf + dense feature + sparse feature  
joint_combine_train.py: train file for ner and sa  

model_joint_sparse.py: crf + sparse feature (not implementation)  

##crf model
model_crf.py: crf + sparse feature  
crf_train.py: train file  

##data prepare
data.py: read vocab, train and test data
