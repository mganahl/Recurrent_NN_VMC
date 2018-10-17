import tensorflow as tf
import numpy as np
import os

class RNNwavefunction(object):
   def __init__(self,systemsize,cell=tf.contrib.rnn.LSTMCell,activation=tf.nn.relu,units=[10],scope='RNNwavefunction',homogeneous=True):
       """
           systemsize:  int
                        number of sites
           cell:        a tensorflow RNN cell
           units:       list of int
                        number of units per RNN layer           
           scope:       str
                        the name of the name-space scope
           homogeneous: bool
                        True: use the same RNN cell at each
                        False: use a different RNN cell at each site
       """
       self.graph=tf.Graph()
       self.scope=scope
       self.N=systemsize
       self.homogeneous=homogeneous
       with self.graph.as_default():
           with tf.variable_scope(self.scope,reuse=tf.AUTO_REUSE):
               if homogeneous:
                   self.lstm=[tf.nn.rnn_cell.MultiRNNCell([cell(units[n],activation=activation,name='LSTM_{0}'.format(n)) for n in range(len(units))])]
               else:
                   self.lstm=[tf.nn.rnn_cell.MultiRNNCell([cell(units[n],activation=activation,name='LSTM_{0}'.format(n)) for n in range(len(units))])]*self.N
   
   def sample(self,numsamples,inputdim):
       """
           generate samples from a probability distribution parametrized by a recurrent network
           ------------------------------------------------------------------------
           Parameters: 
           
           numsamples:      int
                            number of samples to be produced
           inputdim:        int
                            hilbert space dimension
   
           ------------------------------------------------------------------------
           Returns:         a tuple (samples,log-probs)
            
           samples:         tf.Tensor of shape (numsamples,systemsize)
                            the samples in integer encoding
           log-probs        tf.Tensor of shape (numsamples,)
                            the log-probability of each sample 
       """
       with self.graph.as_default():
           with tf.variable_scope(self.scope,reuse=tf.AUTO_REUSE):
               b=np.zeros((numsamples,inputdim)).astype(np.float32)
               b[:,0]=np.ones(numsamples)
               inputs=tf.constant(dtype=tf.float32,value=b,shape=[numsamples,inputdim])
               self.inputdim=inputs.shape[1]
               self.outputdim=self.inputdim
               self.numsamples=inputs.shape[0]
               samples=[]
               one_hot_samples=[]
               probs=[]
               lstm_state=self.lstm[0].zero_state(inputs.shape[0],dtype=tf.float32)
               if not self.homogeneous:
                   lstm_output, lstm_state = self.lstm[0](inputs, lstm_state)
                   output=tf.layers.dense(lstm_output,self.outputdim,activation=tf.nn.softmax,name='wf_dense_{0}'.format(0))
                   probs.append(output)
                   temp=tf.reshape(tf.multinomial(tf.log(output),num_samples=1),[-1,])
                   samples.append(temp)
                   inputs2=tf.one_hot(temp,depth=self.outputdim)
                   one_hot_samples.append(inputs2)

                   for n in range(1,self.N):
                       lstm_output, lstm_state = self.lstm[n](inputs2, lstm_state)
                       output=tf.layers.dense(lstm_output,self.outputdim,activation=tf.nn.softmax,name='wf_dense_{0}'.format(n))
                       probs.append(output)
                       temp=tf.reshape(tf.multinomial(tf.log(output),num_samples=1),[-1,])
                       samples.append(temp)
                       inputs2=tf.one_hot(temp,depth=self.outputdim)
                       one_hot_samples.append(inputs2)


               else:
                   lstm_output, lstm_state = self.lstm[0](inputs, lstm_state)
                   output=tf.layers.dense(lstm_output,self.outputdim,activation=tf.nn.softmax,name='wf_dense_{0}'.format(0))
                   probs.append(output)
                   temp=tf.reshape(tf.multinomial(tf.log(output),num_samples=1),[-1,])
                   samples.append(temp)
                   inputs2=tf.one_hot(temp,depth=self.inputdim)
                   one_hot_samples.append(inputs2)

                   for n in range(1,self.N):
                       lstm_output, lstm_state = self.lstm[0](inputs2, lstm_state)
                       output=tf.layers.dense(lstm_output,self.outputdim,activation=tf.nn.softmax,name='wf_dense_{0}'.format(n))
                       probs.append(output)
                       temp=tf.reshape(tf.multinomial(tf.log(output),num_samples=1),[-1,])
                       samples.append(temp)
                       inputs2=tf.one_hot(temp,depth=self.outputdim)
                       one_hot_samples.append(inputs2)

       self.samples=tf.stack(values=samples,axis=1)
       one_hot_samples=tf.transpose(tf.stack(values=one_hot_samples,axis=2),perm=[0,2,1])
       temp=tf.transpose(tf.stack(values=probs,axis=2),perm=[0,2,1])
       #mask=tf.greater(one_hot_samples,0.0001)
       #zeros = tf.zeros_like(temp)
       #self.log_probs=tf.reduce_sum(tf.log(tf.reduce_sum(tf.where(mask,temp,zeros),axis=2)),axis=1)
       self.log_probs=tf.reduce_sum(tf.log(tf.reduce_sum(tf.multiply(temp,one_hot_samples),axis=2)),axis=1)
       return self.samples,self.log_probs
   
   def probability(self,samples,inputdim):
       """
           calculate the log-probabilities of ```samples``
           ------------------------------------------------------------------------
           Parameters: 
           
           samples:         tf.Tensor
                            a tf.placeholder of shape (number of samples,system-size) 
                            containing the input samples in integer encoding
           inputdim:        int
                            dimension of the input space
                     
           ------------------------------------------------------------------------         
           Returns: 
           log-probs        tf.Tensor of shape (number of samples,)
                            the log-probability of each sample         
           """
       with self.graph.as_default():

           self.inputdim=inputdim
           self.outputdim=self.inputdim

           self.numsamples=samples.shape[0]
           b=np.zeros((self.numsamples,self.inputdim)).astype(np.float32)
           b[:,0]=np.ones(self.numsamples)
           inputs=tf.constant(dtype=tf.float32,value=b,shape=[self.numsamples,self.inputdim])
       
           with tf.variable_scope(self.scope,reuse=tf.AUTO_REUSE):
               probs=[]

               lstm_state=self.lstm[0].zero_state(self.numsamples,dtype=tf.float32)
               if not self.homogeneous:
                   lstm_output, lstm_state = self.lstm[0](inputs, lstm_state)
                   output=tf.layers.dense(lstm_output,self.outputdim,activation=tf.nn.softmax,name='wf_dense_{0}'.format(0))
                   probs.append(output)
                   inputs2=tf.reshape(tf.one_hot(tf.slice(samples,begin=[np.int32(0),np.int32(0)],size=[np.int32(-1),np.int32(1)]),depth=self.outputdim),shape=[self.numsamples,self.inputdim])
                   
                   for n in range(1,self.N):
                       lstm_output, lstm_state = self.lstm[n](inputs2, lstm_state)
                       output=tf.layers.dense(lstm_output,self.outputdim,activation=tf.nn.softmax,name='wf_dense_{0}'.format(n))
                       probs.append(output)
                       inputs2=tf.reshape(tf.one_hot(tf.reshape(tf.slice(samples,begin=[np.int32(0),np.int32(n)],size=[np.int32(-1),np.int32(1)]),shape=[self.numsamples]),depth=self.outputdim),shape=[self.numsamples,self.inputdim])
           
               else:
                   lstm_output, lstm_state = self.lstm[0](inputs, lstm_state)
                   output=tf.layers.dense(lstm_output,self.outputdim,activation=tf.nn.softmax,name='wf_dense_{0}'.format(0))
                   probs.append(output)
                   inputs2=tf.reshape(tf.one_hot(tf.slice(samples,begin=[np.int32(0),np.int32(0)],size=[np.int32(-1),np.int32(1)]),depth=self.outputdim),shape=[self.numsamples,self.inputdim])
                   
                   for n in range(1,self.N):
                       lstm_output, lstm_state = self.lstm[0](inputs2, lstm_state)
                       output=tf.layers.dense(lstm_output,self.outputdim,activation=tf.nn.softmax,name='wf_dense_{0}'.format(n))
                       probs.append(output)
                       inputs2=tf.reshape(tf.one_hot(tf.reshape(tf.slice(samples,begin=[np.int32(0),np.int32(n)],size=[np.int32(-1),np.int32(1)]),shape=[self.numsamples]),depth=self.outputdim),shape=[self.numsamples,self.inputdim])
           
           temp=tf.transpose(tf.stack(values=probs,axis=2),perm=[0,2,1])
           one_hot_samples=tf.one_hot(samples,depth=self.inputdim)
           #mask=tf.greater(one_hot_samples,0.001)
           #zeros = tf.zeros_like(temp)
           #self.log_probs=tf.reduce_sum(tf.log(tf.reduce_sum(tf.where(mask,temp,zeros),axis=2)),axis=1)
           self.log_probs=tf.reduce_sum(tf.log(tf.reduce_sum(tf.multiply(temp,one_hot_samples),axis=2)),axis=1)            

           return self.log_probs





def XXZMatrixElemets(Jz,Jp,Bz,sigmap):
    """
    computes the matrix element of the periodic XXZ Hamiltonian for a given state sigmap
    -----------------------------------------------------------------------------------
    Parameters:
    Jz, Jp, Bz: np.ndarray of shape (N), (N) and (N), respectively, and dtype=float:
                XXZ parameters
    sigmap:     np.ndarrray of dtype=int and shape (N)
                spin-state, integer encoded (using 0 for down spin and 1 for up spin)
    -----------------------------------------------------------------------------------            
    Returns: 2-tuple of type (np.ndarray,np.ndarray)
             sigmas:         np.ndarray of dtype=int and shape (?,N)
                             the states for which there exist non-zero matrix elements for given sigmap
             matrixelements: np.ndarray of dtype=float and shape (?)
                             the non-zero matrix elements
    """
    sigmas=[]
    matrixelements=[]
    N=len(Bz)
    #the diagonal part is simply the sum of all Sz-Sz interactions plus a B field
    diag=np.dot(sigmap-0.5,Bz)
    
    for site in range(N):
        if sigmap[site]!=sigmap[(site+1)%N]:
            diag-=0.25*Jz[site]
        else:
            diag+=0.25*Jz[site]
    matrixelements.append(diag)
    sigmas.append(np.copy(sigmap))
    
    #off-diagonal part:
    for site in range(N):
        if sigmap[site]!=sigmap[(site+1)%N]:
            sig=np.copy(sigmap)
            sig[site]=sig[(site+1)%N]
            sig[(site+1)%N]=sigmap[site]
            sigmas.append(sig)
            matrixelements.append(Jp[site]/2)
    return np.array(sigmas),np.array(matrixelements)

def XXZLocalEnergy(Jz,Jp,Bz,sigmap,RNN):
    """
    DEPRECATED
    computes the local energy for the XXZ model:
    ---------------------------------------------------------------------------------
    Parameters:
    Jz, Jp, Bz: np.ndarray of shape (N-1), (N-1) and (N), respectively, and dtype=float:
                XXZ parameters
    sigmap:     np.ndarrray of dtype=int and shape (N)
                spin-state, integer encoded (using 0 for down spin and 1 for up spin)
    RNN:        fully initialized RNNwavefunction object
    ----------------------------------------------------------------------------------
    Returns:
    the local energy (float) for sigmapp
    """
    sigmas,H=XXZMatrixElemets(Jz,Jp,Bz,sigmap)#note that sigmas[0,:]==sigmap
    with RNN.graph.as_default():
        inputs=tf.placeholder(dtype=tf.int32,shape=[len(sigmas),len(Bz)])
        probs=RNN.probability(inputs,inputdim=2)
        probabilities=sess.run(probs,feed_dict={inputs:sigmas})
    
    return H.dot(probabilities)/probabilities[0]

def XXZLocalEnergies(Jz,Jp,Bz,sigmasp,RNN):
    """
    computes the local energies for the periodic XXZ model for a given spin-state sample sigmasp:
    Eloc(\sigma')=\sum_{sigma} H_{\sigma'\sigma}\psi_{\sigma}/\psi_{\sigma'}
    ----------------------------------------------------------------------------
    Parameters:
    Jz, Jp, Bz: np.ndarray of shape (N), (N) and (N), respectively, and dtype=float:
                XXZ parameters
    sigmasp:    np.ndarrray of dtype=int and shape (numsamples,N)
                spin-states, integer encoded (using 0 for down spin and 1 for up spin)
    RNN:        fully initialized RNNwavefunction object
    ----------------------------------------------------------------------------
    Returns:
    np.ndarray of shape (numsamples)and dtype=float containing the local energies for each samples
    """
    sigmas=np.empty((0,len(Bz)))
    H=np.empty(0)
    slices=[]
    for n in range(sigmasp.shape[0]):
        sigmap=sigmasp[n,:]
        temp1,temp2=XXZMatrixElemets(Jz,Jp,Bz,sigmap)#note that sigmas[0,:]==sigmap
        H=np.append(H,temp2)
        slices.append(slice(sigmas.shape[0],sigmas.shape[0]+temp1.shape[0]))
        sigmas=np.append(sigmas,temp1,axis=0)
    with RNN.graph.as_default():
        temp_inputs=tf.placeholder(dtype=tf.int32,shape=[len(sigmas),len(Bz)])
        temp_probs=RNN.probability(temp_inputs,inputdim=2)
        log_probabilities=sess.run(temp_probs,feed_dict={temp_inputs:sigmas})
    localEnergies=[]
    for n in range(len(slices)):
        s=slices[n]
        localEnergies.append(H[s].dot(np.exp(0.5*(log_probabilities[s]-log_probabilities[s][0]))))
    return np.array(localEnergies)

       
if __name__ == "__main__":
   load=False
   units=[200,200]#list containing the number of hidden units for each layer of the networks
   N=8
   input_dim=2
   numsamples=20 #only for initialization; later I'll use a much larger value (see below)
   #cell=tf.contrib.rnn.LSTMCell()
   wf=RNNwavefunction(N,units=units,cell=tf.contrib.rnn.LSTMCell) #contains the graph with the RNNs
   sampling=wf.sample(numsamples,input_dim) #call this function once to create the dense layers
   with wf.graph.as_default(): #now initialize everything 
       inputs=tf.placeholder(dtype=tf.int32,shape=[numsamples,N])
       learningrate=tf.placeholder(dtype=tf.float32,shape=[])
       probs=wf.probability(inputs,input_dim)
       optimizer=tf.train.AdamOptimizer(learning_rate=learningrate)
       init=tf.global_variables_initializer()
   sess=tf.Session(graph=wf.graph)
   sess.run(init)

   if load:
      path=os.getcwd()
      print((path))
      ending='units'
      for u in units:
          ending+='_{0}'.format(u)
      savename='RNNwavefunction_N{0}_'+ending          
      filename=savename+'.ckpt'

      with tf.variable_scope(wf.scope,reuse=tf.AUTO_REUSE):
          with wf.graph.as_default():
              saver=tf.train.Saver()
              saver.restore(sess,path+'/'+filename)
      meanEnergy=np.load('meanEnergy'+savename+'.npy')
      varEnergy=np.load('varEnergy'+savename+'.npy')
      #meanEnergy=[]
      #varEnergy=[]
      
   else:
      meanEnergy=[]
      varEnergy=[]
   
   path=os.getcwd()
   Jz=np.ones(N)
   Jp=-np.ones(N)
   Bz=np.zeros(N)
   
   #for a given network, generate a large number of samples:
   #numsamples_=[1000,5000,10000,20000]
   numsamples=20000
   lr=np.float32(0.001)
   ending='units'
   for u in units:
       ending+='_{0}'.format(u)
   filename='RNNwavefunction_N{0}_'+ending+'.ckpt'
   with tf.variable_scope(wf.scope,reuse=tf.AUTO_REUSE):
       with wf.graph.as_default():
           Eloc=tf.placeholder(dtype=tf.float32,shape=[numsamples])
           samp=tf.placeholder(dtype=tf.int32,shape=[numsamples,N])
           log_probs_=wf.probability(samp,inputdim=2)
           #now calculate the fake cost function:
           cost=tf.reduce_mean(tf.multiply(log_probs_,Eloc)) #factor of 2 in the above equation 
                                             #cancels when taking log(sqrt(prob))=log(sqrt(psi^2))`
                                             #=log(psi)=2*log(psi^2)->log(psi)=1/2*log(psi^2)=1/2*log_probs
           gradients, variables = zip(*optimizer.compute_gradients(cost))
           #clipped_gradients,_=tf.clip_by_global_norm(gradients,1.0)
           optstep=optimizer.apply_gradients(zip(gradients,variables))
           sess.run(tf.variables_initializer(optimizer.variables()),feed_dict={learningrate: lr})
   for it in range(10000):
   #     if it<10:
   #         numsamples=numsamples_[0]
   #     elif it<30:
   #         numsamples=numsamples_[1]
   #     elif it<50:
   #         numsamples=numsamples_[2]
   #     elif it<70:
   #         numsamples=numsamples_[3]        
       samples,log_probs=sess.run(wf.sample(numsamples=numsamples,inputdim=2))
       local_energies=XXZLocalEnergies(Jz,Jp,Bz,samples,wf)
       meanE=np.mean(local_energies)
       meanEnergy.append(meanE)
       
       varE=np.var(local_energies)
       varEnergy.append(varE)
       print('mean(E): {0} \pm {1}, #samples {2}'.format(meanE,np.sqrt(varE),numsamples))
       with tf.variable_scope(wf.scope,reuse=tf.AUTO_REUSE):
           with wf.graph.as_default():  
               sess.run(optstep,feed_dict={Eloc:local_energies,samp:samples,learningrate: lr})
               if it%10==0:
                   saver=tf.train.Saver()
                   saver.save(sess,path+'/'+filename)
                   np.save('meanEnergy'+savename,meanEnergy)
                   np.save('varEnergy'+savename,varEnergy)                   



