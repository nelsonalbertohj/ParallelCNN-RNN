{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Important Details about EEG data:\n",
    "Sampple Frequency = 200 Hz\n",
    "\n",
    "Number of Samples = 621892\n",
    "\n",
    "Duration of trial = 0.85 seconds after onset, which is indicated by firs non-zero marker value from marker data."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Signal Extraction and Feature Engineering"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [],
   "source": [
    "import scipy.io\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import mne\n",
    "import tensorflow as tf\n",
    "\n",
    "from sklearn.metrics import confusion_matrix, plot_confusion_matrix\n",
    "import pandas as pd\n",
    "import seaborn as sn\n",
    "from sklearn.model_selection import train_test_split\n",
    "sn.set_theme()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {
    "scrolled": true
   },
   "outputs": [],
   "source": [
    "def mat_loader(file,var_list=['o'],def_inputs = False):\n",
    "    '''\n",
    "    file: string, path to file.\n",
    "    var: list of string, name of element(s) in .mat \n",
    "    file to be extracted.\n",
    "    return: dict holding one or more elements from specified var\n",
    "    \n",
    "    required libraries: scipy.io\n",
    "    '''\n",
    "    print('.mat file has the following elements',scipy.io.whosmat(file))\n",
    "    mat_to_dict = {}\n",
    "    for var in var_list:\n",
    "        if def_inputs:\n",
    "            print('Choose from first element from tuples in this list', \n",
    "                  scipy.io.whosmat(file))\n",
    "            var = input('Input string label from above')\n",
    "        mat_load = scipy.io.loadmat('CLA-SubjectJ-170508-3St-LRHand-Inter')\n",
    "        keys = [e for e in dict(mat_load[var].dtype.fields).keys()]\n",
    "        print('Structure had these keys:', keys)\n",
    "        for i in range(len(keys)):\n",
    "            mat_to_dict[keys[i]] = mat_load[var][0][0][i]\n",
    "    return mat_to_dict"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      ".mat file has the following elements [('ans', (1, 1), 'double'), ('o', (1, 1), 'struct'), ('x', (533, 1), 'double')]\n",
      "Structure had these keys: ['id', 'tag', 'nS', 'sampFreq', 'marker', 'marker_old', 'data', 'chnames', 'binsuV']\n"
     ]
    }
   ],
   "source": [
    "mat_dict = mat_loader('CLA-SubjectJ-170508-3St-LRHand-Inter',['o'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {},
   "outputs": [],
   "source": [
    "def eeg_data_loader(mat_loader,eeg_data='data',marker='marker',chan_names='chnames'):\n",
    "    '''\n",
    "    Takes in mat_loader, a dictionary with keys that map to eeg data information,\n",
    "    and extracts data from the given eeg_data, marker, chan_names keys:\n",
    "    \n",
    "    mat_loader: dict, mapping labels to its respective eeg data\n",
    "    eeg_data: string, dictionary key\n",
    "    marker: string, dictionary key\n",
    "    chan_names: string, dictionary key\n",
    "    \n",
    "    return: tuple with 3 elements, (raw_eeg_unfiltered, marker, chan_names)\n",
    "    '''\n",
    "    #Extract eeg data from .mat files\n",
    "    raw_eeg_unfiltered = mat_loader[eeg_data].transpose()[:21,:]\n",
    "    #^only takes the first 21 channels which are the actual EEG channels\n",
    "    marker = mat_loader[marker]\n",
    "    chan_names = mat_loader[chan_names]\n",
    "\n",
    "    #format chan_names as a list to be used in that format for RawEEG function\n",
    "    chan_names = [st[0][0] for st in chan_names]\n",
    "    chan_names.append('STI 1')\n",
    "    \n",
    "    return (raw_eeg_unfiltered,marker,chan_names)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {
    "scrolled": true
   },
   "outputs": [],
   "source": [
    "(raw_eeg_unfiltered,marker,chan_names) = eeg_data_loader(mat_dict)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {},
   "outputs": [],
   "source": [
    "def preprocessing(raw_eeg_unfiltered,marker,samp_freq=200,l_fq=1,h_fq=40):\n",
    "    #Bandpass filter eeg data from 1 Hz to 40 Hz\n",
    "    raw_eeg = mne.filter.filter_data(raw_eeg_unfiltered, samp_freq, l_fq, h_fq)\n",
    "    \n",
    "    #Make marker last EEG label\n",
    "    raw_eeg = np.concatenate((raw_eeg, marker.flatten().reshape(1,-1)),axis=0)\n",
    "    \n",
    "    #Generate RawArray object to facilitate EEG analysis with mne\n",
    "    eeg_raw = mne.io.RawArray(raw_eeg,info= mne.create_info(chan_names,samp_freq))\n",
    "    print(eeg_raw.info)\n",
    "    \n",
    "    #Detect events in the data, containing \n",
    "    eeg_events = mne.find_events(eeg_raw,stim_channel='STI 1')\n",
    "    #^ rows= n_events columns = (time,nothing,event_ID)\n",
    "    \n",
    "    #Extract epochs from raw EEG\n",
    "    eeg_epochs = mne.Epochs(eeg_raw,eeg_events,\n",
    "                        event_id=[1,2,3],\n",
    "                        tmin=-0.2,tmax=0.8)\n",
    "    \n",
    "    return (eeg_raw, eeg_events, eeg_epochs)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Setting up band-pass filter from 1 - 40 Hz\n",
      "\n",
      "FIR filter parameters\n",
      "---------------------\n",
      "Designing a one-pass, zero-phase, non-causal bandpass filter:\n",
      "- Windowed time-domain design (firwin) method\n",
      "- Hamming window with 0.0194 passband ripple and 53 dB stopband attenuation\n",
      "- Lower passband edge: 1.00\n",
      "- Lower transition bandwidth: 1.00 Hz (-6 dB cutoff frequency: 0.50 Hz)\n",
      "- Upper passband edge: 40.00 Hz\n",
      "- Upper transition bandwidth: 10.00 Hz (-6 dB cutoff frequency: 45.00 Hz)\n",
      "- Filter length: 661 samples (3.305 sec)\n",
      "\n",
      "Creating RawArray with float64 data, n_channels=22, n_times=621892\n",
      "    Range : 0 ... 621891 =      0.000 ...  3109.455 secs\n",
      "Ready.\n",
      "<Info | 7 non-empty values\n",
      " bads: []\n",
      " ch_names: Fp1, Fp2, F3, F4, C3, C4, P3, P4, O1, O2, A1, A2, F7, F8, T3, ...\n",
      " chs: 22 MISC\n",
      " custom_ref_applied: False\n",
      " highpass: 0.0 Hz\n",
      " lowpass: 100.0 Hz\n",
      " meas_date: unspecified\n",
      " nchan: 22\n",
      " projs: []\n",
      " sfreq: 200.0 Hz\n",
      ">\n",
      "900 events found\n",
      "Event IDs: [1 2 3]\n",
      "Not setting metadata\n",
      "Not setting metadata\n",
      "900 matching events found\n",
      "Setting baseline interval to [-0.2, 0.0] sec\n",
      "Applying baseline correction (mode: mean)\n",
      "0 projection items activated\n"
     ]
    }
   ],
   "source": [
    "(eeg_raw, eeg_events, eeg_epochs) = preprocessing(raw_eeg_unfiltered,marker)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Loading data for 900 events and 201 original time points ...\n",
      "0 bad epochs dropped\n",
      "Loading data for 900 events and 201 original time points ...\n",
      "(180, 201, 21)\n",
      "(180, 4)\n"
     ]
    }
   ],
   "source": [
    "#Obtain epochs and labels for each epoch\n",
    "np_epochs = eeg_epochs.get_data(picks=eeg_raw.info.ch_names[:-1])\n",
    "\n",
    "df_epochs = eeg_epochs.to_data_frame(picks='all')\n",
    "labels = []\n",
    "for i in df_epochs['epoch'].unique():\n",
    "    labels.append(np.array(df_epochs['condition'][df_epochs['epoch'] == i])[0])\n",
    "\n",
    "X_train, X_test, y_train, y_test = train_test_split(np_epochs, labels, test_size=0.20, random_state=42)\n",
    "y_true = y_test\n",
    "\n",
    "#Reshape X as (epochs,time steps,features) from original (epochs,features,time_setp)\n",
    "X_train = np.transpose(X_train, (0,2,1))\n",
    "X_test = np.transpose(X_test, (0,2,1))\n",
    "\n",
    "#One-hot-encode train labels\n",
    "y_train = tf.keras.utils.to_categorical(y_train) \n",
    "y_test = tf.keras.utils.to_categorical(y_test)\n",
    "\n",
    "print(X_test.shape)\n",
    "print(y_test.shape)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "metadata": {},
   "outputs": [],
   "source": [
    "class eegRNN(tf.keras.Model):\n",
    "    def __init__(self,n_timesteps,n_features,n_outputs):\n",
    "        super().__init__()\n",
    "        self.n_timesteps = n_timesteps\n",
    "        self.n_features = n_features\n",
    "        self.n_outputs =  n_outputs\n",
    "        \n",
    "        self.rnn1 = tf.keras.layers.LSTM(20, return_sequences=True,input_shape= \n",
    "                                         (self.n_timesteps,self.n_features))\n",
    "        self.rnn2 = tf.keras.layers.LSTM(20,return_sequences=True)\n",
    "        self.rnn3 = tf.keras.layers.LSTM(20)\n",
    "        self.rnn_fc = tf.keras.layers.Dense(self.n_outputs, activation='softmax')\n",
    "        \n",
    "    def call(self,inputs):\n",
    "        '''\n",
    "        inputs: np.array with shape [batch, timesteps, feature]\n",
    "        '''\n",
    "        l = self.rnn1(inputs)\n",
    "        l = self.rnn2(l)\n",
    "        l = self.rnn3(l)\n",
    "        l = self.rnn_fc(l)\n",
    "        return l    "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1/30\n",
      "15/15 [==============================] - 7s 116ms/step - loss: 1.3225 - accuracy: 0.3403\n",
      "Epoch 2/30\n",
      "15/15 [==============================] - 2s 111ms/step - loss: 1.1926 - accuracy: 0.4083\n",
      "Epoch 3/30\n",
      "15/15 [==============================] - 2s 110ms/step - loss: 1.0986 - accuracy: 0.4500\n",
      "Epoch 4/30\n",
      "15/15 [==============================] - 2s 110ms/step - loss: 1.0149 - accuracy: 0.5208\n",
      "Epoch 5/30\n",
      "15/15 [==============================] - 2s 111ms/step - loss: 0.8789 - accuracy: 0.6222\n",
      "Epoch 6/30\n",
      "15/15 [==============================] - 2s 110ms/step - loss: 0.7419 - accuracy: 0.7111\n",
      "Epoch 7/30\n",
      "15/15 [==============================] - 2s 112ms/step - loss: 0.6803 - accuracy: 0.7333\n",
      "Epoch 8/30\n",
      "15/15 [==============================] - 2s 109ms/step - loss: 0.5728 - accuracy: 0.7861\n",
      "Epoch 9/30\n",
      "15/15 [==============================] - 2s 115ms/step - loss: 0.5022 - accuracy: 0.8222\n",
      "Epoch 10/30\n",
      "15/15 [==============================] - 2s 109ms/step - loss: 0.4477 - accuracy: 0.8431\n",
      "Epoch 11/30\n",
      "15/15 [==============================] - 2s 108ms/step - loss: 0.3687 - accuracy: 0.8861\n",
      "Epoch 12/30\n",
      "15/15 [==============================] - 2s 109ms/step - loss: 0.3376 - accuracy: 0.8875\n",
      "Epoch 13/30\n",
      "15/15 [==============================] - 2s 107ms/step - loss: 0.2858 - accuracy: 0.9194\n",
      "Epoch 14/30\n",
      "15/15 [==============================] - 2s 110ms/step - loss: 0.2885 - accuracy: 0.9139\n",
      "Epoch 15/30\n",
      "15/15 [==============================] - 2s 115ms/step - loss: 0.2503 - accuracy: 0.9236\n",
      "Epoch 16/30\n",
      "15/15 [==============================] - 2s 125ms/step - loss: 0.2499 - accuracy: 0.9292\n",
      "Epoch 17/30\n",
      "15/15 [==============================] - 2s 114ms/step - loss: 0.1969 - accuracy: 0.9403\n",
      "Epoch 18/30\n",
      "15/15 [==============================] - 2s 114ms/step - loss: 0.2181 - accuracy: 0.9389\n",
      "Epoch 19/30\n",
      "15/15 [==============================] - 2s 116ms/step - loss: 0.2152 - accuracy: 0.9250\n",
      "Epoch 20/30\n",
      "15/15 [==============================] - 2s 113ms/step - loss: 0.1922 - accuracy: 0.9444\n",
      "Epoch 21/30\n",
      "15/15 [==============================] - 2s 120ms/step - loss: 0.1581 - accuracy: 0.9611\n",
      "Epoch 22/30\n",
      "15/15 [==============================] - 2s 113ms/step - loss: 0.1263 - accuracy: 0.9681\n",
      "Epoch 23/30\n",
      "15/15 [==============================] - 2s 115ms/step - loss: 0.1145 - accuracy: 0.9681\n",
      "Epoch 24/30\n",
      "15/15 [==============================] - 2s 118ms/step - loss: 0.1136 - accuracy: 0.9750\n",
      "Epoch 25/30\n",
      "15/15 [==============================] - 2s 119ms/step - loss: 0.0908 - accuracy: 0.9764\n",
      "Epoch 26/30\n",
      "15/15 [==============================] - 2s 111ms/step - loss: 0.0968 - accuracy: 0.9778\n",
      "Epoch 27/30\n",
      "15/15 [==============================] - 2s 112ms/step - loss: 0.0945 - accuracy: 0.9764\n",
      "Epoch 28/30\n",
      "15/15 [==============================] - 2s 110ms/step - loss: 0.1185 - accuracy: 0.9750\n",
      "Epoch 29/30\n",
      "15/15 [==============================] - 2s 111ms/step - loss: 0.1066 - accuracy: 0.9722\n",
      "Epoch 30/30\n",
      "15/15 [==============================] - 2s 110ms/step - loss: 0.0899 - accuracy: 0.9764\n"
     ]
    }
   ],
   "source": [
    "rnn_model = eegRNN(X_train.shape[1],X_train.shape[2],y_train.shape[1])\n",
    "rnn_model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])\n",
    "rnn_history = rnn_model.fit(X_train, y_train, epochs= 30, batch_size = 50)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "scrolled": false
   },
   "outputs": [],
   "source": [
    "rnn_model.save('RNN_model')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "metadata": {
    "scrolled": false
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[<matplotlib.lines.Line2D at 0x1d563780b20>]"
      ]
     },
     "execution_count": 34,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.plot(rnn_history.history['accuracy'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[<matplotlib.lines.Line2D at 0x1d5691c7f70>]"
      ]
     },
     "execution_count": 35,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.plot(rnn_history.history['loss'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "6/6 [==============================] - 1s 28ms/step - loss: 0.7091 - accuracy: 0.7889\n"
     ]
    }
   ],
   "source": [
    "accuracy = rnn_model.evaluate(X_test, y_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<AxesSubplot:>"
      ]
     },
     "execution_count": 37,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 720x504 with 2 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "predictions = rnn_model.predict(X_test)\n",
    "y_pred = [p.argmax() for p in predictions]\n",
    "y_true = [y.argmax() for y in y_test]\n",
    "conf_mat = confusion_matrix(y_true,y_pred)\n",
    "df_conf = pd.DataFrame(conf_mat,index=[i for i in 'Right Left Passive'.split()],\n",
    "                      columns = [i for i in 'Right Left Passive'.split()])\n",
    "plt.figure(figsize = (10,7))\n",
    "sn.heatmap(df_conf, annot=True,cmap='YlGnBu')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "metadata": {},
   "outputs": [],
   "source": [
    "class eegCNN(tf.keras.Model):\n",
    "    def __init__(self,n_timesteps,n_features,n_outputs):\n",
    "        super().__init__()\n",
    "        self.n_timesteps = n_timesteps\n",
    "        self.n_features = n_features\n",
    "        self.n_outputs =  n_outputs\n",
    "        \n",
    "        self.conv1 = tf.keras.layers.Conv1D(filters=50,kernel_size=3,activation='relu',\n",
    "                                           input_shape=[self.n_timesteps,self.n_features])\n",
    "        self.pool1 = tf.keras.layers.MaxPooling1D(pool_size=3)\n",
    "        self.conv2 = tf.keras.layers.Conv1D(filters=100,kernel_size=3,activation='relu')\n",
    "        self.pool2 = tf.keras.layers.MaxPooling1D(pool_size=3)\n",
    "        self.conv3 = tf.keras.layers.Conv1D(filters=200,kernel_size=3,activation='relu')\n",
    "        self.pool3 = tf.keras.layers.MaxPooling1D(pool_size=3)\n",
    "        self.flat = tf.keras.layers.Flatten()\n",
    "        self.fc = tf.keras.layers.Dense(self.n_outputs, activation='softmax')\n",
    "        \n",
    "    def call(self,inputs):\n",
    "        e = self.conv1(inputs)\n",
    "        e = self.pool1(e)\n",
    "        e = self.conv2(e)\n",
    "        e = self.pool2(e)\n",
    "        e = self.conv3(e)\n",
    "        e = self.pool3(e)\n",
    "        e = self.flat(e)\n",
    "        return self.fc(e)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1/20\n",
      "15/15 [==============================] - 1s 12ms/step - loss: 1.7616 - accuracy: 0.4722\n",
      "Epoch 2/20\n",
      "15/15 [==============================] - 0s 12ms/step - loss: 0.5151 - accuracy: 0.7931\n",
      "Epoch 3/20\n",
      "15/15 [==============================] - 0s 12ms/step - loss: 0.2805 - accuracy: 0.8875\n",
      "Epoch 4/20\n",
      "15/15 [==============================] - 0s 12ms/step - loss: 0.1755 - accuracy: 0.9417\n",
      "Epoch 5/20\n",
      "15/15 [==============================] - 0s 13ms/step - loss: 0.1224 - accuracy: 0.9667\n",
      "Epoch 6/20\n",
      "15/15 [==============================] - 0s 12ms/step - loss: 0.0938 - accuracy: 0.9833\n",
      "Epoch 7/20\n",
      "15/15 [==============================] - 0s 12ms/step - loss: 0.0607 - accuracy: 0.9917\n",
      "Epoch 8/20\n",
      "15/15 [==============================] - 0s 11ms/step - loss: 0.0450 - accuracy: 0.9972\n",
      "Epoch 9/20\n",
      "15/15 [==============================] - 0s 12ms/step - loss: 0.0313 - accuracy: 0.9986\n",
      "Epoch 10/20\n",
      "15/15 [==============================] - 0s 12ms/step - loss: 0.0210 - accuracy: 1.0000\n",
      "Epoch 11/20\n",
      "15/15 [==============================] - 0s 12ms/step - loss: 0.0177 - accuracy: 1.0000\n",
      "Epoch 12/20\n",
      "15/15 [==============================] - 0s 12ms/step - loss: 0.0119 - accuracy: 1.0000\n",
      "Epoch 13/20\n",
      "15/15 [==============================] - 0s 12ms/step - loss: 0.0098 - accuracy: 1.0000\n",
      "Epoch 14/20\n",
      "15/15 [==============================] - 0s 12ms/step - loss: 0.0084 - accuracy: 1.0000\n",
      "Epoch 15/20\n",
      "15/15 [==============================] - 0s 12ms/step - loss: 0.0073 - accuracy: 1.0000\n",
      "Epoch 16/20\n",
      "15/15 [==============================] - 0s 12ms/step - loss: 0.0064 - accuracy: 1.0000\n",
      "Epoch 17/20\n",
      "15/15 [==============================] - 0s 12ms/step - loss: 0.0057 - accuracy: 1.0000\n",
      "Epoch 18/20\n",
      "15/15 [==============================] - 0s 12ms/step - loss: 0.0050 - accuracy: 1.0000\n",
      "Epoch 19/20\n",
      "15/15 [==============================] - 0s 12ms/step - loss: 0.0043 - accuracy: 1.0000\n",
      "Epoch 20/20\n",
      "15/15 [==============================] - 0s 14ms/step - loss: 0.0039 - accuracy: 1.0000\n"
     ]
    }
   ],
   "source": [
    "cnn_model = eegCNN(X_train.shape[1],X_train.shape[2],y_train.shape[1])\n",
    "cnn_model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])\n",
    "cnn_history = cnn_model.fit(X_train, y_train, epochs= 20, batch_size = 50)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "scrolled": false
   },
   "outputs": [],
   "source": [
    "#Saving model in Keras:\n",
    "cnn_model.save('CNN_model')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "metadata": {
    "scrolled": false
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[<matplotlib.lines.Line2D at 0x1d569360130>]"
      ]
     },
     "execution_count": 44,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.plot(cnn_history.history['accuracy'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 45,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[<matplotlib.lines.Line2D at 0x1d566655940>]"
      ]
     },
     "execution_count": 45,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.plot(cnn_history.history['loss'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 46,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "6/6 [==============================] - 0s 4ms/step - loss: 0.3930 - accuracy: 0.8889\n"
     ]
    }
   ],
   "source": [
    "accuracy = cnn_model.evaluate(X_test, y_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 48,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<AxesSubplot:>"
      ]
     },
     "execution_count": 48,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 720x504 with 2 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "predictions = cnn_model.predict(X_test)\n",
    "y_pred = [p.argmax() for p in predictions]\n",
    "y_true = [y.argmax() for y in y_test]\n",
    "conf_mat = confusion_matrix(y_true,y_pred)\n",
    "df_conf = pd.DataFrame(conf_mat,index=[i for i in 'Right Left Passive'.split()],\n",
    "                      columns = [i for i in 'Right Left Passive'.split()])\n",
    "plt.figure(figsize = (10,7))\n",
    "sn.heatmap(df_conf, annot=True,cmap='YlGnBu')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 49,
   "metadata": {},
   "outputs": [],
   "source": [
    "class eegConvRNN(tf.keras.Model):\n",
    "    def __init__(self,n_timesteps,n_features,n_outputs):\n",
    "        super().__init__(n_timesteps,n_features,n_outputs)\n",
    "        self.n_timesteps = n_timesteps\n",
    "        self.n_features = n_features\n",
    "        self.n_outputs =  n_outputs\n",
    "        \n",
    "        #RNN Network\n",
    "        self.rnn1 = tf.keras.layers.LSTM(20, return_sequences=True,input_shape= \n",
    "                                         (self.n_timesteps,self.n_features))\n",
    "        self.rnn2 = tf.keras.layers.LSTM(20,return_sequences=True)\n",
    "        self.rnn3 = tf.keras.layers.LSTM(20)\n",
    "        self.rnn_fc = tf.keras.layers.Dense(self.n_outputs, activation='softmax')\n",
    "        \n",
    "        #CNN Network\n",
    "        self.conv1 = tf.keras.layers.Conv1D(filters=50,kernel_size=3,activation='relu',\n",
    "                                           input_shape=[self.n_timesteps,self.n_features])\n",
    "        self.pool1 = tf.keras.layers.MaxPooling1D(pool_size=3)\n",
    "        self.conv2 = tf.keras.layers.Conv1D(filters=100,kernel_size=3,activation='relu')\n",
    "        self.pool2 = tf.keras.layers.MaxPooling1D(pool_size=3)\n",
    "        self.conv3 = tf.keras.layers.Conv1D(filters=200,kernel_size=3,activation='relu')\n",
    "        self.pool3 = tf.keras.layers.MaxPooling1D(pool_size=3)\n",
    "        self.flat = tf.keras.layers.Flatten()\n",
    "        self.cnn_fc = tf.keras.layers.Dense(self.n_outputs, activation='softmax')\n",
    "        \n",
    "        #Averaged Outuput Layer\n",
    "        self.avg = tf.keras.layers.Average()\n",
    "        self.fc_out = tf.keras.layers.Dense(self.n_outputs, activation='softmax')\n",
    "        \n",
    "    def call(self,inputs):\n",
    "        '''\n",
    "        inputs: np.array with shape [batch, timesteps, feature]\n",
    "        '''\n",
    "        #RNN operations\n",
    "        l = self.rnn1(inputs)\n",
    "        l = self.rnn2(l)\n",
    "        l = self.rnn3(l)\n",
    "        l = self.rnn_fc(l)\n",
    "        \n",
    "        #CNN operations\n",
    "        e = self.conv1(inputs)\n",
    "        e = self.pool1(e)\n",
    "        e = self.conv2(e)\n",
    "        e = self.pool2(e)\n",
    "        e = self.conv3(e)\n",
    "        e = self.pool3(e)\n",
    "        e = self.flat(e)\n",
    "        e = self.cnn_fc(e)\n",
    "        \n",
    "        #Averaged Output Layer\n",
    "        out = self.avg([l,e])\n",
    "        return self.fc_out(out)        "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 54,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1/100\n",
      "11/11 [==============================] - 8s 131ms/step - loss: 1.3674 - accuracy: 0.3569\n",
      "Epoch 2/100\n",
      "11/11 [==============================] - 1s 133ms/step - loss: 1.2983 - accuracy: 0.3181\n",
      "Epoch 3/100\n",
      "11/11 [==============================] - 1s 121ms/step - loss: 1.2737 - accuracy: 0.3181\n",
      "Epoch 4/100\n",
      "11/11 [==============================] - 1s 122ms/step - loss: 1.2576 - accuracy: 0.3181\n",
      "Epoch 5/100\n",
      "11/11 [==============================] - 1s 124ms/step - loss: 1.2422 - accuracy: 0.3181\n",
      "Epoch 6/100\n",
      "11/11 [==============================] - 1s 130ms/step - loss: 1.2219 - accuracy: 0.3181\n",
      "Epoch 7/100\n",
      "11/11 [==============================] - 1s 126ms/step - loss: 1.1980 - accuracy: 0.3181\n",
      "Epoch 8/100\n",
      "11/11 [==============================] - 1s 129ms/step - loss: 1.1759 - accuracy: 0.4167\n",
      "Epoch 9/100\n",
      "11/11 [==============================] - 1s 133ms/step - loss: 1.1636 - accuracy: 0.5111\n",
      "Epoch 10/100\n",
      "11/11 [==============================] - 1s 133ms/step - loss: 1.1489 - accuracy: 0.5236\n",
      "Epoch 11/100\n",
      "11/11 [==============================] - 1s 121ms/step - loss: 1.1369 - accuracy: 0.5431\n",
      "Epoch 12/100\n",
      "11/11 [==============================] - 1s 125ms/step - loss: 1.1290 - accuracy: 0.5486\n",
      "Epoch 13/100\n",
      "11/11 [==============================] - 1s 132ms/step - loss: 1.1202 - accuracy: 0.5444\n",
      "Epoch 14/100\n",
      "11/11 [==============================] - 1s 130ms/step - loss: 1.1082 - accuracy: 0.5611\n",
      "Epoch 15/100\n",
      "11/11 [==============================] - 1s 125ms/step - loss: 1.1008 - accuracy: 0.5722\n",
      "Epoch 16/100\n",
      "11/11 [==============================] - 1s 131ms/step - loss: 1.0949 - accuracy: 0.5722\n",
      "Epoch 17/100\n",
      "11/11 [==============================] - 1s 131ms/step - loss: 1.0908 - accuracy: 0.5667\n",
      "Epoch 18/100\n",
      "11/11 [==============================] - 1s 123ms/step - loss: 1.0867 - accuracy: 0.5653\n",
      "Epoch 19/100\n",
      "11/11 [==============================] - 1s 124ms/step - loss: 1.0751 - accuracy: 0.5750\n",
      "Epoch 20/100\n",
      "11/11 [==============================] - 1s 124ms/step - loss: 1.0683 - accuracy: 0.5722\n",
      "Epoch 21/100\n",
      "11/11 [==============================] - 1s 125ms/step - loss: 1.0590 - accuracy: 0.6472\n",
      "Epoch 22/100\n",
      "11/11 [==============================] - 1s 123ms/step - loss: 1.0596 - accuracy: 0.6375\n",
      "Epoch 23/100\n",
      "11/11 [==============================] - 1s 123ms/step - loss: 1.0654 - accuracy: 0.6194\n",
      "Epoch 24/100\n",
      "11/11 [==============================] - 1s 126ms/step - loss: 1.0535 - accuracy: 0.6417\n",
      "Epoch 25/100\n",
      "11/11 [==============================] - 1s 125ms/step - loss: 1.0442 - accuracy: 0.6542\n",
      "Epoch 26/100\n",
      "11/11 [==============================] - 1s 125ms/step - loss: 1.0375 - accuracy: 0.6556\n",
      "Epoch 27/100\n",
      "11/11 [==============================] - 1s 123ms/step - loss: 1.0305 - accuracy: 0.6611\n",
      "Epoch 28/100\n",
      "11/11 [==============================] - 1s 125ms/step - loss: 1.0236 - accuracy: 0.6611\n",
      "Epoch 29/100\n",
      "11/11 [==============================] - 1s 126ms/step - loss: 1.0171 - accuracy: 0.6722\n",
      "Epoch 30/100\n",
      "11/11 [==============================] - 1s 124ms/step - loss: 1.0191 - accuracy: 0.6681\n",
      "Epoch 31/100\n",
      "11/11 [==============================] - 1s 124ms/step - loss: 1.0083 - accuracy: 0.6847\n",
      "Epoch 32/100\n",
      "11/11 [==============================] - 1s 126ms/step - loss: 1.0034 - accuracy: 0.7014\n",
      "Epoch 33/100\n",
      "11/11 [==============================] - 1s 122ms/step - loss: 0.9976 - accuracy: 0.7181\n",
      "Epoch 34/100\n",
      "11/11 [==============================] - 1s 125ms/step - loss: 1.0112 - accuracy: 0.7000\n",
      "Epoch 35/100\n",
      "11/11 [==============================] - 2s 136ms/step - loss: 1.0152 - accuracy: 0.6694\n",
      "Epoch 36/100\n",
      "11/11 [==============================] - 1s 124ms/step - loss: 1.0001 - accuracy: 0.7167\n",
      "Epoch 37/100\n",
      "11/11 [==============================] - 1s 124ms/step - loss: 0.9823 - accuracy: 0.7417\n",
      "Epoch 38/100\n",
      "11/11 [==============================] - 1s 124ms/step - loss: 0.9621 - accuracy: 0.7611\n",
      "Epoch 39/100\n",
      "11/11 [==============================] - 1s 125ms/step - loss: 0.9465 - accuracy: 0.7847\n",
      "Epoch 40/100\n",
      "11/11 [==============================] - 1s 126ms/step - loss: 0.9334 - accuracy: 0.8000\n",
      "Epoch 41/100\n",
      "11/11 [==============================] - 1s 134ms/step - loss: 0.9186 - accuracy: 0.8194\n",
      "Epoch 42/100\n",
      "11/11 [==============================] - 2s 135ms/step - loss: 0.9209 - accuracy: 0.8028\n",
      "Epoch 43/100\n",
      "11/11 [==============================] - 1s 122ms/step - loss: 0.9121 - accuracy: 0.8028\n",
      "Epoch 44/100\n",
      "11/11 [==============================] - 1s 125ms/step - loss: 0.9219 - accuracy: 0.7792\n",
      "Epoch 45/100\n",
      "11/11 [==============================] - 1s 123ms/step - loss: 0.9119 - accuracy: 0.7875\n",
      "Epoch 46/100\n",
      "11/11 [==============================] - 1s 124ms/step - loss: 0.8903 - accuracy: 0.8153\n",
      "Epoch 47/100\n",
      "11/11 [==============================] - 1s 124ms/step - loss: 0.8747 - accuracy: 0.8333\n",
      "Epoch 48/100\n",
      "11/11 [==============================] - 1s 124ms/step - loss: 0.8591 - accuracy: 0.8472\n",
      "Epoch 49/100\n",
      "11/11 [==============================] - 1s 125ms/step - loss: 0.8601 - accuracy: 0.8389\n",
      "Epoch 50/100\n",
      "11/11 [==============================] - 1s 124ms/step - loss: 0.8747 - accuracy: 0.8042\n",
      "Epoch 51/100\n",
      "11/11 [==============================] - 1s 124ms/step - loss: 0.8663 - accuracy: 0.8125\n",
      "Epoch 52/100\n",
      "11/11 [==============================] - 1s 124ms/step - loss: 0.8632 - accuracy: 0.8139\n",
      "Epoch 53/100\n",
      "11/11 [==============================] - 1s 123ms/step - loss: 0.8407 - accuracy: 0.8361\n",
      "Epoch 54/100\n",
      "11/11 [==============================] - 1s 124ms/step - loss: 0.8363 - accuracy: 0.8389\n",
      "Epoch 55/100\n",
      "11/11 [==============================] - 1s 126ms/step - loss: 0.8167 - accuracy: 0.8542\n",
      "Epoch 56/100\n",
      "11/11 [==============================] - 1s 123ms/step - loss: 0.8575 - accuracy: 0.8014\n",
      "Epoch 57/100\n",
      "11/11 [==============================] - 1s 124ms/step - loss: 0.8977 - accuracy: 0.7236\n",
      "Epoch 58/100\n",
      "11/11 [==============================] - 1s 124ms/step - loss: 0.8626 - accuracy: 0.7722\n",
      "Epoch 59/100\n",
      "11/11 [==============================] - 1s 122ms/step - loss: 0.8421 - accuracy: 0.7986\n",
      "Epoch 60/100\n",
      "11/11 [==============================] - 1s 124ms/step - loss: 0.8384 - accuracy: 0.7889\n",
      "Epoch 61/100\n",
      "11/11 [==============================] - 1s 122ms/step - loss: 0.8241 - accuracy: 0.8208\n",
      "Epoch 62/100\n",
      "11/11 [==============================] - 1s 127ms/step - loss: 0.8140 - accuracy: 0.8194\n",
      "Epoch 63/100\n",
      "11/11 [==============================] - 1s 125ms/step - loss: 0.8042 - accuracy: 0.8264\n",
      "Epoch 64/100\n",
      "11/11 [==============================] - 2s 138ms/step - loss: 0.7983 - accuracy: 0.8292\n",
      "Epoch 65/100\n",
      "11/11 [==============================] - 1s 136ms/step - loss: 0.7804 - accuracy: 0.8528\n",
      "Epoch 66/100\n",
      "11/11 [==============================] - 1s 127ms/step - loss: 0.7653 - accuracy: 0.8653\n",
      "Epoch 67/100\n",
      "11/11 [==============================] - 1s 125ms/step - loss: 0.7515 - accuracy: 0.8750\n",
      "Epoch 68/100\n",
      "11/11 [==============================] - 1s 125ms/step - loss: 0.7411 - accuracy: 0.8875\n",
      "Epoch 69/100\n",
      "11/11 [==============================] - 1s 128ms/step - loss: 0.7458 - accuracy: 0.8778\n",
      "Epoch 70/100\n",
      "11/11 [==============================] - 1s 125ms/step - loss: 0.7456 - accuracy: 0.8722\n",
      "Epoch 71/100\n",
      "11/11 [==============================] - 1s 129ms/step - loss: 0.7398 - accuracy: 0.8722\n",
      "Epoch 72/100\n",
      "11/11 [==============================] - 1s 125ms/step - loss: 0.7344 - accuracy: 0.8722\n",
      "Epoch 73/100\n",
      "11/11 [==============================] - 1s 127ms/step - loss: 0.7355 - accuracy: 0.8667\n",
      "Epoch 74/100\n",
      "11/11 [==============================] - 1s 128ms/step - loss: 0.7521 - accuracy: 0.8458\n",
      "Epoch 75/100\n",
      "11/11 [==============================] - 1s 128ms/step - loss: 0.7285 - accuracy: 0.8694\n",
      "Epoch 76/100\n",
      "11/11 [==============================] - 1s 125ms/step - loss: 0.7126 - accuracy: 0.8819\n",
      "Epoch 77/100\n",
      "11/11 [==============================] - 1s 124ms/step - loss: 0.7157 - accuracy: 0.8750\n",
      "Epoch 78/100\n",
      "11/11 [==============================] - 1s 126ms/step - loss: 0.7564 - accuracy: 0.8194\n",
      "Epoch 79/100\n",
      "11/11 [==============================] - 1s 125ms/step - loss: 0.7448 - accuracy: 0.8264\n",
      "Epoch 80/100\n",
      "11/11 [==============================] - 1s 125ms/step - loss: 0.7469 - accuracy: 0.8208\n",
      "Epoch 81/100\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "11/11 [==============================] - 1s 126ms/step - loss: 0.7319 - accuracy: 0.8361\n",
      "Epoch 82/100\n",
      "11/11 [==============================] - 1s 128ms/step - loss: 0.7035 - accuracy: 0.8653\n",
      "Epoch 83/100\n",
      "11/11 [==============================] - 1s 127ms/step - loss: 0.6919 - accuracy: 0.8708\n",
      "Epoch 84/100\n",
      "11/11 [==============================] - 1s 127ms/step - loss: 0.7332 - accuracy: 0.8264\n",
      "Epoch 85/100\n",
      "11/11 [==============================] - 1s 124ms/step - loss: 0.7259 - accuracy: 0.8333\n",
      "Epoch 86/100\n",
      "11/11 [==============================] - 1s 125ms/step - loss: 0.6823 - accuracy: 0.8764\n",
      "Epoch 87/100\n",
      "11/11 [==============================] - 1s 126ms/step - loss: 0.6748 - accuracy: 0.8819\n",
      "Epoch 88/100\n",
      "11/11 [==============================] - 1s 126ms/step - loss: 0.6637 - accuracy: 0.8889\n",
      "Epoch 89/100\n",
      "11/11 [==============================] - 1s 126ms/step - loss: 0.7000 - accuracy: 0.8556\n",
      "Epoch 90/100\n",
      "11/11 [==============================] - 1s 125ms/step - loss: 0.6651 - accuracy: 0.8833\n",
      "Epoch 91/100\n",
      "11/11 [==============================] - 1s 125ms/step - loss: 0.6520 - accuracy: 0.8931\n",
      "Epoch 92/100\n",
      "11/11 [==============================] - 1s 127ms/step - loss: 0.6386 - accuracy: 0.9014\n",
      "Epoch 93/100\n",
      "11/11 [==============================] - 1s 125ms/step - loss: 0.6300 - accuracy: 0.9125\n",
      "Epoch 94/100\n",
      "11/11 [==============================] - 1s 125ms/step - loss: 0.6222 - accuracy: 0.9139\n",
      "Epoch 95/100\n",
      "11/11 [==============================] - 1s 122ms/step - loss: 0.6266 - accuracy: 0.9056\n",
      "Epoch 96/100\n",
      "11/11 [==============================] - 1s 125ms/step - loss: 0.6226 - accuracy: 0.9056\n",
      "Epoch 97/100\n",
      "11/11 [==============================] - 1s 127ms/step - loss: 0.6039 - accuracy: 0.9208\n",
      "Epoch 98/100\n",
      "11/11 [==============================] - 1s 125ms/step - loss: 0.6059 - accuracy: 0.9139\n",
      "Epoch 99/100\n",
      "11/11 [==============================] - 1s 126ms/step - loss: 0.6112 - accuracy: 0.9125\n",
      "Epoch 100/100\n",
      "11/11 [==============================] - 1s 126ms/step - loss: 0.6037 - accuracy: 0.9139\n"
     ]
    }
   ],
   "source": [
    "c_rnnModel = eegConvRNN(X_train.shape[1],X_train.shape[2],y_train.shape[1])\n",
    "c_rnnModel.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])\n",
    "crnn_history = c_rnnModel.fit(X_train, y_train, epochs= 100, batch_size = 70)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "scrolled": true
   },
   "outputs": [],
   "source": [
    "#Saving model in Keras:\n",
    "c_rnnModel.save('RNN_CNN_Model')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 56,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Text(0.5, 0, 'Epochs')"
      ]
     },
     "execution_count": 56,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "sn.lineplot(x=list(range(100)),y=crnn_history.history['accuracy'])\n",
    "plt.ylabel('Accuracy')\n",
    "plt.xlabel('Epochs')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 57,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[<matplotlib.lines.Line2D at 0x1d547b0de50>]"
      ]
     },
     "execution_count": 57,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.plot(crnn_history.history['accuracy'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 58,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "6/6 [==============================] - 1s 28ms/step - loss: 0.8050 - accuracy: 0.7667\n"
     ]
    }
   ],
   "source": [
    "accuracy = c_rnnModel.evaluate(X_test, y_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 60,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<AxesSubplot:>"
      ]
     },
     "execution_count": 60,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 720x504 with 2 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "predictions = c_rnnModel.predict(X_test)\n",
    "y_pred = [p.argmax() for p in predictions]\n",
    "y_true = [y.argmax() for y in y_test]\n",
    "conf_mat = confusion_matrix(y_true,y_pred)\n",
    "df_conf = pd.DataFrame(conf_mat,index=[i for i in 'Right Left Passive'.split()],\n",
    "                      columns = [i for i in 'Right Left Passive'.split()])\n",
    "plt.figure(figsize = (10,7))\n",
    "sn.heatmap(df_conf, annot=True,cmap='Blues')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 63,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "              precision    recall  f1-score   support\n",
      "\n",
      "           1       0.73      0.60      0.66        53\n",
      "           2       0.82      0.83      0.83        72\n",
      "           3       0.73      0.84      0.78        55\n",
      "\n",
      "    accuracy                           0.77       180\n",
      "   macro avg       0.76      0.76      0.76       180\n",
      "weighted avg       0.77      0.77      0.76       180\n",
      "\n"
     ]
    }
   ],
   "source": [
    "print(classification_report(y_true,y_pred))"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
