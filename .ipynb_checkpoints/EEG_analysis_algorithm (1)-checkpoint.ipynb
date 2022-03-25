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
   "execution_count": 9,
   "metadata": {
    "scrolled": false
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Requirement already satisfied: keras in c:\\users\\nelso\\anaconda3\\lib\\site-packages (2.4.3)\n",
      "Requirement already satisfied: pyyaml in c:\\users\\nelso\\anaconda3\\lib\\site-packages (from keras) (5.3.1)\n",
      "Requirement already satisfied: numpy>=1.9.1 in c:\\users\\nelso\\anaconda3\\lib\\site-packages (from keras) (1.19.2)\n",
      "Requirement already satisfied: scipy>=0.14 in c:\\users\\nelso\\anaconda3\\lib\\site-packages (from keras) (1.5.2)\n",
      "Requirement already satisfied: h5py in c:\\users\\nelso\\anaconda3\\lib\\site-packages (from keras) (2.10.0)\n",
      "Requirement already satisfied: six in c:\\users\\nelso\\anaconda3\\lib\\site-packages (from h5py->keras) (1.15.0)\n"
     ]
    }
   ],
   "source": [
    "!pip install keras\n",
    "\n",
    "import scipy.io\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import mne\n",
    "import tensorflow as tf"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 83,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.metrics import confusion_matrix, plot_confusion_matrix\n",
    "import pandas as pd\n",
    "import seaborn as sn\n",
    "sn.set_theme()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
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
   "execution_count": 12,
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
   "execution_count": 13,
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
   "execution_count": 17,
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
   "execution_count": 18,
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
   "execution_count": 19,
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
   "execution_count": 28,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Loading data for 900 events and 201 original time points ...\n"
     ]
    }
   ],
   "source": [
    "np_epochs = eeg_epochs.get_data(picks=eeg_raw.info.ch_names[:-1])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(900, 21, 201)"
      ]
     },
     "execution_count": 29,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "np_epochs.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[<matplotlib.lines.Line2D at 0x12d2da8be80>]"
      ]
     },
     "execution_count": 30,
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
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.plot(np_epochs[0][2])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Loading data for 900 events and 201 original time points ...\n",
      "Index(['time', 'condition', 'epoch', 'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4',\n",
      "       'P3', 'P4', 'O1', 'O2', 'A1', 'A2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6',\n",
      "       'Fz', 'Cz', 'Pz', 'STI 1'],\n",
      "      dtype='object')\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>time</th>\n",
       "      <th>condition</th>\n",
       "      <th>epoch</th>\n",
       "      <th>Fp1</th>\n",
       "      <th>Fp2</th>\n",
       "      <th>F3</th>\n",
       "      <th>F4</th>\n",
       "      <th>C3</th>\n",
       "      <th>C4</th>\n",
       "      <th>P3</th>\n",
       "      <th>...</th>\n",
       "      <th>F7</th>\n",
       "      <th>F8</th>\n",
       "      <th>T3</th>\n",
       "      <th>T4</th>\n",
       "      <th>T5</th>\n",
       "      <th>T6</th>\n",
       "      <th>Fz</th>\n",
       "      <th>Cz</th>\n",
       "      <th>Pz</th>\n",
       "      <th>STI 1</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>-200</td>\n",
       "      <td>3</td>\n",
       "      <td>0</td>\n",
       "      <td>5.053199</td>\n",
       "      <td>2.689989</td>\n",
       "      <td>-5.803276</td>\n",
       "      <td>4.002702</td>\n",
       "      <td>-6.043346</td>\n",
       "      <td>5.773800</td>\n",
       "      <td>-0.234529</td>\n",
       "      <td>...</td>\n",
       "      <td>3.236365</td>\n",
       "      <td>4.221769</td>\n",
       "      <td>12.632798</td>\n",
       "      <td>7.946899</td>\n",
       "      <td>2.946757</td>\n",
       "      <td>1.735178</td>\n",
       "      <td>1.712242</td>\n",
       "      <td>1.619176</td>\n",
       "      <td>-0.883479</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>-195</td>\n",
       "      <td>3</td>\n",
       "      <td>0</td>\n",
       "      <td>6.503455</td>\n",
       "      <td>5.232009</td>\n",
       "      <td>0.832902</td>\n",
       "      <td>1.828446</td>\n",
       "      <td>-4.332377</td>\n",
       "      <td>4.276317</td>\n",
       "      <td>4.386328</td>\n",
       "      <td>...</td>\n",
       "      <td>3.257334</td>\n",
       "      <td>6.122839</td>\n",
       "      <td>14.158893</td>\n",
       "      <td>11.160727</td>\n",
       "      <td>7.626500</td>\n",
       "      <td>4.496519</td>\n",
       "      <td>2.327678</td>\n",
       "      <td>3.848168</td>\n",
       "      <td>3.486017</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>-190</td>\n",
       "      <td>3</td>\n",
       "      <td>0</td>\n",
       "      <td>3.911538</td>\n",
       "      <td>4.800934</td>\n",
       "      <td>3.859604</td>\n",
       "      <td>-1.343946</td>\n",
       "      <td>-1.610117</td>\n",
       "      <td>1.819806</td>\n",
       "      <td>5.091662</td>\n",
       "      <td>...</td>\n",
       "      <td>2.946959</td>\n",
       "      <td>4.309575</td>\n",
       "      <td>6.603486</td>\n",
       "      <td>9.151607</td>\n",
       "      <td>6.171512</td>\n",
       "      <td>3.889637</td>\n",
       "      <td>0.252421</td>\n",
       "      <td>3.040710</td>\n",
       "      <td>2.822846</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>-185</td>\n",
       "      <td>3</td>\n",
       "      <td>0</td>\n",
       "      <td>-0.991432</td>\n",
       "      <td>1.366300</td>\n",
       "      <td>1.864193</td>\n",
       "      <td>-1.580487</td>\n",
       "      <td>-0.429201</td>\n",
       "      <td>0.748691</td>\n",
       "      <td>2.424969</td>\n",
       "      <td>...</td>\n",
       "      <td>-1.152791</td>\n",
       "      <td>0.187703</td>\n",
       "      <td>-3.890810</td>\n",
       "      <td>2.657826</td>\n",
       "      <td>1.102557</td>\n",
       "      <td>1.425391</td>\n",
       "      <td>-1.194037</td>\n",
       "      <td>0.830799</td>\n",
       "      <td>-1.070341</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>-180</td>\n",
       "      <td>3</td>\n",
       "      <td>0</td>\n",
       "      <td>-3.767158</td>\n",
       "      <td>-1.432812</td>\n",
       "      <td>-1.848027</td>\n",
       "      <td>1.027544</td>\n",
       "      <td>-1.404354</td>\n",
       "      <td>1.785669</td>\n",
       "      <td>0.306342</td>\n",
       "      <td>...</td>\n",
       "      <td>-6.994637</td>\n",
       "      <td>-1.371632</td>\n",
       "      <td>-9.674841</td>\n",
       "      <td>-2.264825</td>\n",
       "      <td>-2.245277</td>\n",
       "      <td>0.089745</td>\n",
       "      <td>-0.013839</td>\n",
       "      <td>0.051714</td>\n",
       "      <td>-2.919657</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>180895</th>\n",
       "      <td>780</td>\n",
       "      <td>1</td>\n",
       "      <td>899</td>\n",
       "      <td>-18.801772</td>\n",
       "      <td>-11.325824</td>\n",
       "      <td>-7.682501</td>\n",
       "      <td>2.037508</td>\n",
       "      <td>-0.560179</td>\n",
       "      <td>1.267404</td>\n",
       "      <td>4.063049</td>\n",
       "      <td>...</td>\n",
       "      <td>-10.162613</td>\n",
       "      <td>3.126943</td>\n",
       "      <td>-8.696414</td>\n",
       "      <td>-3.591924</td>\n",
       "      <td>-6.715831</td>\n",
       "      <td>-5.540434</td>\n",
       "      <td>1.245541</td>\n",
       "      <td>7.730074</td>\n",
       "      <td>11.315970</td>\n",
       "      <td>1.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>180896</th>\n",
       "      <td>785</td>\n",
       "      <td>1</td>\n",
       "      <td>899</td>\n",
       "      <td>-17.976744</td>\n",
       "      <td>-13.954906</td>\n",
       "      <td>-7.118366</td>\n",
       "      <td>1.500282</td>\n",
       "      <td>-0.951576</td>\n",
       "      <td>1.089905</td>\n",
       "      <td>3.542149</td>\n",
       "      <td>...</td>\n",
       "      <td>-9.728298</td>\n",
       "      <td>1.671579</td>\n",
       "      <td>-7.943194</td>\n",
       "      <td>-3.849091</td>\n",
       "      <td>-6.393213</td>\n",
       "      <td>-3.297807</td>\n",
       "      <td>0.241899</td>\n",
       "      <td>7.645967</td>\n",
       "      <td>11.359275</td>\n",
       "      <td>1.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>180897</th>\n",
       "      <td>790</td>\n",
       "      <td>1</td>\n",
       "      <td>899</td>\n",
       "      <td>-10.683867</td>\n",
       "      <td>-11.619120</td>\n",
       "      <td>-4.887296</td>\n",
       "      <td>-0.693662</td>\n",
       "      <td>-0.621122</td>\n",
       "      <td>0.589301</td>\n",
       "      <td>2.412776</td>\n",
       "      <td>...</td>\n",
       "      <td>-6.655877</td>\n",
       "      <td>0.755986</td>\n",
       "      <td>-4.370398</td>\n",
       "      <td>-4.575911</td>\n",
       "      <td>-4.890892</td>\n",
       "      <td>0.218031</td>\n",
       "      <td>-1.631375</td>\n",
       "      <td>5.476213</td>\n",
       "      <td>9.755518</td>\n",
       "      <td>1.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>180898</th>\n",
       "      <td>795</td>\n",
       "      <td>1</td>\n",
       "      <td>899</td>\n",
       "      <td>-2.138425</td>\n",
       "      <td>-7.090175</td>\n",
       "      <td>-2.307934</td>\n",
       "      <td>-3.045653</td>\n",
       "      <td>-0.022509</td>\n",
       "      <td>0.422084</td>\n",
       "      <td>1.251983</td>\n",
       "      <td>...</td>\n",
       "      <td>-2.880932</td>\n",
       "      <td>1.682184</td>\n",
       "      <td>0.264064</td>\n",
       "      <td>-3.136103</td>\n",
       "      <td>-2.163110</td>\n",
       "      <td>6.108863</td>\n",
       "      <td>-3.507241</td>\n",
       "      <td>1.933979</td>\n",
       "      <td>7.074663</td>\n",
       "      <td>1.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>180899</th>\n",
       "      <td>800</td>\n",
       "      <td>1</td>\n",
       "      <td>899</td>\n",
       "      <td>1.401806</td>\n",
       "      <td>-3.640922</td>\n",
       "      <td>-1.007667</td>\n",
       "      <td>-3.962688</td>\n",
       "      <td>-0.503156</td>\n",
       "      <td>1.189531</td>\n",
       "      <td>0.176418</td>\n",
       "      <td>...</td>\n",
       "      <td>-0.882754</td>\n",
       "      <td>3.725227</td>\n",
       "      <td>3.347457</td>\n",
       "      <td>0.880432</td>\n",
       "      <td>0.594582</td>\n",
       "      <td>13.756105</td>\n",
       "      <td>-4.842547</td>\n",
       "      <td>-1.496817</td>\n",
       "      <td>4.116201</td>\n",
       "      <td>1.0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>180900 rows × 25 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "        time condition  epoch        Fp1        Fp2        F3        F4  \\\n",
       "0       -200         3      0   5.053199   2.689989 -5.803276  4.002702   \n",
       "1       -195         3      0   6.503455   5.232009  0.832902  1.828446   \n",
       "2       -190         3      0   3.911538   4.800934  3.859604 -1.343946   \n",
       "3       -185         3      0  -0.991432   1.366300  1.864193 -1.580487   \n",
       "4       -180         3      0  -3.767158  -1.432812 -1.848027  1.027544   \n",
       "...      ...       ...    ...        ...        ...       ...       ...   \n",
       "180895   780         1    899 -18.801772 -11.325824 -7.682501  2.037508   \n",
       "180896   785         1    899 -17.976744 -13.954906 -7.118366  1.500282   \n",
       "180897   790         1    899 -10.683867 -11.619120 -4.887296 -0.693662   \n",
       "180898   795         1    899  -2.138425  -7.090175 -2.307934 -3.045653   \n",
       "180899   800         1    899   1.401806  -3.640922 -1.007667 -3.962688   \n",
       "\n",
       "              C3        C4        P3  ...         F7        F8         T3  \\\n",
       "0      -6.043346  5.773800 -0.234529  ...   3.236365  4.221769  12.632798   \n",
       "1      -4.332377  4.276317  4.386328  ...   3.257334  6.122839  14.158893   \n",
       "2      -1.610117  1.819806  5.091662  ...   2.946959  4.309575   6.603486   \n",
       "3      -0.429201  0.748691  2.424969  ...  -1.152791  0.187703  -3.890810   \n",
       "4      -1.404354  1.785669  0.306342  ...  -6.994637 -1.371632  -9.674841   \n",
       "...          ...       ...       ...  ...        ...       ...        ...   \n",
       "180895 -0.560179  1.267404  4.063049  ... -10.162613  3.126943  -8.696414   \n",
       "180896 -0.951576  1.089905  3.542149  ...  -9.728298  1.671579  -7.943194   \n",
       "180897 -0.621122  0.589301  2.412776  ...  -6.655877  0.755986  -4.370398   \n",
       "180898 -0.022509  0.422084  1.251983  ...  -2.880932  1.682184   0.264064   \n",
       "180899 -0.503156  1.189531  0.176418  ...  -0.882754  3.725227   3.347457   \n",
       "\n",
       "               T4        T5         T6        Fz        Cz         Pz  STI 1  \n",
       "0        7.946899  2.946757   1.735178  1.712242  1.619176  -0.883479    0.0  \n",
       "1       11.160727  7.626500   4.496519  2.327678  3.848168   3.486017    0.0  \n",
       "2        9.151607  6.171512   3.889637  0.252421  3.040710   2.822846    0.0  \n",
       "3        2.657826  1.102557   1.425391 -1.194037  0.830799  -1.070341    0.0  \n",
       "4       -2.264825 -2.245277   0.089745 -0.013839  0.051714  -2.919657    0.0  \n",
       "...           ...       ...        ...       ...       ...        ...    ...  \n",
       "180895  -3.591924 -6.715831  -5.540434  1.245541  7.730074  11.315970    1.0  \n",
       "180896  -3.849091 -6.393213  -3.297807  0.241899  7.645967  11.359275    1.0  \n",
       "180897  -4.575911 -4.890892   0.218031 -1.631375  5.476213   9.755518    1.0  \n",
       "180898  -3.136103 -2.163110   6.108863 -3.507241  1.933979   7.074663    1.0  \n",
       "180899   0.880432  0.594582  13.756105 -4.842547 -1.496817   4.116201    1.0  \n",
       "\n",
       "[180900 rows x 25 columns]"
      ]
     },
     "execution_count": 31,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df_epochs = eeg_epochs.to_data_frame(picks='all')\n",
    "print(df_epochs.columns)\n",
    "df_epochs"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "900"
      ]
     },
     "execution_count": 32,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Obtain labels for each epoch (instead of each time step)\n",
    "labels = []\n",
    "for i in df_epochs['epoch'].unique():\n",
    "    labels.append(np.array(df_epochs['condition'][df_epochs['epoch'] == i])[0])\n",
    "len(labels)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.model_selection import train_test_split"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(180, 201, 21)\n",
      "(180, 4)\n"
     ]
    }
   ],
   "source": [
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
   "execution_count": 35,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(720, 201, 21)"
      ]
     },
     "execution_count": 35,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X_train.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
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
   "execution_count": 37,
   "metadata": {},
   "outputs": [],
   "source": [
    "rnn_model = eegRNN(X_train.shape[1],X_train.shape[2],y_train.shape[1])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "metadata": {},
   "outputs": [],
   "source": [
    "rnn_model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1/30\n",
      "15/15 [==============================] - 6s 101ms/step - loss: 1.3707 - accuracy: 0.2847\n",
      "Epoch 2/30\n",
      "15/15 [==============================] - 1s 96ms/step - loss: 1.1890 - accuracy: 0.4451\n",
      "Epoch 3/30\n",
      "15/15 [==============================] - 1s 96ms/step - loss: 1.0882 - accuracy: 0.4486\n",
      "Epoch 4/30\n",
      "15/15 [==============================] - 1s 97ms/step - loss: 1.0115 - accuracy: 0.5068\n",
      "Epoch 5/30\n",
      "15/15 [==============================] - 1s 97ms/step - loss: 0.9338 - accuracy: 0.5355\n",
      "Epoch 6/30\n",
      "15/15 [==============================] - 1s 96ms/step - loss: 0.8565 - accuracy: 0.5660\n",
      "Epoch 7/30\n",
      "15/15 [==============================] - 1s 96ms/step - loss: 0.7878 - accuracy: 0.6099\n",
      "Epoch 8/30\n",
      "15/15 [==============================] - 1s 99ms/step - loss: 0.7257 - accuracy: 0.6676\n",
      "Epoch 9/30\n",
      "15/15 [==============================] - 1s 99ms/step - loss: 0.7081 - accuracy: 0.6819\n",
      "Epoch 10/30\n",
      "15/15 [==============================] - 1s 100ms/step - loss: 0.6722 - accuracy: 0.7364\n",
      "Epoch 11/30\n",
      "15/15 [==============================] - 1s 98ms/step - loss: 0.6648 - accuracy: 0.7098\n",
      "Epoch 12/30\n",
      "15/15 [==============================] - 1s 99ms/step - loss: 0.5851 - accuracy: 0.7474\n",
      "Epoch 13/30\n",
      "15/15 [==============================] - 1s 99ms/step - loss: 0.5538 - accuracy: 0.8043\n",
      "Epoch 14/30\n",
      "15/15 [==============================] - 1s 98ms/step - loss: 0.4943 - accuracy: 0.8331\n",
      "Epoch 15/30\n",
      "15/15 [==============================] - 1s 99ms/step - loss: 0.4958 - accuracy: 0.8103\n",
      "Epoch 16/30\n",
      "15/15 [==============================] - 1s 99ms/step - loss: 0.3968 - accuracy: 0.8797\n",
      "Epoch 17/30\n",
      "15/15 [==============================] - 1s 97ms/step - loss: 0.3686 - accuracy: 0.8873\n",
      "Epoch 18/30\n",
      "15/15 [==============================] - 2s 101ms/step - loss: 0.3267 - accuracy: 0.8984\n",
      "Epoch 19/30\n",
      "15/15 [==============================] - 1s 98ms/step - loss: 0.2829 - accuracy: 0.9227\n",
      "Epoch 20/30\n",
      "15/15 [==============================] - 1s 97ms/step - loss: 0.2727 - accuracy: 0.9174\n",
      "Epoch 21/30\n",
      "15/15 [==============================] - 1s 96ms/step - loss: 0.3347 - accuracy: 0.9028\n",
      "Epoch 22/30\n",
      "15/15 [==============================] - 1s 98ms/step - loss: 0.3226 - accuracy: 0.9029\n",
      "Epoch 23/30\n",
      "15/15 [==============================] - 1s 96ms/step - loss: 0.3078 - accuracy: 0.9057\n",
      "Epoch 24/30\n",
      "15/15 [==============================] - 1s 97ms/step - loss: 0.3317 - accuracy: 0.8942\n",
      "Epoch 25/30\n",
      "15/15 [==============================] - 1s 99ms/step - loss: 0.3194 - accuracy: 0.8881\n",
      "Epoch 26/30\n",
      "15/15 [==============================] - 1s 97ms/step - loss: 0.2297 - accuracy: 0.9356\n",
      "Epoch 27/30\n",
      "15/15 [==============================] - 1s 97ms/step - loss: 0.1877 - accuracy: 0.9558\n",
      "Epoch 28/30\n",
      "15/15 [==============================] - 1s 96ms/step - loss: 0.2029 - accuracy: 0.9493\n",
      "Epoch 29/30\n",
      "15/15 [==============================] - 1s 96ms/step - loss: 0.1848 - accuracy: 0.9494\n",
      "Epoch 30/30\n",
      "15/15 [==============================] - 1s 96ms/step - loss: 0.1583 - accuracy: 0.9613\n"
     ]
    }
   ],
   "source": [
    "rnn_history = rnn_model.fit(X_train, y_train, epochs= 30, batch_size = 50)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "metadata": {
    "scrolled": false
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[<matplotlib.lines.Line2D at 0x12d2d5c9100>]"
      ]
     },
     "execution_count": 40,
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
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.plot(rnn_history.history['accuracy'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[<matplotlib.lines.Line2D at 0x12d2d6210a0>]"
      ]
     },
     "execution_count": 41,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAXUAAAD4CAYAAAATpHZ6AAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjMuMiwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8vihELAAAACXBIWXMAAAsTAAALEwEAmpwYAAAkE0lEQVR4nO3deXRU9d3H8fc3k4UkQCALGEICkR1ZQyCAULeqWBfEuiGCIJSiotU+tfrUtvp0UavVqrWoiIiCaF0Qsa7VFgUVSNj3fQtrwhIIBLL9nj8SW8QACUxyM5PP65wcMjM3M5977smHyZ3f/f3MOYeIiASHEK8DiIiI/6jURUSCiEpdRCSIqNRFRIKISl1EJIiEevXC8fHxrmXLll69vIhIQJo/f36ucy7hRI97VuotW7YkKyvLq5cXEQlIZrb5ZI/r9IuISBBRqYuIBBGVuohIEFGpi4gEEZW6iEgQUamLiAQRlbqISBAJuFJft/sgv3t/BYXFpV5HERGpdQKu1LfuK2DiVxuZuXq311FERGqdgCv1/q3jia8fwbQF27yOIiJS6wRcqYf6QhjYrRmfr9rF/sOFXscREalVAq7UAa5JS6KoxPH+kh1eRxERqVUCstQ7Jjak/VkNmLYg2+soIiK1SkCWuplxTVoSC7fsZ0NOvtdxRERqjYAsdYCB3ZIIMXh3oT4wFRH5VsCWetOG9ejXJoFpC7ZRWuq8jiMiUisEbKkD/DgtiW37C5i3aa/XUUREaoWALvVLOp5FdLhPH5iKiJQL6FKPDPfxo86JfLh0JwWFJV7HERHxXECXOsA1ac3JP1rMpyt2eh1FRMRzAV/qGamxJDWK1LQBIiIEQamHhBiDuicxa20Ouw8c8TqOiIinAr7UAQalJVHq4L1F272OIiLiqaAo9VYJ9emW3IhpuhBJROq4oCh1KBuzvnLHAVZsP+B1FBERzwRNqV/RpRlhPuPdhRqzLiJ1V9CUeuPocC5s34Tpi7ZTXKKl7kSkbjplqZvZRDPbbWbLTvD4EDNbUv71tZl19X/MyrkmrTk5B48ye12uVxFERDxVmXfqk4ABJ3l8I3Cec64L8HtgvB9ynZYL2jWhUVSYxqyLSJ11ylJ3zn0JnHDGLOfc1865feU35wDN/ZStysJDQ7iqazM+Wb6Tg0eKvIohIuIZf59THwl8dKIHzWy0mWWZWVZOTo6fX7rMNWnNOVpcykdLNW2AiNQ9fit1M7uAslK/70TbOOfGO+fSnXPpCQkJ/nrp7+jaPIazE6J5RzM3ikgd5JdSN7MuwARgoHNujz+e8wyy8OO05szduJetew97GUVEpMadcambWQowDRjqnFtz5pHO3NXdkwCYritMRaSOqcyQxteBb4B2ZpZtZiPNbIyZjSnf5LdAHDDOzBaZWVY15q2UpEaR9Dk7jmkLt+GclroTkboj9FQbOOcGn+LxUcAovyXyk2vSkrj37SUs3LqftJTGXscREakRQXNF6fEu65xIvbAQpnyz2esoIiI1JmhLvX5EKMP7pjJt4Tb+uWKX13FERGpE0JY6wD0Xt6FjYkPue2cJuw9qAQ0RCX5BXeoRoT6eGdyNQ0eLufetJfrQVESCXlCXOkDrJg144PIOfLEmh1d1fl1EglzQlzrA0N4tuKBdAg9/uJK1uw56HUdEpNrUiVI3Mx67tiv1I0K5641FHC0u8TqSiEi1qBOlDpDQIILHru3Cyh0HeOLTWnHhq4iI39WZUge4qENThmSk8OKsDXythTREJAjVqVIHeODyDqTGRfM/by0m77DmXBeR4FLnSj0qPJSnb+xOzsGj/Gr6Ug1zFJGgUudKHaBz8xjuubgtHyzZoaXvRCSo1MlSBxhzXit6pcby4IzlmnddRIJGnS11X4jx5PVdMYO7/76I4pJSryOJiJyxOlvqAM0bR/GHqzsxf/M+xs1c73UcEZEzVqdLHWBgtyQGdmvG05+v1TBHEQl4db7UAf5wdSdaJUQzZsp81u3O9zqOiMhpU6kDDeqF8dItPQkPDWHkK5nsPVTodSQRkdOiUi+XHBvF+GHp7Mg7wk8nZ2l+GBEJSCr1Y6SlNOaJ67qSuWkf97+jC5NEJPCccuHpuubKrs3YlHuIJ/65htT4aO66qI3XkUREKk2lXoGxF7ZmY+4hnvznGlrGR3NV12ZeRxIRqRSdfqmAmfHIjzvTq2Usv3hrMfM37/M6kohIpajUTyAi1MfzQ3uQGFOP0a9maSoBEQkIKvWTiI0OZ+LwnhSVlHLrpEwOHNFUvSJSu52y1M1sopntNrNlJ3jczOwZM1tnZkvMLM3/Mb3TKqE+zw/twcbcQ9zx2gKKNEeMiNRilXmnPgkYcJLHLwPalH+NBp4781i1S99W8Tw8qDOz1uby4IzlGuooIrXWKUe/OOe+NLOWJ9lkIPCqK2u6OWbWyMwSnXM7/BWyNri+ZzIbcg/x/BfraREbxU/Pa+V1JBGR7/HHkMYkYOsxt7PL7/teqZvZaMrezZOSkuKHl65Zv7y0HVv3HuaRj1YRHRHKzb1beB1JROQ7/FHqVsF9FZ6fcM6NB8YDpKenB9w5jJAQ4y83dKOgqIRfT19GvTAf1/Zo7nUsEZH/8Mfol2wg+ZjbzYHtfnjeWik8NIRxQ9Lo1zqeX769mPcXB+2uikgA8kepzwCGlY+C6Q3kBdv59OPVC/MxflgP0lvEcs/fF/Hp8p1eRxIRASo3pPF14BugnZllm9lIMxtjZmPKN/kQ2ACsA14Ebq+2tLVIVHgoLw1P55ykGMZOXcgXa3K8jiQignk1PC89Pd1lZWV58tr+lHe4iMEvzmF9Tj6TRvSiT6s4ryOJSBAzs/nOufQTPa4rSs9QTFQYk0f2IiU2ipGvZGqeGBHxlErdD+LqR/DaqAyaNIhg+MR5LM3O8zqSiNRRKnU/adKwHq/9pDcNI8MYOnEuq3ce9DqSiNRBKnU/SmoUydSfZBARGsKQCWXn2UVEapJK3c9axEXz2qjeAAx5ca6m7BWRGqVSrwatm9RnyqgMCopKGPrSXHYfPOJ1JBGpI1Tq1aT9WQ15eURPdh04yrCX5pFXoLnYRaT6qdSrUVpKY8YP68H6nHxunZTJ4cJiryOJSJBTqVez/m0SePrG7izcso/bpiygsFiLbIhI9VGp14AfdU7kkWs688WaHH7+5iJKSgNugkoRCRD+mHpXKuGGninkFRTx8IeraBgZxh+v7oRZRbMWi4icPpV6DRr9g1bsP1zEuJnraRQZxi8HtPc6kogEGZV6Dbv30nbsLygr9pjIMC2LJyJ+pVKvYWbG7wd24kBBEY98tIqYyDBu7BV4S/uJSO2kUveAL8R48vpuHDxSzK/eXUrDyDB+1DnR61giEgQ0+sUj4aEhPH9zD9JSGvOzNxbypRbZEBE/UKl7KDLcx0vDe9K6SQNGT85i3sa9XkcSkQCnUvdYTGQYr97ai2aNIrl1UiZLsvd7HUlEAphKvRZIaFC2yEajqDCGTZzHqp0HvI4kIgFKpV5LJMZEMnVUbyJCQ7h5wjw2aC52ETkNKvVaJCUuitdG9cY5x5AJmotdRKpOpV7LtG5Sn1dH9uLQ0WJufmkuuw5oLnYRqTyVei10TrMYJt3ai9yDR7l5wlz25B/1OpKIBAiVei2VltKYCbf0ZMvewwybqEU2RKRyKlXqZjbAzFab2Tozu7+Cx2PM7H0zW2xmy81shP+j1j19WsXxwtAerNl1kBEvz+PQUS2yISInd8pSNzMf8DfgMqAjMNjMOh632R3ACudcV+B84AkzC/dz1jrp/HZN+OvgNBZn5zHqlSyOFJV4HUlEarHKvFPvBaxzzm1wzhUCbwADj9vGAQ2sbILw+sBeQG8r/WRAp7P483VdmLNxD7dNmU9BoYpdRCpWmVJPArYeczu7/L5jPQt0ALYDS4GfOee+t26bmY02sywzy8rJ0VwnVTGoe3P+eHVnZq7J4drnv9ZwRxGpUGVKvaLleY5fj+1SYBHQDOgGPGtmDb/3Q86Nd86lO+fSExISqhhVbspI4aVb0tmy9zBXPTub2WtzvY4kIrVMZUo9G0g+5nZzyt6RH2sEMM2VWQdsBLSsTzW4sH1TZoztR0KDCIZNnMvzX6zHOa15KiJlKlPqmUAbM0st//DzRmDGcdtsAS4CMLOmQDtggz+Dyn+lxkfz7u3nclmnRB79aBVjpy7UyBgRASpR6s65YmAs8AmwEnjTObfczMaY2ZjyzX4P9DWzpcDnwH3OOZ0bqEbREaE8e1N3/vey9ny0bAeDxn3FxtxDXscSEY+ZV3+6p6enu6ysLE9eO9jMXpvLna8voLjU8dQN3bioQ1OvI4lINTGz+c659BM9ritKg0C/NvHMGNuPlNgoRr6SxVOfraG0VOfZReoilXqQSI6N4p3b+nJNWhJPfbaW0ZOzOHBEUwuI1DUq9SBSL8zHE9d15aErOzJzdQ5X/XU2K7ZrwQ2RukSlHmTMjOHnpvL66N4UFJUwaNxXvJW19dQ/KCJBQaUepHq2jOUfd/anR4vG3Pv2Eu57e4nmjRGpA1TqQSyhQQSTR2Yw9oLW/D1rK9eM+5rNezTsUSSYqdSDnC/E+MWl7Zg4PJ1t+wu44q+z+WT5Tq9jiUg1UanXERe2b8o/7uxHanw0P508n0c+XElxyffmXBORAKdSr0OSY6N4a0wfhmSk8MKXG7jpxbns1hqoIkFFpV7HRIT6+OOgzvzlhq4s3ZbHj56ZzTfr93gdS0T8RKVeRw3q3pz3xp5Lw8hQbn5pLjMWHz/xpogEIpV6Hda2aQNmjO1HjxaNufuNhUxfuM3rSCJyhlTqdVz9iFAmjehJRmoc97y5iLfnZ3sdSUTOgEpdiAoPZeLwnpzbKp57317Mm5m6AlUkUKnUBYDIcB8Tbkmnf5sEfvnOEqbO3eJ1JBE5DSp1+Y96YT7GD+3BBe0S+NW7S5k8Z7PXkUSkilTq8h31wnw8P7QHP+zQhN9MX8akrzZ6HUlEqkClLt8TEepj3JAeXNKxKQ+9v4IJs7TcrEigUKlLhcJDQ/jbkDQu63QWf/hgJeO/XO91JBGpBJW6nFCYL4RnBnfnii6JPPzhKsbNXOd1JBE5hVCvA0jtFuYL4akbuuELMR77eDULNu+jb6t4eqXG0iGxIb4Q8zqiiBxDpS6nFOoL4cnru3FWTD3+sXgHn63cDZRduJTWojG9WjamZ8tYuiY3ol6Yz+O0InWbOefNqvPp6ekuKyvLk9eWM7N9fwGZm/aWfW3cx+pdBwEI94XQpXkMPVNj6dsqjn6t4zHTO3kRfzKz+c659BM+rlKXM7X/cCFZm/aRuWkv8zbtZWl2HsWljq7JjXjgRx3olRrrdUSRoOGXUjezAcDTgA+Y4Jx7tIJtzgeeAsKAXOfceSd7TpV68CooLOH9Jdt58tM17DxwhIs7NuX+y9rTKqG+19FEAt4Zl7qZ+YA1wMVANpAJDHbOrThmm0bA18AA59wWM2vinNt9sudVqQe/gsISJn61kedmrqegqITBvZL52UVtSWgQ4XU0kYB1qlKvzJDGXsA659wG51wh8AYw8LhtbgKmOee2AJyq0KVuiAz3cccFrZl57/kMyUjhjXlbOf/xf/PM52s5XFjsdTyRoFSZUk8Cjp22L7v8vmO1BRqb2Uwzm29mw/wVUAJffP0IfjewE5/e8wP6t0ngyX+u4fzHZ/LGvC2UlHrzmY5IsKpMqVc0fOH438RQoAdwOXAp8Bsza/u9JzIbbWZZZpaVk5NT5bAS2M5OqM/zQ3vw9pg+NG8cyf3TlnLZ018yf/M+r6OJBI3KlHo2kHzM7ebA8WufZQMfO+cOOedygS+Brsc/kXNuvHMu3TmXnpCQcLqZJcClt4zlndv68tyQNA4XljB84jxW7zzodSyRoFCZUs8E2phZqpmFAzcCM47b5j2gv5mFmlkUkAGs9G9UCSZmxmWdE/n7T/sQGe5jxMvz2HXgiNexRALeKUvdOVcMjAU+oayo33TOLTezMWY2pnyblcDHwBJgHmXDHpdVX2wJFkmNIpk4vCd5BUXcOimTQ0f1AarImdDFR1Ir/Hv1bka9kkX/NvFMGJZOqE9zzYlUxB9DGkWq3QXtmvD7gZ2YuTqH37y3HK/ebIgEOk3oJbXGTRkpZO87zLiZ60mOjeT281t7HUkk4KjUpVb5xSXtyN5XwGMfryapUSQDux1/SYSInIxKXWqVkBDj8eu6sPPAEe59awlnNaxHxtlxXscSCRg6py61TkSoj/FDe5AcG8noyfNZtzvf60giAUOlLrVSo6hwJo3oRZjPGP7yPHIOHvU6kkhAUKlLrZUcG8VLt/RkT34ho17J1CRgIpWgUpdarWtyI54Z3J2l2/K46/WFFJWUeh1JpFZTqUutd3HHpvzfVefw2crd3Dl1IYXFKnaRE1GpS0AY2qclv72iIx8v38ltU+ZzpKjE60gitZJKXQLGrf1S+cPVnfh81W5+8mqWil2kAip1CSg3927BY9d2Yfa6XEa8rA9PRY6nUpeAc316Mn+5vhtzN+7hlonzOHikyOtIIrWGSl0C0tXdk/jr4DQWbtnP0JfmkVegYhcBlboEsMu7JDJuSBrLt+cxZMIc9h0q9DqSiOdU6hLQLjnnLMYPS2fNrnwGvziH3HxdeSp1mxbJkKDw1bpcRr6SSfPGUUwdlUGThvW+87hzjryCIrbsPcyWvYfZvOcwW/cexgzuG9CeRlHhHiUXqZpTLZKhUpegMXfDHm6dlElCgwhG9j+b7H1lxb15T1mRHzzy3ZEy8fUjyCsopENiQ6aMyqBhvTCPkotUnkpd6pT5m/cxfOI8Dh4tJtwXQvPYSFJio777FRdFcuMooiNC+XzlLsZMmU+npBgmj8ygfoRmo5baTaUudU5eQRGHjhbTtGE9fCF2yu0/XraTO6YuIC2lEa/c2ouocBW71F5ao1TqnJjIMJo1iqxUoQMM6HQWT9/Yjfmb9zFyUhYFhbpSVQKXSl0EuKJLM568vhtzNu5h9GRNQSCBS6UuUu7q7kn86cddmLU2l9tfW6DZICUgqdRFjnF9ejIPD+rMv1btZuzUBZq/XQKOSl3kODdlpPB/V53Dpyt2cfcbiyhWsUsAqVSpm9kAM1ttZuvM7P6TbNfTzErM7Fr/RRSpebf0bcmvL+/AB0t38Iu3FlNS6s0oMZGqOuXYLTPzAX8DLgaygUwzm+GcW1HBdn8CPqmOoCI1bVT/syksKeWxj1cT6gvhsR93IaSSI2pEvFKZAbm9gHXOuQ0AZvYGMBBYcdx2dwLvAD39mlDEQ7ef35qiYsdfPlvDml0HuaJLIpd1SiQ5NsrraCIVqkypJwFbj7mdDWQcu4GZJQGDgAs5Samb2WhgNEBKSkpVs4p44q6LWpPQIILX523h4Q9X8fCHqzinWUMu63QWAzol0rpJfa8jivxHZUq9or83jz/B+BRwn3OuxOzEf54658YD46HsitJKZhTxlJlxU0YKN2WksHXvYT5etpOPlu3gz5+u4c+frqFNk/r/KfgOiQ042e+ASHU75TQBZtYHeMg5d2n57f8FcM49csw2G/lv+ccDh4HRzrnpJ3peTRMggW5n3hE+Wb6TD5fuIHPTXkodtIiLYmDXZtxxYWsiQn1eR5QgdMZzv5hZKLAGuAjYBmQCNznnlp9g+0nAP5xzb5/seVXqEkxy84/y6fJdfLRsB7PW5tK3VRwvDO1BA838KH52xnO/OOeKgbGUjWpZCbzpnFtuZmPMbIz/oooErvj6EdyUkcLkkRn8+bquzN24l8EvziHnoBbtON7R4hLu+fsi3p6f7XWUoKRZGkWqwb9W7eL21xbQtGE9Jt+aQUqcRst863+nLeH1eVvxhRhTRmbQp1Wc15ECimZpFPHAhe2bMvUnvckrKOKa575m+fY8ryPVClPnbuH1eVu59dxUWsZFccfUBWTvO+x1rKCiUhepJmkpjXl7TB/CfcYNL8zh6/W5Xkfy1IIt+3hwxjLOa5vAA5d34MVh6RSVlPLTyfM13bEfqdRFqlHrJg145/a+JMbUY/jETD5cusPrSJ7YffAIt02ZT2JMJM/c2B1fiHF2Qn2eubE7K3Yc4L53luDVqeBgo1IXqWaJMZG8NaYPnZvHcMfUBUyZs9nrSDWqsLiUO15bwIGCYl4Y2oOYqP+OCLqgfRN+cUk7ZizezouzNniYMnio1EVqQKOocKaMzODCdk349fRl/OWfazx7Z+qc4/kv1nPuo//i8U9WsSOvoFpf748frCBz0z7+dG0XOiQ2/N7jt5/fiss7J/LoR6v4Yk1OtWapC1TqIjUkMtzHC0N7cF2P5jz9+VoemL6sxmd/dM7xyEerePSjVUSF+xg3cz39/vRv7nhtAfM27vX7fzRvz8/mlW8285P+qVzVtVmF25gZj1/XhbZNG3Dn1AVsyj3k1wx1jUpdpAaF+kJ47Nou3H5+K6bO3cKISZnk5tfMWPbiklLue2cJ47/cwLA+Lfjk7h/w5b0XMLJfKrPW5nD9C99w+TOzeTNzq1+W81uancev3l1K31Zx3Deg/Um3jQoP5cVh6YSEGKMnZ3HoaPEZv35dpXHqIh6ZOncLD72/nJjIMP5yfTf6tYmvttc6UlTCz95YyCfLd3HXRW2454dtvjNHzeHCYqYv3M6krzeyZlc+jaPCuLFXCkN7t6BZo8gqv96e/KNc9exXAMwYey5x9SMq9XNfrctl6EtzuaTjWTx3c5rm0anAGU8TUF1U6iKwaucBxk5dyPqcfMac14qfX9yWMJ9//4DOP1rM6Fez+Hr9Hh68siMjzk094bbOOb7ZsIdJX23is5W7MDMu6diUQd2T6NMqrlLTHhSXlDJs4jyyNu/jnTF96dw8pkp5J8zawB8+WMn/XNyWOy9qU6WfrQtU6iK1XEFhCb/7x3Jen7eVbsmN+Ovg7n6br33voUKGvzyP5dsP8Pi1XbgmrXmlf3br3sNMmbOZNzK3kldQhC/E6J7ciP5tEujfNp4uSTGEVvAf0B8/WMGLszby5+u6cm2Pyr/et5xz/PzNxUxftI0Xh6bzw45Nq/wcwUylLhIgPliyg/unLQEHD1/TmStP8MFiZW3fX8DQl+aSva+Av92UdtrleLS4hAWb9zNrbQ6z1+WydFsezkGDeqGc2yqe/m3j6d86gZS4KGYs3s5dry9kWJ8W/G5gp9POfqSohOue/4ZNuYd4945zNWf9MVTqIgFk697D3PXGQhZu2c8N6ck8eFVHosIrs+zBd63PyWfohLkcPFLMhFvSyTjbf/Or7D1UyFfrcpm9NpdZa3PYnncEKJt2eNeBI3ROiuG1Ub0JDz2z00jb9xdw5V9nU79eKE/f2J1uyY38kD7wqdRFAkxRSSlPfbaGcTPXc3Z8NM/elFbh+O4TWbYtj1smzgPglVt70Smpaue0q8I5x/qcQ8xem8OstbnkHirkxWE9aNKgnl+ef/7mfdw2ZT45+Ue5sWcKv7y0HY2jw/3y3IFKpS4SoL5al8vdf19EXkERd13YmuTYKELM8IXYf/71hUCI/fd2bv5RHnh3GTGRYUwe2YuzEwL/tMXBI0U8/dlaXv56Ew3rhXL/Ze25rkdynV0EXKUuEsD25B/lF28t5t+rK3+lZesm9Zk8sheJMVUfilibrdp5gN9OX868TXtJS2nE76/uxDnNqu+vkNpKpS4S4JxzZO8roLCklNJSR4lzlJQ6SkuhxDlKnSu7v9RR6qBL8xiiI6p+Hj4QOOeYtmAbj3y0kr2HChnWpyX3XNyWmMi6s8KUSl1Egk5eQRFPfLqaKXM2Exsdwa9+1J5B3ZPqxMVKWiRDRIJOTGQYvxvYiRlj+9G8cSQ/f3MxN4yfw7rd+V5H85xKXUQCVqekGKbd1pdHr+nMml0HueKvs5g8Z3OdnptdpS4iAS0kxLixVwqf3vMDeqXG8Zvpyxj1SlaNTZRW26jURSQoNGlQj0nDe/LglR2ZtS6XAU/NYubq3V7HqnEqdREJGiEhxohzU8tmhowOZ/jLmTw0Y7lfphIOFCp1EQk67c9qyHtjz2XEuS2Z9PUmBj77Fat2HvA6Vo1QqYtIUKoX5uPBK89h0oie7DlUyFXPfsXE2RspPcVqU/lHi1m2LY/3F2/nuZnrWb49r4YS+0elxqmb2QDgacAHTHDOPXrc40OA+8pv5gO3OecWn+w5NU5dRGrKnvyj3PfOEj5buZv+beJ5eFBnDhUWsyn3EBtzD7MxN59NuYfZuOcQOQe/+wFruC+EX1/RgaG9W9SKcfBnfPGRmfmANcDFQDaQCQx2zq04Zpu+wErn3D4zuwx4yDmXcbLnVamLSE1yzvHa3C384YMVHCkq/c5j8fUjSI2PIjU+mpbx0aTGRZOaEE1MZBgPvLuMf63aXbY49o87V2qhkOp0qlKvzLXEvYB1zrkN5U/4BjAQ+E+pO+e+Pmb7OUDVZ8YXEalGZsbNvVvQ++w4Plu5i2aNIkmNi6ZlfNRJi3rCsHTGz9rA45+sZvn2PP42JK1WzzlTmVJPArYeczsbONm78JHARxU9YGajgdEAKSkplYwoIuI/rZvUr9KiGyEhxpjzWtGjRWPGTl3AoHFf8+CVHbmpV0qtOB1zvMp8UFpR6grP2ZjZBZSV+n0VPe6cG++cS3fOpSckJFQ+pYiIx3q2jOXDu/qTkRrLA+8u4+6/L+LQ0WKvY31PZUo9G0g+5nZzYPvxG5lZF2ACMNA5t8c/8UREao+4+hG8MqIX/3NxW95fvJ0rn51d64ZKVqbUM4E2ZpZqZuHAjcCMYzcwsxRgGjDUObfG/zFFRGqHkBDjzovaMGVUBgePFHP1377izaytp/7BGnLKUnfOFQNjgU+AlcCbzrnlZjbGzMaUb/ZbIA4YZ2aLzEzDWkQkqPVtFc8Hd/Wje3Jjfvn2Eu56fSGLt+73fDIxzacuInIGSkodT3++ludnrqewpJTU+Giu7pbE1d2b0SIu2u+vp0UyRERqQF5BER8v28G7C7cxZ8NeANJSGjGoexKXd2lGrJ8WzFapi4jUsO37C5ixeDvTF25j1c6DhIYY57VN4OruSfywQ1Miw32n/dwqdRERD63ccYDpi7bx3sLt7DxwhOhwH/dc3JZR/c8+refzxxWlIiJymjokNqRDYkN+eWl75m7cw3sLt5MYE1ltr6dSFxGpAb4Qo2+rePq2iq/W19HUuyIiQUSlLiISRFTqIiJBRKUuIhJEVOoiIkFEpS4iEkRU6iIiQUSlLiISRDybJsDMcoDNp/nj8UCuH+PUBsG2T8G2PxB8+xRs+wPBt08V7U8L59wJl47zrNTPhJllnWzug0AUbPsUbPsDwbdPwbY/EHz7dDr7o9MvIiJBRKUuIhJEArXUx3sdoBoE2z4F2/5A8O1TsO0PBN8+VXl/AvKcuoiIVCxQ36mLiEgFVOoiIkEk4ErdzAaY2WozW2dm93udxx/MbJOZLTWzRWYWcGv8mdlEM9ttZsuOuS/WzP5pZmvL/23sZcaqOsE+PWRm28qP0yIz+5GXGavCzJLN7N9mttLMlpvZz8rvD8jjdJL9CeRjVM/M5pnZ4vJ9+r/y+6t0jALqnLqZ+YA1wMVANpAJDHbOrfA02Bkys01AunMuIC+aMLMfAPnAq865TuX3PQbsdc49Wv6fb2Pn3H1e5qyKE+zTQ0C+c+7PXmY7HWaWCCQ65xaYWQNgPnA1MJwAPE4n2Z/rCdxjZEC0cy7fzMKA2cDPgGuowjEKtHfqvYB1zrkNzrlC4A1goMeZ6jzn3JfA3uPuHgi8Uv79K5T9wgWME+xTwHLO7XDOLSj//iCwEkgiQI/TSfYnYLky+eU3w8q/HFU8RoFW6knA1mNuZxPgB7KcAz41s/lmNtrrMH7S1Dm3A8p+AYEmHufxl7FmtqT89ExAnKo4npm1BLoDcwmC43Tc/kAAHyMz85nZImA38E/nXJWPUaCVulVwX+CcPzqxc51zacBlwB3lf/pL7fMc0AroBuwAnvA0zWkws/rAO8DdzrkDXuc5UxXsT0AfI+dciXOuG9Ac6GVmnar6HIFW6tlA8jG3mwPbPcriN8657eX/7gbepew0U6DbVX7e89vzn7s9znPGnHO7yn/pSoEXCbDjVH6e9h3gNefctPK7A/Y4VbQ/gX6MvuWc2w/MBAZQxWMUaKWeCbQxs1QzCwduBGZ4nOmMmFl0+Qc9mFk0cAmw7OQ/FRBmALeUf38L8J6HWfzi21+scoMIoONU/iHcS8BK59yTxzwUkMfpRPsT4McowcwalX8fCfwQWEUVj1FAjX4BKB+i9BTgAyY65/7obaIzY2ZnU/buHCAUmBpo+2RmrwPnUzZN6C7gQWA68CaQAmwBrnPOBcwHjyfYp/Mp+7PeAZuAn357rrO2M7N+wCxgKVBafvevKDsPHXDH6ST7M5jAPUZdKPsg1EfZG+43nXO/M7M4qnCMAq7URUTkxALt9IuIiJyESl1EJIio1EVEgohKXUQkiKjURUSCiEpdRCSIqNRFRILI/wN6OPT2n0UVLgAAAABJRU5ErkJggg==\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.plot(rnn_history.history['loss'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "6/6 [==============================] - 1s 26ms/step - loss: 0.6261 - accuracy: 0.8000\n"
     ]
    }
   ],
   "source": [
    "accuracy = rnn_model.evaluate(X_test, y_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[2, 2, 3, 1, 2, 3, 1, 3, 1, 1, 1, 2, 1, 1, 3, 1, 1, 2, 3, 1, 1, 3, 3, 2, 2, 2, 1, 2, 2, 3, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 3, 2, 3, 2, 3, 3, 3, 1, 3, 1, 1, 2, 1, 3, 3, 1, 2, 1, 2, 1, 2, 1, 3, 3, 2, 2, 1, 1, 3, 3, 3, 1, 2, 2, 3, 2, 1, 3, 3, 1, 1, 1, 3, 2, 3, 2, 1, 3, 2, 2, 3, 2, 3, 3, 3, 2, 2, 1, 1, 1, 1, 3, 2, 1, 1, 3, 2, 3, 2, 2, 2, 1, 1, 2, 2, 2, 2, 1, 2, 1, 2, 3, 2, 3, 1, 2, 1, 2, 3, 3, 2, 1, 3, 3, 3, 3, 3, 2, 3, 1, 3, 2, 2, 2, 1, 1, 2, 1, 1, 2, 2, 2, 2, 1, 1, 1, 2, 1, 1, 2, 3, 2, 1, 3, 3, 2, 1, 2, 1, 3, 1, 3, 3, 3, 3, 1]\n",
      "[2, 2, 1, 1, 2, 3, 1, 3, 1, 1, 1, 2, 1, 1, 3, 1, 1, 2, 3, 1, 1, 3, 3, 2, 2, 2, 1, 1, 2, 3, 2, 2, 2, 1, 2, 3, 2, 2, 2, 2, 2, 2, 1, 2, 3, 2, 3, 2, 3, 3, 3, 1, 3, 2, 2, 2, 3, 3, 3, 1, 2, 2, 2, 1, 1, 1, 3, 3, 2, 2, 3, 2, 3, 3, 3, 1, 2, 2, 3, 2, 1, 3, 3, 1, 2, 1, 3, 3, 3, 2, 1, 3, 2, 2, 2, 1, 3, 1, 1, 2, 2, 1, 1, 1, 1, 3, 2, 1, 2, 3, 2, 3, 2, 2, 1, 1, 2, 2, 2, 1, 2, 1, 2, 1, 2, 3, 1, 3, 2, 2, 1, 2, 3, 3, 2, 1, 3, 3, 3, 3, 2, 2, 3, 3, 3, 2, 2, 2, 2, 1, 3, 3, 2, 2, 2, 2, 2, 1, 2, 1, 2, 1, 3, 2, 3, 2, 3, 3, 1, 2, 1, 1, 1, 3, 3, 1, 3, 1, 3, 1]\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<AxesSubplot:>"
      ]
     },
     "execution_count": 43,
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
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "predictions = rnn_model.predict(X_test)\n",
    "y_pred = [p.argmax() for p in predictions]\n",
    "print(y_pred)\n",
    "y_true = [y.argmax() for y in y_test]\n",
    "print(y_true)\n",
    "conf_mat = confusion_matrix(y_true,y_pred)\n",
    "df_conf = pd.DataFrame(conf_mat,index=[i for i in 'Right Left Passive'.split()],\n",
    "                      columns = [i for i in 'Right Left Passive'.split()])\n",
    "plt.figure(figsize = (10,7))\n",
    "sn.heatmap(df_conf, annot=True,cmap='YlGnBu')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING:absl:Found untraced functions such as lstm_cell_layer_call_and_return_conditional_losses, lstm_cell_layer_call_fn, lstm_cell_1_layer_call_and_return_conditional_losses, lstm_cell_1_layer_call_fn, lstm_cell_2_layer_call_and_return_conditional_losses while saving (showing 5 of 15). These functions will not be directly callable after loading.\n",
      "WARNING:absl:Found untraced functions such as lstm_cell_layer_call_and_return_conditional_losses, lstm_cell_layer_call_fn, lstm_cell_1_layer_call_and_return_conditional_losses, lstm_cell_1_layer_call_fn, lstm_cell_2_layer_call_and_return_conditional_losses while saving (showing 5 of 15). These functions will not be directly callable after loading.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "INFO:tensorflow:Assets written to: C:/Users/nelso/Google Drive/School/MIT 2020-2021/6.s191/EEG_analysis_RNNandCNN\\assets\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "INFO:tensorflow:Assets written to: C:/Users/nelso/Google Drive/School/MIT 2020-2021/6.s191/EEG_analysis_RNNandCNN\\assets\n"
     ]
    }
   ],
   "source": [
    "rnn_model.save('C:/Users/nelso/Google Drive/School/MIT 2020-2021/6.s191/EEG_analysis_RNNandCNN')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 45,
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
   "execution_count": 46,
   "metadata": {},
   "outputs": [],
   "source": [
    "cnn_model = eegCNN(X_train.shape[1],X_train.shape[2],y_train.shape[1])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 47,
   "metadata": {},
   "outputs": [],
   "source": [
    "cnn_model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 48,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1/20\n",
      "15/15 [==============================] - 1s 10ms/step - loss: 2.6187 - accuracy: 0.3935\n",
      "Epoch 2/20\n",
      "15/15 [==============================] - 0s 10ms/step - loss: 0.6941 - accuracy: 0.7037\n",
      "Epoch 3/20\n",
      "15/15 [==============================] - 0s 11ms/step - loss: 0.4924 - accuracy: 0.8134\n",
      "Epoch 4/20\n",
      "15/15 [==============================] - 0s 10ms/step - loss: 0.2582 - accuracy: 0.9171\n",
      "Epoch 5/20\n",
      "15/15 [==============================] - 0s 10ms/step - loss: 0.1668 - accuracy: 0.9557\n",
      "Epoch 6/20\n",
      "15/15 [==============================] - 0s 10ms/step - loss: 0.1120 - accuracy: 0.9762\n",
      "Epoch 7/20\n",
      "15/15 [==============================] - 0s 10ms/step - loss: 0.0723 - accuracy: 0.9978\n",
      "Epoch 8/20\n",
      "15/15 [==============================] - 0s 10ms/step - loss: 0.0455 - accuracy: 1.0000\n",
      "Epoch 9/20\n",
      "15/15 [==============================] - 0s 10ms/step - loss: 0.0309 - accuracy: 1.0000\n",
      "Epoch 10/20\n",
      "15/15 [==============================] - 0s 10ms/step - loss: 0.0247 - accuracy: 1.0000\n",
      "Epoch 11/20\n",
      "15/15 [==============================] - 0s 10ms/step - loss: 0.0186 - accuracy: 1.0000\n",
      "Epoch 12/20\n",
      "15/15 [==============================] - 0s 10ms/step - loss: 0.0151 - accuracy: 1.0000\n",
      "Epoch 13/20\n",
      "15/15 [==============================] - 0s 10ms/step - loss: 0.0123 - accuracy: 1.0000\n",
      "Epoch 14/20\n",
      "15/15 [==============================] - 0s 10ms/step - loss: 0.0097 - accuracy: 1.0000\n",
      "Epoch 15/20\n",
      "15/15 [==============================] - 0s 10ms/step - loss: 0.0083 - accuracy: 1.0000\n",
      "Epoch 16/20\n",
      "15/15 [==============================] - 0s 10ms/step - loss: 0.0073 - accuracy: 1.0000\n",
      "Epoch 17/20\n",
      "15/15 [==============================] - 0s 11ms/step - loss: 0.0060 - accuracy: 1.0000\n",
      "Epoch 18/20\n",
      "15/15 [==============================] - 0s 11ms/step - loss: 0.0047 - accuracy: 1.0000\n",
      "Epoch 19/20\n",
      "15/15 [==============================] - 0s 13ms/step - loss: 0.0046 - accuracy: 1.0000\n",
      "Epoch 20/20\n",
      "15/15 [==============================] - 0s 14ms/step - loss: 0.0041 - accuracy: 1.0000\n"
     ]
    }
   ],
   "source": [
    "cnn_history = cnn_model.fit(X_train, y_train, epochs= 20, batch_size = 50)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 49,
   "metadata": {
    "scrolled": false
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[<matplotlib.lines.Line2D at 0x12d6af536a0>]"
      ]
     },
     "execution_count": 49,
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
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.plot(cnn_history.history['accuracy'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 50,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[<matplotlib.lines.Line2D at 0x12d6b002730>]"
      ]
     },
     "execution_count": 50,
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
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.plot(cnn_history.history['loss'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 51,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "6/6 [==============================] - 0s 4ms/step - loss: 0.4431 - accuracy: 0.8833\n"
     ]
    }
   ],
   "source": [
    "accuracy = cnn_model.evaluate(X_test, y_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 52,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[2, 2, 1, 1, 2, 2, 1, 3, 1, 1, 1, 2, 3, 1, 3, 1, 1, 2, 3, 1, 1, 3, 3, 2, 3, 2, 1, 1, 2, 3, 2, 2, 2, 1, 2, 1, 2, 2, 2, 2, 2, 2, 1, 2, 3, 2, 3, 2, 3, 3, 3, 1, 3, 2, 2, 2, 3, 3, 3, 1, 2, 2, 2, 1, 1, 1, 3, 3, 1, 2, 3, 2, 3, 3, 3, 1, 2, 2, 3, 2, 1, 3, 3, 1, 2, 1, 3, 2, 3, 2, 1, 3, 2, 2, 2, 2, 2, 1, 3, 1, 2, 1, 1, 1, 1, 3, 2, 1, 2, 3, 2, 3, 2, 2, 1, 1, 3, 2, 2, 1, 2, 1, 2, 1, 2, 3, 1, 3, 2, 2, 1, 2, 3, 3, 2, 1, 2, 3, 3, 3, 2, 2, 3, 3, 1, 2, 2, 2, 2, 1, 3, 3, 1, 2, 2, 2, 2, 1, 3, 1, 2, 3, 3, 2, 3, 2, 3, 3, 3, 2, 3, 2, 1, 3, 3, 3, 3, 1, 1, 1]\n",
      "[2, 2, 1, 1, 2, 3, 1, 3, 1, 1, 1, 2, 1, 1, 3, 1, 1, 2, 3, 1, 1, 3, 3, 2, 2, 2, 1, 1, 2, 3, 2, 2, 2, 1, 2, 3, 2, 2, 2, 2, 2, 2, 1, 2, 3, 2, 3, 2, 3, 3, 3, 1, 3, 2, 2, 2, 3, 3, 3, 1, 2, 2, 2, 1, 1, 1, 3, 3, 2, 2, 3, 2, 3, 3, 3, 1, 2, 2, 3, 2, 1, 3, 3, 1, 2, 1, 3, 3, 3, 2, 1, 3, 2, 2, 2, 1, 3, 1, 1, 2, 2, 1, 1, 1, 1, 3, 2, 1, 2, 3, 2, 3, 2, 2, 1, 1, 2, 2, 2, 1, 2, 1, 2, 1, 2, 3, 1, 3, 2, 2, 1, 2, 3, 3, 2, 1, 3, 3, 3, 3, 2, 2, 3, 3, 3, 2, 2, 2, 2, 1, 3, 3, 2, 2, 2, 2, 2, 1, 2, 1, 2, 1, 3, 2, 3, 2, 3, 3, 1, 2, 1, 1, 1, 3, 3, 1, 3, 1, 3, 1]\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<AxesSubplot:>"
      ]
     },
     "execution_count": 52,
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
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "predictions = cnn_model.predict(X_test)\n",
    "y_pred = [p.argmax() for p in predictions]\n",
    "print(y_pred)\n",
    "y_true = [y.argmax() for y in y_test]\n",
    "print(y_true)\n",
    "conf_mat = confusion_matrix(y_true,y_pred)\n",
    "df_conf = pd.DataFrame(conf_mat,index=[i for i in 'Right Left Passive'.split()],\n",
    "                      columns = [i for i in 'Right Left Passive'.split()])\n",
    "plt.figure(figsize = (10,7))\n",
    "sn.heatmap(df_conf, annot=True,cmap='YlGnBu')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 53,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "INFO:tensorflow:Assets written to: C:/Users/nelso/Google Drive/School/MIT 2020-2021/6.s191/EEG_analysis_RNNandCNN\\assets\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "INFO:tensorflow:Assets written to: C:/Users/nelso/Google Drive/School/MIT 2020-2021/6.s191/EEG_analysis_RNNandCNN\\assets\n"
     ]
    }
   ],
   "source": [
    "#Saving model in Keras:\n",
    "cnn_model.save('C:/Users/nelso/Google Drive/School/MIT 2020-2021/6.s191/EEG_analysis_RNNandCNN')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 54,
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
   "execution_count": 55,
   "metadata": {},
   "outputs": [],
   "source": [
    "c_rnnModel = eegConvRNN(X_train.shape[1],X_train.shape[2],y_train.shape[1])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 56,
   "metadata": {},
   "outputs": [],
   "source": [
    "c_rnnModel.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 57,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1/100\n",
      "15/15 [==============================] - 6s 105ms/step - loss: 1.3540 - accuracy: 0.3183\n",
      "Epoch 2/100\n",
      "15/15 [==============================] - 2s 106ms/step - loss: 1.2842 - accuracy: 0.3409\n",
      "Epoch 3/100\n",
      "15/15 [==============================] - 2s 107ms/step - loss: 1.2701 - accuracy: 0.2909\n",
      "Epoch 4/100\n",
      "15/15 [==============================] - 2s 104ms/step - loss: 1.2026 - accuracy: 0.3096\n",
      "Epoch 5/100\n",
      "15/15 [==============================] - 2s 105ms/step - loss: 1.1544 - accuracy: 0.4780\n",
      "Epoch 6/100\n",
      "15/15 [==============================] - 2s 105ms/step - loss: 1.1089 - accuracy: 0.5951\n",
      "Epoch 7/100\n",
      "15/15 [==============================] - 2s 104ms/step - loss: 1.0890 - accuracy: 0.6204\n",
      "Epoch 8/100\n",
      "15/15 [==============================] - 2s 103ms/step - loss: 1.0616 - accuracy: 0.6348\n",
      "Epoch 9/100\n",
      "15/15 [==============================] - 2s 103ms/step - loss: 1.0542 - accuracy: 0.6257\n",
      "Epoch 10/100\n",
      "15/15 [==============================] - 2s 103ms/step - loss: 1.0333 - accuracy: 0.8319\n",
      "Epoch 11/100\n",
      "15/15 [==============================] - 2s 104ms/step - loss: 1.0221 - accuracy: 0.8698\n",
      "Epoch 12/100\n",
      "15/15 [==============================] - 2s 103ms/step - loss: 0.9961 - accuracy: 0.8182\n",
      "Epoch 13/100\n",
      "15/15 [==============================] - 2s 102ms/step - loss: 0.9929 - accuracy: 0.8603\n",
      "Epoch 14/100\n",
      "15/15 [==============================] - 2s 103ms/step - loss: 0.9840 - accuracy: 0.8394\n",
      "Epoch 15/100\n",
      "15/15 [==============================] - 2s 105ms/step - loss: 0.9625 - accuracy: 0.8718\n",
      "Epoch 16/100\n",
      "15/15 [==============================] - 2s 106ms/step - loss: 0.9633 - accuracy: 0.8144\n",
      "Epoch 17/100\n",
      "15/15 [==============================] - 2s 103ms/step - loss: 0.9339 - accuracy: 0.9023\n",
      "Epoch 18/100\n",
      "15/15 [==============================] - 2s 101ms/step - loss: 0.9205 - accuracy: 0.8924\n",
      "Epoch 19/100\n",
      "15/15 [==============================] - 2s 103ms/step - loss: 0.9110 - accuracy: 0.8949\n",
      "Epoch 20/100\n",
      "15/15 [==============================] - 2s 105ms/step - loss: 0.8882 - accuracy: 0.9343\n",
      "Epoch 21/100\n",
      "15/15 [==============================] - 2s 106ms/step - loss: 0.8901 - accuracy: 0.9086\n",
      "Epoch 22/100\n",
      "15/15 [==============================] - 2s 105ms/step - loss: 0.8730 - accuracy: 0.9127\n",
      "Epoch 23/100\n",
      "15/15 [==============================] - 2s 106ms/step - loss: 0.8617 - accuracy: 0.8857\n",
      "Epoch 24/100\n",
      "15/15 [==============================] - 2s 102ms/step - loss: 0.8568 - accuracy: 0.8881\n",
      "Epoch 25/100\n",
      "15/15 [==============================] - 2s 102ms/step - loss: 0.8225 - accuracy: 0.9022\n",
      "Epoch 26/100\n",
      "15/15 [==============================] - 2s 102ms/step - loss: 0.8287 - accuracy: 0.8879\n",
      "Epoch 27/100\n",
      "15/15 [==============================] - 2s 102ms/step - loss: 0.8202 - accuracy: 0.8863\n",
      "Epoch 28/100\n",
      "15/15 [==============================] - 2s 102ms/step - loss: 0.8069 - accuracy: 0.8940\n",
      "Epoch 29/100\n",
      "15/15 [==============================] - 2s 102ms/step - loss: 0.7956 - accuracy: 0.8981\n",
      "Epoch 30/100\n",
      "15/15 [==============================] - 2s 102ms/step - loss: 0.7611 - accuracy: 0.9146\n",
      "Epoch 31/100\n",
      "15/15 [==============================] - 2s 101ms/step - loss: 0.7447 - accuracy: 0.9104\n",
      "Epoch 32/100\n",
      "15/15 [==============================] - 2s 102ms/step - loss: 0.7401 - accuracy: 0.9001\n",
      "Epoch 33/100\n",
      "15/15 [==============================] - 2s 103ms/step - loss: 0.7469 - accuracy: 0.9038\n",
      "Epoch 34/100\n",
      "15/15 [==============================] - 2s 102ms/step - loss: 0.7510 - accuracy: 0.8532\n",
      "Epoch 35/100\n",
      "15/15 [==============================] - 2s 102ms/step - loss: 0.7206 - accuracy: 0.8990\n",
      "Epoch 36/100\n",
      "15/15 [==============================] - 2s 104ms/step - loss: 0.7091 - accuracy: 0.9145\n",
      "Epoch 37/100\n",
      "15/15 [==============================] - 2s 103ms/step - loss: 0.7050 - accuracy: 0.9097\n",
      "Epoch 38/100\n",
      "15/15 [==============================] - 2s 103ms/step - loss: 0.6993 - accuracy: 0.8830\n",
      "Epoch 39/100\n",
      "15/15 [==============================] - 2s 102ms/step - loss: 0.7321 - accuracy: 0.8766\n",
      "Epoch 40/100\n",
      "15/15 [==============================] - 2s 103ms/step - loss: 0.7232 - accuracy: 0.8791\n",
      "Epoch 41/100\n",
      "15/15 [==============================] - 2s 103ms/step - loss: 0.6845 - accuracy: 0.9056\n",
      "Epoch 42/100\n",
      "15/15 [==============================] - 2s 104ms/step - loss: 0.6655 - accuracy: 0.9065\n",
      "Epoch 43/100\n",
      "15/15 [==============================] - 2s 103ms/step - loss: 0.6610 - accuracy: 0.9253\n",
      "Epoch 44/100\n",
      "15/15 [==============================] - 2s 106ms/step - loss: 0.6620 - accuracy: 0.9102\n",
      "Epoch 45/100\n",
      "15/15 [==============================] - 2s 103ms/step - loss: 0.6751 - accuracy: 0.9339\n",
      "Epoch 46/100\n",
      "15/15 [==============================] - 2s 103ms/step - loss: 0.6482 - accuracy: 0.9168\n",
      "Epoch 47/100\n",
      "15/15 [==============================] - 2s 102ms/step - loss: 0.6374 - accuracy: 0.9169\n",
      "Epoch 48/100\n",
      "15/15 [==============================] - 2s 103ms/step - loss: 0.6417 - accuracy: 0.9031\n",
      "Epoch 49/100\n",
      "15/15 [==============================] - 2s 103ms/step - loss: 0.6146 - accuracy: 0.9175\n",
      "Epoch 50/100\n",
      "15/15 [==============================] - 2s 103ms/step - loss: 0.6110 - accuracy: 0.9249\n",
      "Epoch 51/100\n",
      "15/15 [==============================] - 2s 102ms/step - loss: 0.6148 - accuracy: 0.9264\n",
      "Epoch 52/100\n",
      "15/15 [==============================] - 2s 103ms/step - loss: 0.5952 - accuracy: 0.9261\n",
      "Epoch 53/100\n",
      "15/15 [==============================] - 2s 103ms/step - loss: 0.5790 - accuracy: 0.9343\n",
      "Epoch 54/100\n",
      "15/15 [==============================] - 2s 102ms/step - loss: 0.5759 - accuracy: 0.9175\n",
      "Epoch 55/100\n",
      "15/15 [==============================] - 2s 102ms/step - loss: 0.5652 - accuracy: 0.9183\n",
      "Epoch 56/100\n",
      "15/15 [==============================] - 2s 104ms/step - loss: 0.5524 - accuracy: 0.9299\n",
      "Epoch 57/100\n",
      "15/15 [==============================] - 2s 103ms/step - loss: 0.6171 - accuracy: 0.8717\n",
      "Epoch 58/100\n",
      "15/15 [==============================] - 2s 102ms/step - loss: 0.5913 - accuracy: 0.8720\n",
      "Epoch 59/100\n",
      "15/15 [==============================] - 2s 102ms/step - loss: 0.6011 - accuracy: 0.8811\n",
      "Epoch 60/100\n",
      "15/15 [==============================] - 2s 102ms/step - loss: 0.5714 - accuracy: 0.8995\n",
      "Epoch 61/100\n",
      "15/15 [==============================] - 2s 102ms/step - loss: 0.5423 - accuracy: 0.9129\n",
      "Epoch 62/100\n",
      "15/15 [==============================] - 2s 102ms/step - loss: 0.5553 - accuracy: 0.8944\n",
      "Epoch 63/100\n",
      "15/15 [==============================] - 2s 102ms/step - loss: 0.5645 - accuracy: 0.8677\n",
      "Epoch 64/100\n",
      "15/15 [==============================] - 2s 102ms/step - loss: 0.5157 - accuracy: 0.9291\n",
      "Epoch 65/100\n",
      "15/15 [==============================] - 2s 102ms/step - loss: 0.5109 - accuracy: 0.9403\n",
      "Epoch 66/100\n",
      "15/15 [==============================] - 2s 102ms/step - loss: 0.5057 - accuracy: 0.9372\n",
      "Epoch 67/100\n",
      "15/15 [==============================] - 2s 104ms/step - loss: 0.4922 - accuracy: 0.9405\n",
      "Epoch 68/100\n",
      "15/15 [==============================] - 2s 103ms/step - loss: 0.5266 - accuracy: 0.9006\n",
      "Epoch 69/100\n",
      "15/15 [==============================] - 2s 103ms/step - loss: 0.5107 - accuracy: 0.9069\n",
      "Epoch 70/100\n",
      "15/15 [==============================] - 2s 103ms/step - loss: 0.4950 - accuracy: 0.9325\n",
      "Epoch 71/100\n",
      "15/15 [==============================] - 2s 102ms/step - loss: 0.5009 - accuracy: 0.9198\n",
      "Epoch 72/100\n",
      "15/15 [==============================] - 2s 103ms/step - loss: 0.4936 - accuracy: 0.9296\n",
      "Epoch 73/100\n",
      "15/15 [==============================] - 2s 102ms/step - loss: 0.4812 - accuracy: 0.9448\n",
      "Epoch 74/100\n",
      "15/15 [==============================] - 2s 102ms/step - loss: 0.4934 - accuracy: 0.9149\n",
      "Epoch 75/100\n",
      "15/15 [==============================] - 2s 102ms/step - loss: 0.5149 - accuracy: 0.8921\n",
      "Epoch 76/100\n",
      "15/15 [==============================] - 2s 102ms/step - loss: 0.5314 - accuracy: 0.8700\n",
      "Epoch 77/100\n",
      "15/15 [==============================] - 2s 103ms/step - loss: 0.6360 - accuracy: 0.7728\n",
      "Epoch 78/100\n",
      "15/15 [==============================] - 2s 102ms/step - loss: 0.5692 - accuracy: 0.8924\n",
      "Epoch 79/100\n",
      "15/15 [==============================] - 2s 102ms/step - loss: 0.5092 - accuracy: 0.9277\n",
      "Epoch 80/100\n",
      "15/15 [==============================] - 2s 102ms/step - loss: 0.4806 - accuracy: 0.9240\n",
      "Epoch 81/100\n",
      "15/15 [==============================] - 2s 102ms/step - loss: 0.5177 - accuracy: 0.8613\n",
      "Epoch 82/100\n",
      "15/15 [==============================] - 2s 102ms/step - loss: 0.4780 - accuracy: 0.9141\n",
      "Epoch 83/100\n",
      "15/15 [==============================] - 2s 102ms/step - loss: 0.5177 - accuracy: 0.8987\n",
      "Epoch 84/100\n",
      "15/15 [==============================] - 2s 102ms/step - loss: 0.4866 - accuracy: 0.9125\n",
      "Epoch 85/100\n",
      "15/15 [==============================] - 2s 102ms/step - loss: 0.4524 - accuracy: 0.9320\n",
      "Epoch 86/100\n",
      "15/15 [==============================] - 2s 103ms/step - loss: 0.4372 - accuracy: 0.9336\n",
      "Epoch 87/100\n",
      "15/15 [==============================] - 2s 102ms/step - loss: 0.4474 - accuracy: 0.9243\n",
      "Epoch 88/100\n",
      "15/15 [==============================] - 2s 102ms/step - loss: 0.4424 - accuracy: 0.9319\n",
      "Epoch 89/100\n",
      "15/15 [==============================] - 2s 102ms/step - loss: 0.4223 - accuracy: 0.9366\n",
      "Epoch 90/100\n",
      "15/15 [==============================] - 2s 102ms/step - loss: 0.4229 - accuracy: 0.9406\n",
      "Epoch 91/100\n",
      "15/15 [==============================] - 2s 102ms/step - loss: 0.4212 - accuracy: 0.9271\n",
      "Epoch 92/100\n",
      "15/15 [==============================] - 2s 102ms/step - loss: 0.4119 - accuracy: 0.9484\n",
      "Epoch 93/100\n",
      "15/15 [==============================] - 2s 102ms/step - loss: 0.3825 - accuracy: 0.9571\n",
      "Epoch 94/100\n",
      "15/15 [==============================] - 2s 105ms/step - loss: 0.3873 - accuracy: 0.9596\n",
      "Epoch 95/100\n",
      "15/15 [==============================] - 2s 104ms/step - loss: 0.3925 - accuracy: 0.9514\n",
      "Epoch 96/100\n",
      "15/15 [==============================] - 2s 103ms/step - loss: 0.3792 - accuracy: 0.9579\n",
      "Epoch 97/100\n",
      "15/15 [==============================] - 2s 103ms/step - loss: 0.3811 - accuracy: 0.9482\n",
      "Epoch 98/100\n",
      "15/15 [==============================] - 2s 104ms/step - loss: 0.3702 - accuracy: 0.9669\n",
      "Epoch 99/100\n",
      "15/15 [==============================] - 2s 103ms/step - loss: 0.3584 - accuracy: 0.9737\n",
      "Epoch 100/100\n",
      "15/15 [==============================] - 2s 103ms/step - loss: 0.3660 - accuracy: 0.9662\n"
     ]
    }
   ],
   "source": [
    "crnn_history = c_rnnModel.fit(X_train, y_train, epochs= 100, batch_size = 50)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 86,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Text(0.5, 0, 'Epochs')"
      ]
     },
     "execution_count": 86,
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
    "sn.lineplot(x=list(range(100)),y=crnn_history.history['loss'])\n",
    "plt.ylabel('Loss')\n",
    "plt.xlabel('Epochs')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 88,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Text(0.5, 0, 'Epochs')"
      ]
     },
     "execution_count": 88,
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
   "execution_count": 60,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[<matplotlib.lines.Line2D at 0x12d6c20a520>]"
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
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.plot(crnn_history.history['accuracy'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 61,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "6/6 [==============================] - 1s 29ms/step - loss: 0.5114 - accuracy: 0.8722\n"
     ]
    }
   ],
   "source": [
    "accuracy = c_rnnModel.evaluate(X_test, y_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 97,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[2, 2, 3, 1, 2, 3, 1, 3, 1, 1, 1, 2, 1, 1, 3, 1, 3, 2, 3, 1, 1, 3, 3, 2, 3, 2, 1, 1, 2, 3, 2, 2, 2, 1, 2, 3, 2, 2, 2, 2, 2, 2, 1, 2, 3, 2, 3, 2, 3, 3, 3, 1, 3, 2, 2, 2, 1, 3, 3, 1, 2, 2, 2, 1, 1, 1, 3, 3, 2, 2, 3, 2, 1, 3, 3, 1, 2, 2, 1, 2, 1, 3, 3, 1, 2, 1, 3, 3, 3, 2, 1, 3, 2, 3, 3, 2, 3, 1, 1, 3, 2, 1, 1, 1, 1, 3, 2, 3, 3, 3, 2, 3, 2, 2, 1, 1, 2, 2, 2, 1, 2, 1, 2, 1, 2, 3, 3, 3, 2, 2, 1, 2, 3, 3, 2, 1, 2, 3, 3, 3, 2, 2, 3, 1, 1, 2, 2, 2, 2, 1, 3, 3, 1, 2, 2, 2, 2, 1, 1, 1, 2, 1, 3, 2, 3, 2, 1, 3, 3, 2, 1, 1, 3, 3, 3, 3, 3, 1, 1, 1]\n",
      "[2, 2, 1, 1, 2, 3, 1, 3, 1, 1, 1, 2, 1, 1, 3, 1, 1, 2, 3, 1, 1, 3, 3, 2, 2, 2, 1, 1, 2, 3, 2, 2, 2, 1, 2, 3, 2, 2, 2, 2, 2, 2, 1, 2, 3, 2, 3, 2, 3, 3, 3, 1, 3, 2, 2, 2, 3, 3, 3, 1, 2, 2, 2, 1, 1, 1, 3, 3, 2, 2, 3, 2, 3, 3, 3, 1, 2, 2, 3, 2, 1, 3, 3, 1, 2, 1, 3, 3, 3, 2, 1, 3, 2, 2, 2, 1, 3, 1, 1, 2, 2, 1, 1, 1, 1, 3, 2, 1, 2, 3, 2, 3, 2, 2, 1, 1, 2, 2, 2, 1, 2, 1, 2, 1, 2, 3, 1, 3, 2, 2, 1, 2, 3, 3, 2, 1, 3, 3, 3, 3, 2, 2, 3, 3, 3, 2, 2, 2, 2, 1, 3, 3, 2, 2, 2, 2, 2, 1, 2, 1, 2, 1, 3, 2, 3, 2, 3, 3, 1, 2, 1, 1, 1, 3, 3, 1, 3, 1, 3, 1]\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<AxesSubplot:>"
      ]
     },
     "execution_count": 97,
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
    "print(y_pred)\n",
    "y_true = [y.argmax() for y in y_test]\n",
    "print(y_true)\n",
    "conf_mat = confusion_matrix(y_true,y_pred)\n",
    "df_conf = pd.DataFrame(conf_mat,index=[i for i in 'Right Left Passive'.split()],\n",
    "                      columns = [i for i in 'Right Left Passive'.split()])\n",
    "plt.figure(figsize = (10,7))\n",
    "sn.heatmap(df_conf, annot=True,cmap='Blues')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 75,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "{'precision': 0.8768387347630962, 'recall': 0.8722222222222222, 'f1-score': 0.873713588565121, 'support': 180}\n",
      "0.8722222222222222\n"
     ]
    }
   ],
   "source": [
    "from sklearn.metrics import classification_report\n",
    "class_metrics = classification_report (y_true,y_pred,output_dict=True)\n",
    "print(class_metrics['weighted avg'])\n",
    "print(class_metrics['accuracy'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "test_crnnModel = "
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
