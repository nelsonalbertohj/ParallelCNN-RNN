??
??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
?
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
?
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ??
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.7.02v2.7.0-rc1-69-gc256c071bb28??
?
eeg_cnn/conv1d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*&
shared_nameeeg_cnn/conv1d/kernel
?
)eeg_cnn/conv1d/kernel/Read/ReadVariableOpReadVariableOpeeg_cnn/conv1d/kernel*"
_output_shapes
:2*
dtype0
~
eeg_cnn/conv1d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*$
shared_nameeeg_cnn/conv1d/bias
w
'eeg_cnn/conv1d/bias/Read/ReadVariableOpReadVariableOpeeg_cnn/conv1d/bias*
_output_shapes
:2*
dtype0
?
eeg_cnn/conv1d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:2d*(
shared_nameeeg_cnn/conv1d_1/kernel
?
+eeg_cnn/conv1d_1/kernel/Read/ReadVariableOpReadVariableOpeeg_cnn/conv1d_1/kernel*"
_output_shapes
:2d*
dtype0
?
eeg_cnn/conv1d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*&
shared_nameeeg_cnn/conv1d_1/bias
{
)eeg_cnn/conv1d_1/bias/Read/ReadVariableOpReadVariableOpeeg_cnn/conv1d_1/bias*
_output_shapes
:d*
dtype0
?
eeg_cnn/conv1d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:d?*(
shared_nameeeg_cnn/conv1d_2/kernel
?
+eeg_cnn/conv1d_2/kernel/Read/ReadVariableOpReadVariableOpeeg_cnn/conv1d_2/kernel*#
_output_shapes
:d?*
dtype0
?
eeg_cnn/conv1d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*&
shared_nameeeg_cnn/conv1d_2/bias
|
)eeg_cnn/conv1d_2/bias/Read/ReadVariableOpReadVariableOpeeg_cnn/conv1d_2/bias*
_output_shapes	
:?*
dtype0
?
eeg_cnn/dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?	*'
shared_nameeeg_cnn/dense_1/kernel
?
*eeg_cnn/dense_1/kernel/Read/ReadVariableOpReadVariableOpeeg_cnn/dense_1/kernel*
_output_shapes
:	?	*
dtype0
?
eeg_cnn/dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameeeg_cnn/dense_1/bias
y
(eeg_cnn/dense_1/bias/Read/ReadVariableOpReadVariableOpeeg_cnn/dense_1/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
?
Adam/eeg_cnn/conv1d/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*-
shared_nameAdam/eeg_cnn/conv1d/kernel/m
?
0Adam/eeg_cnn/conv1d/kernel/m/Read/ReadVariableOpReadVariableOpAdam/eeg_cnn/conv1d/kernel/m*"
_output_shapes
:2*
dtype0
?
Adam/eeg_cnn/conv1d/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*+
shared_nameAdam/eeg_cnn/conv1d/bias/m
?
.Adam/eeg_cnn/conv1d/bias/m/Read/ReadVariableOpReadVariableOpAdam/eeg_cnn/conv1d/bias/m*
_output_shapes
:2*
dtype0
?
Adam/eeg_cnn/conv1d_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:2d*/
shared_name Adam/eeg_cnn/conv1d_1/kernel/m
?
2Adam/eeg_cnn/conv1d_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/eeg_cnn/conv1d_1/kernel/m*"
_output_shapes
:2d*
dtype0
?
Adam/eeg_cnn/conv1d_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*-
shared_nameAdam/eeg_cnn/conv1d_1/bias/m
?
0Adam/eeg_cnn/conv1d_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/eeg_cnn/conv1d_1/bias/m*
_output_shapes
:d*
dtype0
?
Adam/eeg_cnn/conv1d_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:d?*/
shared_name Adam/eeg_cnn/conv1d_2/kernel/m
?
2Adam/eeg_cnn/conv1d_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/eeg_cnn/conv1d_2/kernel/m*#
_output_shapes
:d?*
dtype0
?
Adam/eeg_cnn/conv1d_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*-
shared_nameAdam/eeg_cnn/conv1d_2/bias/m
?
0Adam/eeg_cnn/conv1d_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/eeg_cnn/conv1d_2/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/eeg_cnn/dense_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?	*.
shared_nameAdam/eeg_cnn/dense_1/kernel/m
?
1Adam/eeg_cnn/dense_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/eeg_cnn/dense_1/kernel/m*
_output_shapes
:	?	*
dtype0
?
Adam/eeg_cnn/dense_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_nameAdam/eeg_cnn/dense_1/bias/m
?
/Adam/eeg_cnn/dense_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/eeg_cnn/dense_1/bias/m*
_output_shapes
:*
dtype0
?
Adam/eeg_cnn/conv1d/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*-
shared_nameAdam/eeg_cnn/conv1d/kernel/v
?
0Adam/eeg_cnn/conv1d/kernel/v/Read/ReadVariableOpReadVariableOpAdam/eeg_cnn/conv1d/kernel/v*"
_output_shapes
:2*
dtype0
?
Adam/eeg_cnn/conv1d/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*+
shared_nameAdam/eeg_cnn/conv1d/bias/v
?
.Adam/eeg_cnn/conv1d/bias/v/Read/ReadVariableOpReadVariableOpAdam/eeg_cnn/conv1d/bias/v*
_output_shapes
:2*
dtype0
?
Adam/eeg_cnn/conv1d_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:2d*/
shared_name Adam/eeg_cnn/conv1d_1/kernel/v
?
2Adam/eeg_cnn/conv1d_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/eeg_cnn/conv1d_1/kernel/v*"
_output_shapes
:2d*
dtype0
?
Adam/eeg_cnn/conv1d_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*-
shared_nameAdam/eeg_cnn/conv1d_1/bias/v
?
0Adam/eeg_cnn/conv1d_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/eeg_cnn/conv1d_1/bias/v*
_output_shapes
:d*
dtype0
?
Adam/eeg_cnn/conv1d_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:d?*/
shared_name Adam/eeg_cnn/conv1d_2/kernel/v
?
2Adam/eeg_cnn/conv1d_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/eeg_cnn/conv1d_2/kernel/v*#
_output_shapes
:d?*
dtype0
?
Adam/eeg_cnn/conv1d_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*-
shared_nameAdam/eeg_cnn/conv1d_2/bias/v
?
0Adam/eeg_cnn/conv1d_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/eeg_cnn/conv1d_2/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/eeg_cnn/dense_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?	*.
shared_nameAdam/eeg_cnn/dense_1/kernel/v
?
1Adam/eeg_cnn/dense_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/eeg_cnn/dense_1/kernel/v*
_output_shapes
:	?	*
dtype0
?
Adam/eeg_cnn/dense_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_nameAdam/eeg_cnn/dense_1/bias/v
?
/Adam/eeg_cnn/dense_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/eeg_cnn/dense_1/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
?4
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?4
value?4B?4 B?4
?
	conv1
	pool1
	conv2
	pool2
	conv3
	pool3
flat
fc
		optimizer

	variables
trainable_variables
regularization_losses
	keras_api

signatures
h

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
R
	variables
trainable_variables
regularization_losses
	keras_api
h

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
R
	variables
 trainable_variables
!regularization_losses
"	keras_api
h

#kernel
$bias
%	variables
&trainable_variables
'regularization_losses
(	keras_api
R
)	variables
*trainable_variables
+regularization_losses
,	keras_api
R
-	variables
.trainable_variables
/regularization_losses
0	keras_api
h

1kernel
2bias
3	variables
4trainable_variables
5regularization_losses
6	keras_api
?
7iter

8beta_1

9beta_2
	:decay
;learning_ratemtmumvmw#mx$my1mz2m{v|v}v~v#v?$v?1v?2v?
8
0
1
2
3
#4
$5
16
27
8
0
1
2
3
#4
$5
16
27
 
?
<non_trainable_variables

=layers
>metrics
?layer_regularization_losses
@layer_metrics

	variables
trainable_variables
regularization_losses
 
RP
VARIABLE_VALUEeeg_cnn/conv1d/kernel'conv1/kernel/.ATTRIBUTES/VARIABLE_VALUE
NL
VARIABLE_VALUEeeg_cnn/conv1d/bias%conv1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
Anon_trainable_variables

Blayers
Cmetrics
Dlayer_regularization_losses
Elayer_metrics
	variables
trainable_variables
regularization_losses
 
 
 
?
Fnon_trainable_variables

Glayers
Hmetrics
Ilayer_regularization_losses
Jlayer_metrics
	variables
trainable_variables
regularization_losses
TR
VARIABLE_VALUEeeg_cnn/conv1d_1/kernel'conv2/kernel/.ATTRIBUTES/VARIABLE_VALUE
PN
VARIABLE_VALUEeeg_cnn/conv1d_1/bias%conv2/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
Knon_trainable_variables

Llayers
Mmetrics
Nlayer_regularization_losses
Olayer_metrics
	variables
trainable_variables
regularization_losses
 
 
 
?
Pnon_trainable_variables

Qlayers
Rmetrics
Slayer_regularization_losses
Tlayer_metrics
	variables
 trainable_variables
!regularization_losses
TR
VARIABLE_VALUEeeg_cnn/conv1d_2/kernel'conv3/kernel/.ATTRIBUTES/VARIABLE_VALUE
PN
VARIABLE_VALUEeeg_cnn/conv1d_2/bias%conv3/bias/.ATTRIBUTES/VARIABLE_VALUE

#0
$1

#0
$1
 
?
Unon_trainable_variables

Vlayers
Wmetrics
Xlayer_regularization_losses
Ylayer_metrics
%	variables
&trainable_variables
'regularization_losses
 
 
 
?
Znon_trainable_variables

[layers
\metrics
]layer_regularization_losses
^layer_metrics
)	variables
*trainable_variables
+regularization_losses
 
 
 
?
_non_trainable_variables

`layers
ametrics
blayer_regularization_losses
clayer_metrics
-	variables
.trainable_variables
/regularization_losses
PN
VARIABLE_VALUEeeg_cnn/dense_1/kernel$fc/kernel/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEeeg_cnn/dense_1/bias"fc/bias/.ATTRIBUTES/VARIABLE_VALUE

10
21

10
21
 
?
dnon_trainable_variables

elayers
fmetrics
glayer_regularization_losses
hlayer_metrics
3	variables
4trainable_variables
5regularization_losses
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
 
8
0
1
2
3
4
5
6
7

i0
j1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
4
	ktotal
	lcount
m	variables
n	keras_api
D
	ototal
	pcount
q
_fn_kwargs
r	variables
s	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

k0
l1

m	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

o0
p1

r	variables
us
VARIABLE_VALUEAdam/eeg_cnn/conv1d/kernel/mCconv1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUEAdam/eeg_cnn/conv1d/bias/mAconv1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/eeg_cnn/conv1d_1/kernel/mCconv2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUEAdam/eeg_cnn/conv1d_1/bias/mAconv2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/eeg_cnn/conv1d_2/kernel/mCconv3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUEAdam/eeg_cnn/conv1d_2/bias/mAconv3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUEAdam/eeg_cnn/dense_1/kernel/m@fc/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/eeg_cnn/dense_1/bias/m>fc/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEAdam/eeg_cnn/conv1d/kernel/vCconv1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUEAdam/eeg_cnn/conv1d/bias/vAconv1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/eeg_cnn/conv1d_1/kernel/vCconv2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUEAdam/eeg_cnn/conv1d_1/bias/vAconv2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/eeg_cnn/conv1d_2/kernel/vCconv3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUEAdam/eeg_cnn/conv1d_2/bias/vAconv3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUEAdam/eeg_cnn/dense_1/kernel/v@fc/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/eeg_cnn/dense_1/bias/v>fc/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_input_1Placeholder*,
_output_shapes
:??????????*
dtype0*!
shape:??????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1eeg_cnn/conv1d/kerneleeg_cnn/conv1d/biaseeg_cnn/conv1d_1/kerneleeg_cnn/conv1d_1/biaseeg_cnn/conv1d_2/kerneleeg_cnn/conv1d_2/biaseeg_cnn/dense_1/kerneleeg_cnn/dense_1/bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *,
f'R%
#__inference_signature_wrapper_29410
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename)eeg_cnn/conv1d/kernel/Read/ReadVariableOp'eeg_cnn/conv1d/bias/Read/ReadVariableOp+eeg_cnn/conv1d_1/kernel/Read/ReadVariableOp)eeg_cnn/conv1d_1/bias/Read/ReadVariableOp+eeg_cnn/conv1d_2/kernel/Read/ReadVariableOp)eeg_cnn/conv1d_2/bias/Read/ReadVariableOp*eeg_cnn/dense_1/kernel/Read/ReadVariableOp(eeg_cnn/dense_1/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp0Adam/eeg_cnn/conv1d/kernel/m/Read/ReadVariableOp.Adam/eeg_cnn/conv1d/bias/m/Read/ReadVariableOp2Adam/eeg_cnn/conv1d_1/kernel/m/Read/ReadVariableOp0Adam/eeg_cnn/conv1d_1/bias/m/Read/ReadVariableOp2Adam/eeg_cnn/conv1d_2/kernel/m/Read/ReadVariableOp0Adam/eeg_cnn/conv1d_2/bias/m/Read/ReadVariableOp1Adam/eeg_cnn/dense_1/kernel/m/Read/ReadVariableOp/Adam/eeg_cnn/dense_1/bias/m/Read/ReadVariableOp0Adam/eeg_cnn/conv1d/kernel/v/Read/ReadVariableOp.Adam/eeg_cnn/conv1d/bias/v/Read/ReadVariableOp2Adam/eeg_cnn/conv1d_1/kernel/v/Read/ReadVariableOp0Adam/eeg_cnn/conv1d_1/bias/v/Read/ReadVariableOp2Adam/eeg_cnn/conv1d_2/kernel/v/Read/ReadVariableOp0Adam/eeg_cnn/conv1d_2/bias/v/Read/ReadVariableOp1Adam/eeg_cnn/dense_1/kernel/v/Read/ReadVariableOp/Adam/eeg_cnn/dense_1/bias/v/Read/ReadVariableOpConst*.
Tin'
%2#	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *'
f"R 
__inference__traced_save_29798
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameeeg_cnn/conv1d/kerneleeg_cnn/conv1d/biaseeg_cnn/conv1d_1/kerneleeg_cnn/conv1d_1/biaseeg_cnn/conv1d_2/kerneleeg_cnn/conv1d_2/biaseeg_cnn/dense_1/kerneleeg_cnn/dense_1/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1Adam/eeg_cnn/conv1d/kernel/mAdam/eeg_cnn/conv1d/bias/mAdam/eeg_cnn/conv1d_1/kernel/mAdam/eeg_cnn/conv1d_1/bias/mAdam/eeg_cnn/conv1d_2/kernel/mAdam/eeg_cnn/conv1d_2/bias/mAdam/eeg_cnn/dense_1/kernel/mAdam/eeg_cnn/dense_1/bias/mAdam/eeg_cnn/conv1d/kernel/vAdam/eeg_cnn/conv1d/bias/vAdam/eeg_cnn/conv1d_1/kernel/vAdam/eeg_cnn/conv1d_1/bias/vAdam/eeg_cnn/conv1d_2/kernel/vAdam/eeg_cnn/conv1d_2/bias/vAdam/eeg_cnn/dense_1/kernel/vAdam/eeg_cnn/dense_1/bias/v*-
Tin&
$2"*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? **
f%R#
!__inference__traced_restore_29907??
?
f
J__inference_max_pooling1d_2_layer_call_and_return_conditional_losses_29645

inputs
identityP
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :t

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*0
_output_shapes
:???????????
MaxPoolMaxPoolExpandDims:output:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
r
SqueezeSqueezeMaxPool:output:0*
T0*,
_output_shapes
:??????????*
squeeze_dims
]
IdentityIdentitySqueeze:output:0*
T0*,
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
'__inference_eeg_cnn_layer_call_fn_29431

inputs
unknown:2
	unknown_0:2
	unknown_1:2d
	unknown_2:d 
	unknown_3:d?
	unknown_4:	?
	unknown_5:	?	
	unknown_6:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_eeg_cnn_layer_call_and_return_conditional_losses_29252o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:??????????: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
(__inference_conv1d_2_layer_call_fn_29603

inputs
unknown:d?
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_conv1d_2_layer_call_and_return_conditional_losses_29211t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????d: : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????d
 
_user_specified_nameinputs
?
K
/__inference_max_pooling1d_1_layer_call_fn_29578

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????d* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_max_pooling1d_1_layer_call_and_return_conditional_losses_29193d
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:?????????d"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@d:S O
+
_output_shapes
:?????????@d
 
_user_specified_nameinputs
?
?
C__inference_conv1d_2_layer_call_and_return_conditional_losses_29619

inputsB
+conv1d_expanddims_1_readvariableop_resource:d?.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?"Conv1D/ExpandDims_1/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
Conv1D/ExpandDims
ExpandDimsinputsConv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????d?
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:d?*
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:d??
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
?
Conv1D/SqueezeSqueezeConv1D:output:0*
T0*,
_output_shapes
:??????????*
squeeze_dims

?????????s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????U
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:??????????f
IdentityIdentityRelu:activations:0^NoOp*
T0*,
_output_shapes
:???????????
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????d: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:?????????d
 
_user_specified_nameinputs
?
f
J__inference_max_pooling1d_2_layer_call_and_return_conditional_losses_29123

inputs
identityP
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*A
_output_shapes/
-:+????????????????????????????
MaxPoolMaxPoolExpandDims:output:0*A
_output_shapes/
-:+???????????????????????????*
ksize
*
paddingVALID*
strides
?
SqueezeSqueezeMaxPool:output:0*
T0*=
_output_shapes+
):'???????????????????????????*
squeeze_dims
n
IdentityIdentitySqueeze:output:0*
T0*=
_output_shapes+
):'???????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
I
-__inference_max_pooling1d_layer_call_fn_29522

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'???????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_max_pooling1d_layer_call_and_return_conditional_losses_29093v
IdentityIdentityPartitionedCall:output:0*
T0*=
_output_shapes+
):'???????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
^
B__inference_flatten_layer_call_and_return_conditional_losses_29232

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????	Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
d
H__inference_max_pooling1d_layer_call_and_return_conditional_losses_29162

inputs
identityP
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :t

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*0
_output_shapes
:??????????2?
MaxPoolMaxPoolExpandDims:output:0*/
_output_shapes
:?????????B2*
ksize
*
paddingVALID*
strides
q
SqueezeSqueezeMaxPool:output:0*
T0*+
_output_shapes
:?????????B2*
squeeze_dims
\
IdentityIdentitySqueeze:output:0*
T0*+
_output_shapes
:?????????B2"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????2:T P
,
_output_shapes
:??????????2
 
_user_specified_nameinputs
?
?
C__inference_conv1d_1_layer_call_and_return_conditional_losses_29180

inputsA
+conv1d_expanddims_1_readvariableop_resource:2d-
biasadd_readvariableop_resource:d
identity??BiasAdd/ReadVariableOp?"Conv1D/ExpandDims_1/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
Conv1D/ExpandDims
ExpandDimsinputsConv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????B2?
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:2d*
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2d?
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????@d*
paddingVALID*
strides
?
Conv1D/SqueezeSqueezeConv1D:output:0*
T0*+
_output_shapes
:?????????@d*
squeeze_dims

?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype0?
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????@dT
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:?????????@de
IdentityIdentityRelu:activations:0^NoOp*
T0*+
_output_shapes
:?????????@d?
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????B2: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:?????????B2
 
_user_specified_nameinputs
?Q
?
 __inference__wrapped_model_29081
input_1P
:eeg_cnn_conv1d_conv1d_expanddims_1_readvariableop_resource:2<
.eeg_cnn_conv1d_biasadd_readvariableop_resource:2R
<eeg_cnn_conv1d_1_conv1d_expanddims_1_readvariableop_resource:2d>
0eeg_cnn_conv1d_1_biasadd_readvariableop_resource:dS
<eeg_cnn_conv1d_2_conv1d_expanddims_1_readvariableop_resource:d??
0eeg_cnn_conv1d_2_biasadd_readvariableop_resource:	?A
.eeg_cnn_dense_1_matmul_readvariableop_resource:	?	=
/eeg_cnn_dense_1_biasadd_readvariableop_resource:
identity??%eeg_cnn/conv1d/BiasAdd/ReadVariableOp?1eeg_cnn/conv1d/Conv1D/ExpandDims_1/ReadVariableOp?'eeg_cnn/conv1d_1/BiasAdd/ReadVariableOp?3eeg_cnn/conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp?'eeg_cnn/conv1d_2/BiasAdd/ReadVariableOp?3eeg_cnn/conv1d_2/Conv1D/ExpandDims_1/ReadVariableOp?&eeg_cnn/dense_1/BiasAdd/ReadVariableOp?%eeg_cnn/dense_1/MatMul/ReadVariableOpo
$eeg_cnn/conv1d/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
 eeg_cnn/conv1d/Conv1D/ExpandDims
ExpandDimsinput_1-eeg_cnn/conv1d/Conv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:???????????
1eeg_cnn/conv1d/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp:eeg_cnn_conv1d_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:2*
dtype0h
&eeg_cnn/conv1d/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
"eeg_cnn/conv1d/Conv1D/ExpandDims_1
ExpandDims9eeg_cnn/conv1d/Conv1D/ExpandDims_1/ReadVariableOp:value:0/eeg_cnn/conv1d/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2?
eeg_cnn/conv1d/Conv1DConv2D)eeg_cnn/conv1d/Conv1D/ExpandDims:output:0+eeg_cnn/conv1d/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:??????????2*
paddingVALID*
strides
?
eeg_cnn/conv1d/Conv1D/SqueezeSqueezeeeg_cnn/conv1d/Conv1D:output:0*
T0*,
_output_shapes
:??????????2*
squeeze_dims

??????????
%eeg_cnn/conv1d/BiasAdd/ReadVariableOpReadVariableOp.eeg_cnn_conv1d_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0?
eeg_cnn/conv1d/BiasAddBiasAdd&eeg_cnn/conv1d/Conv1D/Squeeze:output:0-eeg_cnn/conv1d/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????2s
eeg_cnn/conv1d/ReluRelueeg_cnn/conv1d/BiasAdd:output:0*
T0*,
_output_shapes
:??????????2f
$eeg_cnn/max_pooling1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
 eeg_cnn/max_pooling1d/ExpandDims
ExpandDims!eeg_cnn/conv1d/Relu:activations:0-eeg_cnn/max_pooling1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:??????????2?
eeg_cnn/max_pooling1d/MaxPoolMaxPool)eeg_cnn/max_pooling1d/ExpandDims:output:0*/
_output_shapes
:?????????B2*
ksize
*
paddingVALID*
strides
?
eeg_cnn/max_pooling1d/SqueezeSqueeze&eeg_cnn/max_pooling1d/MaxPool:output:0*
T0*+
_output_shapes
:?????????B2*
squeeze_dims
q
&eeg_cnn/conv1d_1/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
"eeg_cnn/conv1d_1/Conv1D/ExpandDims
ExpandDims&eeg_cnn/max_pooling1d/Squeeze:output:0/eeg_cnn/conv1d_1/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????B2?
3eeg_cnn/conv1d_1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp<eeg_cnn_conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:2d*
dtype0j
(eeg_cnn/conv1d_1/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
$eeg_cnn/conv1d_1/Conv1D/ExpandDims_1
ExpandDims;eeg_cnn/conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp:value:01eeg_cnn/conv1d_1/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2d?
eeg_cnn/conv1d_1/Conv1DConv2D+eeg_cnn/conv1d_1/Conv1D/ExpandDims:output:0-eeg_cnn/conv1d_1/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????@d*
paddingVALID*
strides
?
eeg_cnn/conv1d_1/Conv1D/SqueezeSqueeze eeg_cnn/conv1d_1/Conv1D:output:0*
T0*+
_output_shapes
:?????????@d*
squeeze_dims

??????????
'eeg_cnn/conv1d_1/BiasAdd/ReadVariableOpReadVariableOp0eeg_cnn_conv1d_1_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0?
eeg_cnn/conv1d_1/BiasAddBiasAdd(eeg_cnn/conv1d_1/Conv1D/Squeeze:output:0/eeg_cnn/conv1d_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????@dv
eeg_cnn/conv1d_1/ReluRelu!eeg_cnn/conv1d_1/BiasAdd:output:0*
T0*+
_output_shapes
:?????????@dh
&eeg_cnn/max_pooling1d_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
"eeg_cnn/max_pooling1d_1/ExpandDims
ExpandDims#eeg_cnn/conv1d_1/Relu:activations:0/eeg_cnn/max_pooling1d_1/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????@d?
eeg_cnn/max_pooling1d_1/MaxPoolMaxPool+eeg_cnn/max_pooling1d_1/ExpandDims:output:0*/
_output_shapes
:?????????d*
ksize
*
paddingVALID*
strides
?
eeg_cnn/max_pooling1d_1/SqueezeSqueeze(eeg_cnn/max_pooling1d_1/MaxPool:output:0*
T0*+
_output_shapes
:?????????d*
squeeze_dims
q
&eeg_cnn/conv1d_2/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
"eeg_cnn/conv1d_2/Conv1D/ExpandDims
ExpandDims(eeg_cnn/max_pooling1d_1/Squeeze:output:0/eeg_cnn/conv1d_2/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????d?
3eeg_cnn/conv1d_2/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp<eeg_cnn_conv1d_2_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:d?*
dtype0j
(eeg_cnn/conv1d_2/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
$eeg_cnn/conv1d_2/Conv1D/ExpandDims_1
ExpandDims;eeg_cnn/conv1d_2/Conv1D/ExpandDims_1/ReadVariableOp:value:01eeg_cnn/conv1d_2/Conv1D/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:d??
eeg_cnn/conv1d_2/Conv1DConv2D+eeg_cnn/conv1d_2/Conv1D/ExpandDims:output:0-eeg_cnn/conv1d_2/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
?
eeg_cnn/conv1d_2/Conv1D/SqueezeSqueeze eeg_cnn/conv1d_2/Conv1D:output:0*
T0*,
_output_shapes
:??????????*
squeeze_dims

??????????
'eeg_cnn/conv1d_2/BiasAdd/ReadVariableOpReadVariableOp0eeg_cnn_conv1d_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
eeg_cnn/conv1d_2/BiasAddBiasAdd(eeg_cnn/conv1d_2/Conv1D/Squeeze:output:0/eeg_cnn/conv1d_2/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????w
eeg_cnn/conv1d_2/ReluRelu!eeg_cnn/conv1d_2/BiasAdd:output:0*
T0*,
_output_shapes
:??????????h
&eeg_cnn/max_pooling1d_2/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
"eeg_cnn/max_pooling1d_2/ExpandDims
ExpandDims#eeg_cnn/conv1d_2/Relu:activations:0/eeg_cnn/max_pooling1d_2/ExpandDims/dim:output:0*
T0*0
_output_shapes
:???????????
eeg_cnn/max_pooling1d_2/MaxPoolMaxPool+eeg_cnn/max_pooling1d_2/ExpandDims:output:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
?
eeg_cnn/max_pooling1d_2/SqueezeSqueeze(eeg_cnn/max_pooling1d_2/MaxPool:output:0*
T0*,
_output_shapes
:??????????*
squeeze_dims
f
eeg_cnn/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  ?
eeg_cnn/flatten/ReshapeReshape(eeg_cnn/max_pooling1d_2/Squeeze:output:0eeg_cnn/flatten/Const:output:0*
T0*(
_output_shapes
:??????????	?
%eeg_cnn/dense_1/MatMul/ReadVariableOpReadVariableOp.eeg_cnn_dense_1_matmul_readvariableop_resource*
_output_shapes
:	?	*
dtype0?
eeg_cnn/dense_1/MatMulMatMul eeg_cnn/flatten/Reshape:output:0-eeg_cnn/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
&eeg_cnn/dense_1/BiasAdd/ReadVariableOpReadVariableOp/eeg_cnn_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
eeg_cnn/dense_1/BiasAddBiasAdd eeg_cnn/dense_1/MatMul:product:0.eeg_cnn/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????v
eeg_cnn/dense_1/SoftmaxSoftmax eeg_cnn/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????p
IdentityIdentity!eeg_cnn/dense_1/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp&^eeg_cnn/conv1d/BiasAdd/ReadVariableOp2^eeg_cnn/conv1d/Conv1D/ExpandDims_1/ReadVariableOp(^eeg_cnn/conv1d_1/BiasAdd/ReadVariableOp4^eeg_cnn/conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp(^eeg_cnn/conv1d_2/BiasAdd/ReadVariableOp4^eeg_cnn/conv1d_2/Conv1D/ExpandDims_1/ReadVariableOp'^eeg_cnn/dense_1/BiasAdd/ReadVariableOp&^eeg_cnn/dense_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:??????????: : : : : : : : 2N
%eeg_cnn/conv1d/BiasAdd/ReadVariableOp%eeg_cnn/conv1d/BiasAdd/ReadVariableOp2f
1eeg_cnn/conv1d/Conv1D/ExpandDims_1/ReadVariableOp1eeg_cnn/conv1d/Conv1D/ExpandDims_1/ReadVariableOp2R
'eeg_cnn/conv1d_1/BiasAdd/ReadVariableOp'eeg_cnn/conv1d_1/BiasAdd/ReadVariableOp2j
3eeg_cnn/conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp3eeg_cnn/conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp2R
'eeg_cnn/conv1d_2/BiasAdd/ReadVariableOp'eeg_cnn/conv1d_2/BiasAdd/ReadVariableOp2j
3eeg_cnn/conv1d_2/Conv1D/ExpandDims_1/ReadVariableOp3eeg_cnn/conv1d_2/Conv1D/ExpandDims_1/ReadVariableOp2P
&eeg_cnn/dense_1/BiasAdd/ReadVariableOp&eeg_cnn/dense_1/BiasAdd/ReadVariableOp2N
%eeg_cnn/dense_1/MatMul/ReadVariableOp%eeg_cnn/dense_1/MatMul/ReadVariableOp:U Q
,
_output_shapes
:??????????
!
_user_specified_name	input_1
?H
?
B__inference_eeg_cnn_layer_call_and_return_conditional_losses_29492

inputsH
2conv1d_conv1d_expanddims_1_readvariableop_resource:24
&conv1d_biasadd_readvariableop_resource:2J
4conv1d_1_conv1d_expanddims_1_readvariableop_resource:2d6
(conv1d_1_biasadd_readvariableop_resource:dK
4conv1d_2_conv1d_expanddims_1_readvariableop_resource:d?7
(conv1d_2_biasadd_readvariableop_resource:	?9
&dense_1_matmul_readvariableop_resource:	?	5
'dense_1_biasadd_readvariableop_resource:
identity??conv1d/BiasAdd/ReadVariableOp?)conv1d/Conv1D/ExpandDims_1/ReadVariableOp?conv1d_1/BiasAdd/ReadVariableOp?+conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp?conv1d_2/BiasAdd/ReadVariableOp?+conv1d_2/Conv1D/ExpandDims_1/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOpg
conv1d/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
conv1d/Conv1D/ExpandDims
ExpandDimsinputs%conv1d/Conv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:???????????
)conv1d/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp2conv1d_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:2*
dtype0`
conv1d/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
conv1d/Conv1D/ExpandDims_1
ExpandDims1conv1d/Conv1D/ExpandDims_1/ReadVariableOp:value:0'conv1d/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2?
conv1d/Conv1DConv2D!conv1d/Conv1D/ExpandDims:output:0#conv1d/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:??????????2*
paddingVALID*
strides
?
conv1d/Conv1D/SqueezeSqueezeconv1d/Conv1D:output:0*
T0*,
_output_shapes
:??????????2*
squeeze_dims

??????????
conv1d/BiasAdd/ReadVariableOpReadVariableOp&conv1d_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0?
conv1d/BiasAddBiasAddconv1d/Conv1D/Squeeze:output:0%conv1d/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????2c
conv1d/ReluReluconv1d/BiasAdd:output:0*
T0*,
_output_shapes
:??????????2^
max_pooling1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
max_pooling1d/ExpandDims
ExpandDimsconv1d/Relu:activations:0%max_pooling1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:??????????2?
max_pooling1d/MaxPoolMaxPool!max_pooling1d/ExpandDims:output:0*/
_output_shapes
:?????????B2*
ksize
*
paddingVALID*
strides
?
max_pooling1d/SqueezeSqueezemax_pooling1d/MaxPool:output:0*
T0*+
_output_shapes
:?????????B2*
squeeze_dims
i
conv1d_1/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
conv1d_1/Conv1D/ExpandDims
ExpandDimsmax_pooling1d/Squeeze:output:0'conv1d_1/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????B2?
+conv1d_1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:2d*
dtype0b
 conv1d_1/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
conv1d_1/Conv1D/ExpandDims_1
ExpandDims3conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp:value:0)conv1d_1/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2d?
conv1d_1/Conv1DConv2D#conv1d_1/Conv1D/ExpandDims:output:0%conv1d_1/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????@d*
paddingVALID*
strides
?
conv1d_1/Conv1D/SqueezeSqueezeconv1d_1/Conv1D:output:0*
T0*+
_output_shapes
:?????????@d*
squeeze_dims

??????????
conv1d_1/BiasAdd/ReadVariableOpReadVariableOp(conv1d_1_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0?
conv1d_1/BiasAddBiasAdd conv1d_1/Conv1D/Squeeze:output:0'conv1d_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????@df
conv1d_1/ReluReluconv1d_1/BiasAdd:output:0*
T0*+
_output_shapes
:?????????@d`
max_pooling1d_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
max_pooling1d_1/ExpandDims
ExpandDimsconv1d_1/Relu:activations:0'max_pooling1d_1/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????@d?
max_pooling1d_1/MaxPoolMaxPool#max_pooling1d_1/ExpandDims:output:0*/
_output_shapes
:?????????d*
ksize
*
paddingVALID*
strides
?
max_pooling1d_1/SqueezeSqueeze max_pooling1d_1/MaxPool:output:0*
T0*+
_output_shapes
:?????????d*
squeeze_dims
i
conv1d_2/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
conv1d_2/Conv1D/ExpandDims
ExpandDims max_pooling1d_1/Squeeze:output:0'conv1d_2/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????d?
+conv1d_2/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_2_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:d?*
dtype0b
 conv1d_2/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
conv1d_2/Conv1D/ExpandDims_1
ExpandDims3conv1d_2/Conv1D/ExpandDims_1/ReadVariableOp:value:0)conv1d_2/Conv1D/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:d??
conv1d_2/Conv1DConv2D#conv1d_2/Conv1D/ExpandDims:output:0%conv1d_2/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
?
conv1d_2/Conv1D/SqueezeSqueezeconv1d_2/Conv1D:output:0*
T0*,
_output_shapes
:??????????*
squeeze_dims

??????????
conv1d_2/BiasAdd/ReadVariableOpReadVariableOp(conv1d_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
conv1d_2/BiasAddBiasAdd conv1d_2/Conv1D/Squeeze:output:0'conv1d_2/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????g
conv1d_2/ReluReluconv1d_2/BiasAdd:output:0*
T0*,
_output_shapes
:??????????`
max_pooling1d_2/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
max_pooling1d_2/ExpandDims
ExpandDimsconv1d_2/Relu:activations:0'max_pooling1d_2/ExpandDims/dim:output:0*
T0*0
_output_shapes
:???????????
max_pooling1d_2/MaxPoolMaxPool#max_pooling1d_2/ExpandDims:output:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
?
max_pooling1d_2/SqueezeSqueeze max_pooling1d_2/MaxPool:output:0*
T0*,
_output_shapes
:??????????*
squeeze_dims
^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  ?
flatten/ReshapeReshape max_pooling1d_2/Squeeze:output:0flatten/Const:output:0*
T0*(
_output_shapes
:??????????	?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	?	*
dtype0?
dense_1/MatMulMatMulflatten/Reshape:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????f
dense_1/SoftmaxSoftmaxdense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????h
IdentityIdentitydense_1/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^conv1d/BiasAdd/ReadVariableOp*^conv1d/Conv1D/ExpandDims_1/ReadVariableOp ^conv1d_1/BiasAdd/ReadVariableOp,^conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp ^conv1d_2/BiasAdd/ReadVariableOp,^conv1d_2/Conv1D/ExpandDims_1/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:??????????: : : : : : : : 2>
conv1d/BiasAdd/ReadVariableOpconv1d/BiasAdd/ReadVariableOp2V
)conv1d/Conv1D/ExpandDims_1/ReadVariableOp)conv1d/Conv1D/ExpandDims_1/ReadVariableOp2B
conv1d_1/BiasAdd/ReadVariableOpconv1d_1/BiasAdd/ReadVariableOp2Z
+conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp+conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp2B
conv1d_2/BiasAdd/ReadVariableOpconv1d_2/BiasAdd/ReadVariableOp2Z
+conv1d_2/Conv1D/ExpandDims_1/ReadVariableOp+conv1d_2/Conv1D/ExpandDims_1/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
f
J__inference_max_pooling1d_1_layer_call_and_return_conditional_losses_29108

inputs
identityP
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*A
_output_shapes/
-:+????????????????????????????
MaxPoolMaxPoolExpandDims:output:0*A
_output_shapes/
-:+???????????????????????????*
ksize
*
paddingVALID*
strides
?
SqueezeSqueezeMaxPool:output:0*
T0*=
_output_shapes+
):'???????????????????????????*
squeeze_dims
n
IdentityIdentitySqueeze:output:0*
T0*=
_output_shapes+
):'???????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
f
J__inference_max_pooling1d_1_layer_call_and_return_conditional_losses_29586

inputs
identityP
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*A
_output_shapes/
-:+????????????????????????????
MaxPoolMaxPoolExpandDims:output:0*A
_output_shapes/
-:+???????????????????????????*
ksize
*
paddingVALID*
strides
?
SqueezeSqueezeMaxPool:output:0*
T0*=
_output_shapes+
):'???????????????????????????*
squeeze_dims
n
IdentityIdentitySqueeze:output:0*
T0*=
_output_shapes+
):'???????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
d
H__inference_max_pooling1d_layer_call_and_return_conditional_losses_29535

inputs
identityP
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*A
_output_shapes/
-:+????????????????????????????
MaxPoolMaxPoolExpandDims:output:0*A
_output_shapes/
-:+???????????????????????????*
ksize
*
paddingVALID*
strides
?
SqueezeSqueezeMaxPool:output:0*
T0*=
_output_shapes+
):'???????????????????????????*
squeeze_dims
n
IdentityIdentitySqueeze:output:0*
T0*=
_output_shapes+
):'???????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?	
?
'__inference_eeg_cnn_layer_call_fn_29271
input_1
unknown:2
	unknown_0:2
	unknown_1:2d
	unknown_2:d 
	unknown_3:d?
	unknown_4:	?
	unknown_5:	?	
	unknown_6:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_eeg_cnn_layer_call_and_return_conditional_losses_29252o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:??????????: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
,
_output_shapes
:??????????
!
_user_specified_name	input_1
?	
?
#__inference_signature_wrapper_29410
input_1
unknown:2
	unknown_0:2
	unknown_1:2d
	unknown_2:d 
	unknown_3:d?
	unknown_4:	?
	unknown_5:	?	
	unknown_6:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *)
f$R"
 __inference__wrapped_model_29081o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:??????????: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
,
_output_shapes
:??????????
!
_user_specified_name	input_1
?
?
C__inference_conv1d_2_layer_call_and_return_conditional_losses_29211

inputsB
+conv1d_expanddims_1_readvariableop_resource:d?.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?"Conv1D/ExpandDims_1/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
Conv1D/ExpandDims
ExpandDimsinputsConv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????d?
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:d?*
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:d??
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
?
Conv1D/SqueezeSqueezeConv1D:output:0*
T0*,
_output_shapes
:??????????*
squeeze_dims

?????????s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????U
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:??????????f
IdentityIdentityRelu:activations:0^NoOp*
T0*,
_output_shapes
:???????????
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????d: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:?????????d
 
_user_specified_nameinputs
?!
?
B__inference_eeg_cnn_layer_call_and_return_conditional_losses_29381
input_1"
conv1d_29356:2
conv1d_29358:2$
conv1d_1_29362:2d
conv1d_1_29364:d%
conv1d_2_29368:d?
conv1d_2_29370:	? 
dense_1_29375:	?	
dense_1_29377:
identity??conv1d/StatefulPartitionedCall? conv1d_1/StatefulPartitionedCall? conv1d_2/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?
conv1d/StatefulPartitionedCallStatefulPartitionedCallinput_1conv1d_29356conv1d_29358*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_conv1d_layer_call_and_return_conditional_losses_29149?
max_pooling1d/PartitionedCallPartitionedCall'conv1d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????B2* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_max_pooling1d_layer_call_and_return_conditional_losses_29162?
 conv1d_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling1d/PartitionedCall:output:0conv1d_1_29362conv1d_1_29364*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????@d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_conv1d_1_layer_call_and_return_conditional_losses_29180?
max_pooling1d_1/PartitionedCallPartitionedCall)conv1d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????d* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_max_pooling1d_1_layer_call_and_return_conditional_losses_29193?
 conv1d_2/StatefulPartitionedCallStatefulPartitionedCall(max_pooling1d_1/PartitionedCall:output:0conv1d_2_29368conv1d_2_29370*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_conv1d_2_layer_call_and_return_conditional_losses_29211?
max_pooling1d_2/PartitionedCallPartitionedCall)conv1d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_max_pooling1d_2_layer_call_and_return_conditional_losses_29224?
flatten/PartitionedCallPartitionedCall(max_pooling1d_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????	* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_29232?
dense_1/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_1_29375dense_1_29377*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_29245w
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^conv1d/StatefulPartitionedCall!^conv1d_1/StatefulPartitionedCall!^conv1d_2/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:??????????: : : : : : : : 2@
conv1d/StatefulPartitionedCallconv1d/StatefulPartitionedCall2D
 conv1d_1/StatefulPartitionedCall conv1d_1/StatefulPartitionedCall2D
 conv1d_2/StatefulPartitionedCall conv1d_2/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:U Q
,
_output_shapes
:??????????
!
_user_specified_name	input_1
?
K
/__inference_max_pooling1d_2_layer_call_fn_29629

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_max_pooling1d_2_layer_call_and_return_conditional_losses_29224e
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?F
?
__inference__traced_save_29798
file_prefix4
0savev2_eeg_cnn_conv1d_kernel_read_readvariableop2
.savev2_eeg_cnn_conv1d_bias_read_readvariableop6
2savev2_eeg_cnn_conv1d_1_kernel_read_readvariableop4
0savev2_eeg_cnn_conv1d_1_bias_read_readvariableop6
2savev2_eeg_cnn_conv1d_2_kernel_read_readvariableop4
0savev2_eeg_cnn_conv1d_2_bias_read_readvariableop5
1savev2_eeg_cnn_dense_1_kernel_read_readvariableop3
/savev2_eeg_cnn_dense_1_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop;
7savev2_adam_eeg_cnn_conv1d_kernel_m_read_readvariableop9
5savev2_adam_eeg_cnn_conv1d_bias_m_read_readvariableop=
9savev2_adam_eeg_cnn_conv1d_1_kernel_m_read_readvariableop;
7savev2_adam_eeg_cnn_conv1d_1_bias_m_read_readvariableop=
9savev2_adam_eeg_cnn_conv1d_2_kernel_m_read_readvariableop;
7savev2_adam_eeg_cnn_conv1d_2_bias_m_read_readvariableop<
8savev2_adam_eeg_cnn_dense_1_kernel_m_read_readvariableop:
6savev2_adam_eeg_cnn_dense_1_bias_m_read_readvariableop;
7savev2_adam_eeg_cnn_conv1d_kernel_v_read_readvariableop9
5savev2_adam_eeg_cnn_conv1d_bias_v_read_readvariableop=
9savev2_adam_eeg_cnn_conv1d_1_kernel_v_read_readvariableop;
7savev2_adam_eeg_cnn_conv1d_1_bias_v_read_readvariableop=
9savev2_adam_eeg_cnn_conv1d_2_kernel_v_read_readvariableop;
7savev2_adam_eeg_cnn_conv1d_2_bias_v_read_readvariableop<
8savev2_adam_eeg_cnn_dense_1_kernel_v_read_readvariableop:
6savev2_adam_eeg_cnn_dense_1_bias_v_read_readvariableop
savev2_const

identity_1??MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : ?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*?
value?B?"B'conv1/kernel/.ATTRIBUTES/VARIABLE_VALUEB%conv1/bias/.ATTRIBUTES/VARIABLE_VALUEB'conv2/kernel/.ATTRIBUTES/VARIABLE_VALUEB%conv2/bias/.ATTRIBUTES/VARIABLE_VALUEB'conv3/kernel/.ATTRIBUTES/VARIABLE_VALUEB%conv3/bias/.ATTRIBUTES/VARIABLE_VALUEB$fc/kernel/.ATTRIBUTES/VARIABLE_VALUEB"fc/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBCconv1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBAconv1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCconv2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBAconv2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCconv3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBAconv3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB@fc/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB>fc/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCconv1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBAconv1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCconv2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBAconv2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCconv3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBAconv3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB@fc/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB>fc/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*W
valueNBL"B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:00savev2_eeg_cnn_conv1d_kernel_read_readvariableop.savev2_eeg_cnn_conv1d_bias_read_readvariableop2savev2_eeg_cnn_conv1d_1_kernel_read_readvariableop0savev2_eeg_cnn_conv1d_1_bias_read_readvariableop2savev2_eeg_cnn_conv1d_2_kernel_read_readvariableop0savev2_eeg_cnn_conv1d_2_bias_read_readvariableop1savev2_eeg_cnn_dense_1_kernel_read_readvariableop/savev2_eeg_cnn_dense_1_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop7savev2_adam_eeg_cnn_conv1d_kernel_m_read_readvariableop5savev2_adam_eeg_cnn_conv1d_bias_m_read_readvariableop9savev2_adam_eeg_cnn_conv1d_1_kernel_m_read_readvariableop7savev2_adam_eeg_cnn_conv1d_1_bias_m_read_readvariableop9savev2_adam_eeg_cnn_conv1d_2_kernel_m_read_readvariableop7savev2_adam_eeg_cnn_conv1d_2_bias_m_read_readvariableop8savev2_adam_eeg_cnn_dense_1_kernel_m_read_readvariableop6savev2_adam_eeg_cnn_dense_1_bias_m_read_readvariableop7savev2_adam_eeg_cnn_conv1d_kernel_v_read_readvariableop5savev2_adam_eeg_cnn_conv1d_bias_v_read_readvariableop9savev2_adam_eeg_cnn_conv1d_1_kernel_v_read_readvariableop7savev2_adam_eeg_cnn_conv1d_1_bias_v_read_readvariableop9savev2_adam_eeg_cnn_conv1d_2_kernel_v_read_readvariableop7savev2_adam_eeg_cnn_conv1d_2_bias_v_read_readvariableop8savev2_adam_eeg_cnn_dense_1_kernel_v_read_readvariableop6savev2_adam_eeg_cnn_dense_1_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *0
dtypes&
$2"	?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*?
_input_shapes?
?: :2:2:2d:d:d?:?:	?	:: : : : : : : : : :2:2:2d:d:d?:?:	?	::2:2:2d:d:d?:?:	?	:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:($
"
_output_shapes
:2: 

_output_shapes
:2:($
"
_output_shapes
:2d: 

_output_shapes
:d:)%
#
_output_shapes
:d?:!

_output_shapes	
:?:%!

_output_shapes
:	?	: 

_output_shapes
::	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :($
"
_output_shapes
:2: 

_output_shapes
:2:($
"
_output_shapes
:2d: 

_output_shapes
:d:)%
#
_output_shapes
:d?:!

_output_shapes	
:?:%!

_output_shapes
:	?	: 

_output_shapes
::($
"
_output_shapes
:2: 

_output_shapes
:2:($
"
_output_shapes
:2d: 

_output_shapes
:d:)%
#
_output_shapes
:d?:!

_output_shapes	
:?:% !

_output_shapes
:	?	: !

_output_shapes
::"

_output_shapes
: 
?
d
H__inference_max_pooling1d_layer_call_and_return_conditional_losses_29543

inputs
identityP
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :t

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*0
_output_shapes
:??????????2?
MaxPoolMaxPoolExpandDims:output:0*/
_output_shapes
:?????????B2*
ksize
*
paddingVALID*
strides
q
SqueezeSqueezeMaxPool:output:0*
T0*+
_output_shapes
:?????????B2*
squeeze_dims
\
IdentityIdentitySqueeze:output:0*
T0*+
_output_shapes
:?????????B2"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????2:T P
,
_output_shapes
:??????????2
 
_user_specified_nameinputs
?
?
&__inference_conv1d_layer_call_fn_29501

inputs
unknown:2
	unknown_0:2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_conv1d_layer_call_and_return_conditional_losses_29149t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:??????????2`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
A__inference_conv1d_layer_call_and_return_conditional_losses_29517

inputsA
+conv1d_expanddims_1_readvariableop_resource:2-
biasadd_readvariableop_resource:2
identity??BiasAdd/ReadVariableOp?"Conv1D/ExpandDims_1/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
Conv1D/ExpandDims
ExpandDimsinputsConv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:???????????
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:2*
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2?
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:??????????2*
paddingVALID*
strides
?
Conv1D/SqueezeSqueezeConv1D:output:0*
T0*,
_output_shapes
:??????????2*
squeeze_dims

?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:2*
dtype0?
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????2U
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:??????????2f
IdentityIdentityRelu:activations:0^NoOp*
T0*,
_output_shapes
:??????????2?
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
C
'__inference_flatten_layer_call_fn_29650

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????	* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_29232a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
K
/__inference_max_pooling1d_2_layer_call_fn_29624

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'???????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_max_pooling1d_2_layer_call_and_return_conditional_losses_29123v
IdentityIdentityPartitionedCall:output:0*
T0*=
_output_shapes+
):'???????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
?
C__inference_conv1d_1_layer_call_and_return_conditional_losses_29568

inputsA
+conv1d_expanddims_1_readvariableop_resource:2d-
biasadd_readvariableop_resource:d
identity??BiasAdd/ReadVariableOp?"Conv1D/ExpandDims_1/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
Conv1D/ExpandDims
ExpandDimsinputsConv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????B2?
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:2d*
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2d?
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????@d*
paddingVALID*
strides
?
Conv1D/SqueezeSqueezeConv1D:output:0*
T0*+
_output_shapes
:?????????@d*
squeeze_dims

?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype0?
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????@dT
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:?????????@de
IdentityIdentityRelu:activations:0^NoOp*
T0*+
_output_shapes
:?????????@d?
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????B2: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:?????????B2
 
_user_specified_nameinputs
?
^
B__inference_flatten_layer_call_and_return_conditional_losses_29656

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????	Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
'__inference_dense_1_layer_call_fn_29665

inputs
unknown:	?	
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_29245o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????	: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????	
 
_user_specified_nameinputs
?
?
A__inference_conv1d_layer_call_and_return_conditional_losses_29149

inputsA
+conv1d_expanddims_1_readvariableop_resource:2-
biasadd_readvariableop_resource:2
identity??BiasAdd/ReadVariableOp?"Conv1D/ExpandDims_1/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
Conv1D/ExpandDims
ExpandDimsinputsConv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:???????????
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:2*
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2?
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:??????????2*
paddingVALID*
strides
?
Conv1D/SqueezeSqueezeConv1D:output:0*
T0*,
_output_shapes
:??????????2*
squeeze_dims

?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:2*
dtype0?
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????2U
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:??????????2f
IdentityIdentityRelu:activations:0^NoOp*
T0*,
_output_shapes
:??????????2?
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
B__inference_dense_1_layer_call_and_return_conditional_losses_29676

inputs1
matmul_readvariableop_resource:	?	-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?	*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????	
 
_user_specified_nameinputs
?
f
J__inference_max_pooling1d_1_layer_call_and_return_conditional_losses_29594

inputs
identityP
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :s

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????@d?
MaxPoolMaxPoolExpandDims:output:0*/
_output_shapes
:?????????d*
ksize
*
paddingVALID*
strides
q
SqueezeSqueezeMaxPool:output:0*
T0*+
_output_shapes
:?????????d*
squeeze_dims
\
IdentityIdentitySqueeze:output:0*
T0*+
_output_shapes
:?????????d"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@d:S O
+
_output_shapes
:?????????@d
 
_user_specified_nameinputs
?
I
-__inference_max_pooling1d_layer_call_fn_29527

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????B2* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_max_pooling1d_layer_call_and_return_conditional_losses_29162d
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:?????????B2"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????2:T P
,
_output_shapes
:??????????2
 
_user_specified_nameinputs
?
f
J__inference_max_pooling1d_2_layer_call_and_return_conditional_losses_29224

inputs
identityP
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :t

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*0
_output_shapes
:???????????
MaxPoolMaxPoolExpandDims:output:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
r
SqueezeSqueezeMaxPool:output:0*
T0*,
_output_shapes
:??????????*
squeeze_dims
]
IdentityIdentitySqueeze:output:0*
T0*,
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?!
?
B__inference_eeg_cnn_layer_call_and_return_conditional_losses_29252

inputs"
conv1d_29150:2
conv1d_29152:2$
conv1d_1_29181:2d
conv1d_1_29183:d%
conv1d_2_29212:d?
conv1d_2_29214:	? 
dense_1_29246:	?	
dense_1_29248:
identity??conv1d/StatefulPartitionedCall? conv1d_1/StatefulPartitionedCall? conv1d_2/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?
conv1d/StatefulPartitionedCallStatefulPartitionedCallinputsconv1d_29150conv1d_29152*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_conv1d_layer_call_and_return_conditional_losses_29149?
max_pooling1d/PartitionedCallPartitionedCall'conv1d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????B2* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_max_pooling1d_layer_call_and_return_conditional_losses_29162?
 conv1d_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling1d/PartitionedCall:output:0conv1d_1_29181conv1d_1_29183*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????@d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_conv1d_1_layer_call_and_return_conditional_losses_29180?
max_pooling1d_1/PartitionedCallPartitionedCall)conv1d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????d* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_max_pooling1d_1_layer_call_and_return_conditional_losses_29193?
 conv1d_2/StatefulPartitionedCallStatefulPartitionedCall(max_pooling1d_1/PartitionedCall:output:0conv1d_2_29212conv1d_2_29214*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_conv1d_2_layer_call_and_return_conditional_losses_29211?
max_pooling1d_2/PartitionedCallPartitionedCall)conv1d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_max_pooling1d_2_layer_call_and_return_conditional_losses_29224?
flatten/PartitionedCallPartitionedCall(max_pooling1d_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????	* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_29232?
dense_1/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_1_29246dense_1_29248*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_29245w
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^conv1d/StatefulPartitionedCall!^conv1d_1/StatefulPartitionedCall!^conv1d_2/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:??????????: : : : : : : : 2@
conv1d/StatefulPartitionedCallconv1d/StatefulPartitionedCall2D
 conv1d_1/StatefulPartitionedCall conv1d_1/StatefulPartitionedCall2D
 conv1d_2/StatefulPartitionedCall conv1d_2/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
f
J__inference_max_pooling1d_2_layer_call_and_return_conditional_losses_29637

inputs
identityP
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*A
_output_shapes/
-:+????????????????????????????
MaxPoolMaxPoolExpandDims:output:0*A
_output_shapes/
-:+???????????????????????????*
ksize
*
paddingVALID*
strides
?
SqueezeSqueezeMaxPool:output:0*
T0*=
_output_shapes+
):'???????????????????????????*
squeeze_dims
n
IdentityIdentitySqueeze:output:0*
T0*=
_output_shapes+
):'???????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
?
(__inference_conv1d_1_layer_call_fn_29552

inputs
unknown:2d
	unknown_0:d
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????@d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_conv1d_1_layer_call_and_return_conditional_losses_29180s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????@d`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????B2: : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????B2
 
_user_specified_nameinputs
?
f
J__inference_max_pooling1d_1_layer_call_and_return_conditional_losses_29193

inputs
identityP
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :s

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????@d?
MaxPoolMaxPoolExpandDims:output:0*/
_output_shapes
:?????????d*
ksize
*
paddingVALID*
strides
q
SqueezeSqueezeMaxPool:output:0*
T0*+
_output_shapes
:?????????d*
squeeze_dims
\
IdentityIdentitySqueeze:output:0*
T0*+
_output_shapes
:?????????d"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@d:S O
+
_output_shapes
:?????????@d
 
_user_specified_nameinputs
?

?
B__inference_dense_1_layer_call_and_return_conditional_losses_29245

inputs1
matmul_readvariableop_resource:	?	-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?	*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????	
 
_user_specified_nameinputs
?
d
H__inference_max_pooling1d_layer_call_and_return_conditional_losses_29093

inputs
identityP
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*A
_output_shapes/
-:+????????????????????????????
MaxPoolMaxPoolExpandDims:output:0*A
_output_shapes/
-:+???????????????????????????*
ksize
*
paddingVALID*
strides
?
SqueezeSqueezeMaxPool:output:0*
T0*=
_output_shapes+
):'???????????????????????????*
squeeze_dims
n
IdentityIdentitySqueeze:output:0*
T0*=
_output_shapes+
):'???????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
K
/__inference_max_pooling1d_1_layer_call_fn_29573

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'???????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_max_pooling1d_1_layer_call_and_return_conditional_losses_29108v
IdentityIdentityPartitionedCall:output:0*
T0*=
_output_shapes+
):'???????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
??
?
!__inference__traced_restore_29907
file_prefix<
&assignvariableop_eeg_cnn_conv1d_kernel:24
&assignvariableop_1_eeg_cnn_conv1d_bias:2@
*assignvariableop_2_eeg_cnn_conv1d_1_kernel:2d6
(assignvariableop_3_eeg_cnn_conv1d_1_bias:dA
*assignvariableop_4_eeg_cnn_conv1d_2_kernel:d?7
(assignvariableop_5_eeg_cnn_conv1d_2_bias:	?<
)assignvariableop_6_eeg_cnn_dense_1_kernel:	?	5
'assignvariableop_7_eeg_cnn_dense_1_bias:&
assignvariableop_8_adam_iter:	 (
assignvariableop_9_adam_beta_1: )
assignvariableop_10_adam_beta_2: (
assignvariableop_11_adam_decay: 0
&assignvariableop_12_adam_learning_rate: #
assignvariableop_13_total: #
assignvariableop_14_count: %
assignvariableop_15_total_1: %
assignvariableop_16_count_1: F
0assignvariableop_17_adam_eeg_cnn_conv1d_kernel_m:2<
.assignvariableop_18_adam_eeg_cnn_conv1d_bias_m:2H
2assignvariableop_19_adam_eeg_cnn_conv1d_1_kernel_m:2d>
0assignvariableop_20_adam_eeg_cnn_conv1d_1_bias_m:dI
2assignvariableop_21_adam_eeg_cnn_conv1d_2_kernel_m:d??
0assignvariableop_22_adam_eeg_cnn_conv1d_2_bias_m:	?D
1assignvariableop_23_adam_eeg_cnn_dense_1_kernel_m:	?	=
/assignvariableop_24_adam_eeg_cnn_dense_1_bias_m:F
0assignvariableop_25_adam_eeg_cnn_conv1d_kernel_v:2<
.assignvariableop_26_adam_eeg_cnn_conv1d_bias_v:2H
2assignvariableop_27_adam_eeg_cnn_conv1d_1_kernel_v:2d>
0assignvariableop_28_adam_eeg_cnn_conv1d_1_bias_v:dI
2assignvariableop_29_adam_eeg_cnn_conv1d_2_kernel_v:d??
0assignvariableop_30_adam_eeg_cnn_conv1d_2_bias_v:	?D
1assignvariableop_31_adam_eeg_cnn_dense_1_kernel_v:	?	=
/assignvariableop_32_adam_eeg_cnn_dense_1_bias_v:
identity_34??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*?
value?B?"B'conv1/kernel/.ATTRIBUTES/VARIABLE_VALUEB%conv1/bias/.ATTRIBUTES/VARIABLE_VALUEB'conv2/kernel/.ATTRIBUTES/VARIABLE_VALUEB%conv2/bias/.ATTRIBUTES/VARIABLE_VALUEB'conv3/kernel/.ATTRIBUTES/VARIABLE_VALUEB%conv3/bias/.ATTRIBUTES/VARIABLE_VALUEB$fc/kernel/.ATTRIBUTES/VARIABLE_VALUEB"fc/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBCconv1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBAconv1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCconv2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBAconv2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCconv3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBAconv3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB@fc/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB>fc/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCconv1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBAconv1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCconv2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBAconv2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCconv3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBAconv3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB@fc/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB>fc/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*W
valueNBL"B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::*0
dtypes&
$2"	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOpAssignVariableOp&assignvariableop_eeg_cnn_conv1d_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_1AssignVariableOp&assignvariableop_1_eeg_cnn_conv1d_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_2AssignVariableOp*assignvariableop_2_eeg_cnn_conv1d_1_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_3AssignVariableOp(assignvariableop_3_eeg_cnn_conv1d_1_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_4AssignVariableOp*assignvariableop_4_eeg_cnn_conv1d_2_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_5AssignVariableOp(assignvariableop_5_eeg_cnn_conv1d_2_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_6AssignVariableOp)assignvariableop_6_eeg_cnn_dense_1_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_7AssignVariableOp'assignvariableop_7_eeg_cnn_dense_1_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0	*
_output_shapes
:?
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_iterIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_9AssignVariableOpassignvariableop_9_adam_beta_1Identity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_10AssignVariableOpassignvariableop_10_adam_beta_2Identity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_11AssignVariableOpassignvariableop_11_adam_decayIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_12AssignVariableOp&assignvariableop_12_adam_learning_rateIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_13AssignVariableOpassignvariableop_13_totalIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_14AssignVariableOpassignvariableop_14_countIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_15AssignVariableOpassignvariableop_15_total_1Identity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_16AssignVariableOpassignvariableop_16_count_1Identity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_17AssignVariableOp0assignvariableop_17_adam_eeg_cnn_conv1d_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_18AssignVariableOp.assignvariableop_18_adam_eeg_cnn_conv1d_bias_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_19AssignVariableOp2assignvariableop_19_adam_eeg_cnn_conv1d_1_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_20AssignVariableOp0assignvariableop_20_adam_eeg_cnn_conv1d_1_bias_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_21AssignVariableOp2assignvariableop_21_adam_eeg_cnn_conv1d_2_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_22AssignVariableOp0assignvariableop_22_adam_eeg_cnn_conv1d_2_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_23AssignVariableOp1assignvariableop_23_adam_eeg_cnn_dense_1_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_24AssignVariableOp/assignvariableop_24_adam_eeg_cnn_dense_1_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_25AssignVariableOp0assignvariableop_25_adam_eeg_cnn_conv1d_kernel_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_26AssignVariableOp.assignvariableop_26_adam_eeg_cnn_conv1d_bias_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_27AssignVariableOp2assignvariableop_27_adam_eeg_cnn_conv1d_1_kernel_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_28AssignVariableOp0assignvariableop_28_adam_eeg_cnn_conv1d_1_bias_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_29AssignVariableOp2assignvariableop_29_adam_eeg_cnn_conv1d_2_kernel_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_30AssignVariableOp0assignvariableop_30_adam_eeg_cnn_conv1d_2_bias_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_31AssignVariableOp1assignvariableop_31_adam_eeg_cnn_dense_1_kernel_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_32AssignVariableOp/assignvariableop_32_adam_eeg_cnn_dense_1_bias_vIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ?
Identity_33Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_34IdentityIdentity_33:output:0^NoOp_1*
T0*
_output_shapes
: ?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_34Identity_34:output:0*W
_input_shapesF
D: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
@
input_15
serving_default_input_1:0??????????<
output_10
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
?
	conv1
	pool1
	conv2
	pool2
	conv3
	pool3
flat
fc
		optimizer

	variables
trainable_variables
regularization_losses
	keras_api

signatures
?__call__
+?&call_and_return_all_conditional_losses
?_default_save_signature"
_tf_keras_model
?

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
	variables
trainable_variables
regularization_losses
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
	variables
 trainable_variables
!regularization_losses
"	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

#kernel
$bias
%	variables
&trainable_variables
'regularization_losses
(	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
)	variables
*trainable_variables
+regularization_losses
,	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
-	variables
.trainable_variables
/regularization_losses
0	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

1kernel
2bias
3	variables
4trainable_variables
5regularization_losses
6	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
7iter

8beta_1

9beta_2
	:decay
;learning_ratemtmumvmw#mx$my1mz2m{v|v}v~v#v?$v?1v?2v?"
	optimizer
X
0
1
2
3
#4
$5
16
27"
trackable_list_wrapper
X
0
1
2
3
#4
$5
16
27"
trackable_list_wrapper
 "
trackable_list_wrapper
?
<non_trainable_variables

=layers
>metrics
?layer_regularization_losses
@layer_metrics

	variables
trainable_variables
regularization_losses
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
+:)22eeg_cnn/conv1d/kernel
!:22eeg_cnn/conv1d/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Anon_trainable_variables

Blayers
Cmetrics
Dlayer_regularization_losses
Elayer_metrics
	variables
trainable_variables
regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Fnon_trainable_variables

Glayers
Hmetrics
Ilayer_regularization_losses
Jlayer_metrics
	variables
trainable_variables
regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-:+2d2eeg_cnn/conv1d_1/kernel
#:!d2eeg_cnn/conv1d_1/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Knon_trainable_variables

Llayers
Mmetrics
Nlayer_regularization_losses
Olayer_metrics
	variables
trainable_variables
regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Pnon_trainable_variables

Qlayers
Rmetrics
Slayer_regularization_losses
Tlayer_metrics
	variables
 trainable_variables
!regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.:,d?2eeg_cnn/conv1d_2/kernel
$:"?2eeg_cnn/conv1d_2/bias
.
#0
$1"
trackable_list_wrapper
.
#0
$1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Unon_trainable_variables

Vlayers
Wmetrics
Xlayer_regularization_losses
Ylayer_metrics
%	variables
&trainable_variables
'regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Znon_trainable_variables

[layers
\metrics
]layer_regularization_losses
^layer_metrics
)	variables
*trainable_variables
+regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
_non_trainable_variables

`layers
ametrics
blayer_regularization_losses
clayer_metrics
-	variables
.trainable_variables
/regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
):'	?	2eeg_cnn/dense_1/kernel
": 2eeg_cnn/dense_1/bias
.
10
21"
trackable_list_wrapper
.
10
21"
trackable_list_wrapper
 "
trackable_list_wrapper
?
dnon_trainable_variables

elayers
fmetrics
glayer_regularization_losses
hlayer_metrics
3	variables
4trainable_variables
5regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 "
trackable_list_wrapper
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
.
i0
j1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
N
	ktotal
	lcount
m	variables
n	keras_api"
_tf_keras_metric
^
	ototal
	pcount
q
_fn_kwargs
r	variables
s	keras_api"
_tf_keras_metric
:  (2total
:  (2count
.
k0
l1"
trackable_list_wrapper
-
m	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
o0
p1"
trackable_list_wrapper
-
r	variables"
_generic_user_object
0:.22Adam/eeg_cnn/conv1d/kernel/m
&:$22Adam/eeg_cnn/conv1d/bias/m
2:02d2Adam/eeg_cnn/conv1d_1/kernel/m
(:&d2Adam/eeg_cnn/conv1d_1/bias/m
3:1d?2Adam/eeg_cnn/conv1d_2/kernel/m
):'?2Adam/eeg_cnn/conv1d_2/bias/m
.:,	?	2Adam/eeg_cnn/dense_1/kernel/m
':%2Adam/eeg_cnn/dense_1/bias/m
0:.22Adam/eeg_cnn/conv1d/kernel/v
&:$22Adam/eeg_cnn/conv1d/bias/v
2:02d2Adam/eeg_cnn/conv1d_1/kernel/v
(:&d2Adam/eeg_cnn/conv1d_1/bias/v
3:1d?2Adam/eeg_cnn/conv1d_2/kernel/v
):'?2Adam/eeg_cnn/conv1d_2/bias/v
.:,	?	2Adam/eeg_cnn/dense_1/kernel/v
':%2Adam/eeg_cnn/dense_1/bias/v
?2?
'__inference_eeg_cnn_layer_call_fn_29271
'__inference_eeg_cnn_layer_call_fn_29431?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_eeg_cnn_layer_call_and_return_conditional_losses_29492
B__inference_eeg_cnn_layer_call_and_return_conditional_losses_29381?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
 __inference__wrapped_model_29081input_1"?
???
FullArgSpec
args? 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
&__inference_conv1d_layer_call_fn_29501?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
A__inference_conv1d_layer_call_and_return_conditional_losses_29517?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
-__inference_max_pooling1d_layer_call_fn_29522
-__inference_max_pooling1d_layer_call_fn_29527?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
H__inference_max_pooling1d_layer_call_and_return_conditional_losses_29535
H__inference_max_pooling1d_layer_call_and_return_conditional_losses_29543?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_conv1d_1_layer_call_fn_29552?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_conv1d_1_layer_call_and_return_conditional_losses_29568?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
/__inference_max_pooling1d_1_layer_call_fn_29573
/__inference_max_pooling1d_1_layer_call_fn_29578?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
J__inference_max_pooling1d_1_layer_call_and_return_conditional_losses_29586
J__inference_max_pooling1d_1_layer_call_and_return_conditional_losses_29594?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_conv1d_2_layer_call_fn_29603?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_conv1d_2_layer_call_and_return_conditional_losses_29619?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
/__inference_max_pooling1d_2_layer_call_fn_29624
/__inference_max_pooling1d_2_layer_call_fn_29629?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
J__inference_max_pooling1d_2_layer_call_and_return_conditional_losses_29637
J__inference_max_pooling1d_2_layer_call_and_return_conditional_losses_29645?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_flatten_layer_call_fn_29650?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_flatten_layer_call_and_return_conditional_losses_29656?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_dense_1_layer_call_fn_29665?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_dense_1_layer_call_and_return_conditional_losses_29676?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
#__inference_signature_wrapper_29410input_1"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
 __inference__wrapped_model_29081v#$125?2
+?(
&?#
input_1??????????
? "3?0
.
output_1"?
output_1??????????
C__inference_conv1d_1_layer_call_and_return_conditional_losses_29568d3?0
)?&
$?!
inputs?????????B2
? ")?&
?
0?????????@d
? ?
(__inference_conv1d_1_layer_call_fn_29552W3?0
)?&
$?!
inputs?????????B2
? "??????????@d?
C__inference_conv1d_2_layer_call_and_return_conditional_losses_29619e#$3?0
)?&
$?!
inputs?????????d
? "*?'
 ?
0??????????
? ?
(__inference_conv1d_2_layer_call_fn_29603X#$3?0
)?&
$?!
inputs?????????d
? "????????????
A__inference_conv1d_layer_call_and_return_conditional_losses_29517f4?1
*?'
%?"
inputs??????????
? "*?'
 ?
0??????????2
? ?
&__inference_conv1d_layer_call_fn_29501Y4?1
*?'
%?"
inputs??????????
? "???????????2?
B__inference_dense_1_layer_call_and_return_conditional_losses_29676]120?-
&?#
!?
inputs??????????	
? "%?"
?
0?????????
? {
'__inference_dense_1_layer_call_fn_29665P120?-
&?#
!?
inputs??????????	
? "???????????
B__inference_eeg_cnn_layer_call_and_return_conditional_losses_29381h#$125?2
+?(
&?#
input_1??????????
? "%?"
?
0?????????
? ?
B__inference_eeg_cnn_layer_call_and_return_conditional_losses_29492g#$124?1
*?'
%?"
inputs??????????
? "%?"
?
0?????????
? ?
'__inference_eeg_cnn_layer_call_fn_29271[#$125?2
+?(
&?#
input_1??????????
? "???????????
'__inference_eeg_cnn_layer_call_fn_29431Z#$124?1
*?'
%?"
inputs??????????
? "???????????
B__inference_flatten_layer_call_and_return_conditional_losses_29656^4?1
*?'
%?"
inputs??????????
? "&?#
?
0??????????	
? |
'__inference_flatten_layer_call_fn_29650Q4?1
*?'
%?"
inputs??????????
? "???????????	?
J__inference_max_pooling1d_1_layer_call_and_return_conditional_losses_29586?E?B
;?8
6?3
inputs'???????????????????????????
? ";?8
1?.
0'???????????????????????????
? ?
J__inference_max_pooling1d_1_layer_call_and_return_conditional_losses_29594`3?0
)?&
$?!
inputs?????????@d
? ")?&
?
0?????????d
? ?
/__inference_max_pooling1d_1_layer_call_fn_29573wE?B
;?8
6?3
inputs'???????????????????????????
? ".?+'????????????????????????????
/__inference_max_pooling1d_1_layer_call_fn_29578S3?0
)?&
$?!
inputs?????????@d
? "??????????d?
J__inference_max_pooling1d_2_layer_call_and_return_conditional_losses_29637?E?B
;?8
6?3
inputs'???????????????????????????
? ";?8
1?.
0'???????????????????????????
? ?
J__inference_max_pooling1d_2_layer_call_and_return_conditional_losses_29645b4?1
*?'
%?"
inputs??????????
? "*?'
 ?
0??????????
? ?
/__inference_max_pooling1d_2_layer_call_fn_29624wE?B
;?8
6?3
inputs'???????????????????????????
? ".?+'????????????????????????????
/__inference_max_pooling1d_2_layer_call_fn_29629U4?1
*?'
%?"
inputs??????????
? "????????????
H__inference_max_pooling1d_layer_call_and_return_conditional_losses_29535?E?B
;?8
6?3
inputs'???????????????????????????
? ";?8
1?.
0'???????????????????????????
? ?
H__inference_max_pooling1d_layer_call_and_return_conditional_losses_29543a4?1
*?'
%?"
inputs??????????2
? ")?&
?
0?????????B2
? ?
-__inference_max_pooling1d_layer_call_fn_29522wE?B
;?8
6?3
inputs'???????????????????????????
? ".?+'????????????????????????????
-__inference_max_pooling1d_layer_call_fn_29527T4?1
*?'
%?"
inputs??????????2
? "??????????B2?
#__inference_signature_wrapper_29410?#$12@?=
? 
6?3
1
input_1&?#
input_1??????????"3?0
.
output_1"?
output_1?????????