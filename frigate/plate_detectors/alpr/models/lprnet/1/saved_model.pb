ů
EÝD
W
AddN
inputs"T*N
sum"T"
Nint(0"!
Ttype:
2	
D
AddV2
x"T
y"T
z"T"
Ttype:
2	
E
AssignAddVariableOp
resource
value"dtype"
dtypetype
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 
ź
AvgPool

value"T
output"T"
ksize	list(int)(0"
strides	list(int)(0""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW"
Ttype:
2
Ő
AvgPoolGrad
orig_input_shape	
grad"T
output"T"
ksize	list(int)(0"
strides	list(int)(0""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW"
Ttype:
2
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
~
BiasAddGrad
out_backprop"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
R
BroadcastGradientArgs
s0"T
s1"T
r0"T
r1"T"
Ttype0:
2	
Z
BroadcastTo

input"T
shape"Tidx
output"T"	
Ttype"
Tidxtype0:
2	

CTCBeamSearchDecoder
inputs"T
sequence_length
decoded_indices	*	top_paths
decoded_values	*	top_paths
decoded_shape	*	top_paths
log_probability"T"

beam_widthint(0"
	top_pathsint(0"
merge_repeatedbool("
Ttype0:
2
ü
CTCLoss
inputs"T
labels_indices	
labels_values
sequence_length	
loss"T
gradient"T"(
preprocess_collapse_repeatedbool( "
ctc_merge_repeatedbool("-
!ignore_longer_outputs_than_inputsbool( "
Ttype0:
2
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
I
ConcatOffset

concat_dim
shape*N
offset*N"
Nint(0
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype

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
Ŕ
Conv2DBackpropFilter

input"T
filter_sizes
out_backprop"T
output"T"
Ttype:
2"
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
Ŕ
Conv2DBackpropInput
input_sizes
filter"T
out_backprop"T
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
S
DynamicStitch
indices*N
data"T*N
merged"T"
Nint(0"	
Ttype
š
EditDistance
hypothesis_indices	
hypothesis_values"T
hypothesis_shape	
truth_indices	
truth_values"T
truth_shape	

output"
	normalizebool("	
Ttype
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
,
Floor
x"T
y"T"
Ttype:
2
A
FloorDiv
x"T
y"T
z"T"
Ttype:
2	
?
FloorMod
x"T
y"T
z"T"
Ttype:
2	
ů
FusedBatchNormGradV3

y_backprop"T
x"T	
scale
reserve_space_1"U
reserve_space_2"U
reserve_space_3"U

x_backprop"T
scale_backprop"U
offset_backprop"U
reserve_space_4"U
reserve_space_5"U"
Ttype:
2"
Utype:
2"
epsilonfloat%ˇŃ8";
data_formatstringNHWC:
NHWCNCHWNDHWCNCDHW"
is_trainingbool(
ű
FusedBatchNormV3
x"T

scale"U
offset"U	
mean"U
variance"U
y"T

batch_mean"U
batch_variance"U
reserve_space_1"U
reserve_space_2"U
reserve_space_3"U"
Ttype:
2"
Utype:
2"
epsilonfloat%ˇŃ8"&
exponential_avg_factorfloat%  ?";
data_formatstringNHWC:
NHWCNCHWNDHWCNCDHW"
is_trainingbool(
.
Identity

input"T
output"T"	
Ttype
:
InvertPermutation
x"T
y"T"
Ttype0:
2	

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

MaxPoolGrad

orig_input"T
orig_output"T	
grad"T
output"T"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW"
Ttype0:
2	
>
Maximum
x"T
y"T
z"T"
Ttype:
2	

Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	

MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( 
?
Mul
x"T
y"T
z"T"
Ttype:
2	
0
Neg
x"T
y"T"
Ttype:
2
	
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
X
PlaceholderWithDefault
input"dtype
output"dtype"
dtypetype"
shapeshape
8
Pow
x"T
y"T
z"T"
Ttype:
2
	
L
PreventGradient

input"T
output"T"	
Ttype"
messagestring 

Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
~
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	
f
Range
start"Tidx
limit"Tidx
delta"Tidx
output"Tidx" 
Tidxtype0:
2
	
)
Rank

input"T

output"	
Ttype
@
ReadVariableOp
resource
value"dtype"
dtypetype
@
RealDiv
x"T
y"T
z"T"
Ttype:
2	
E
Relu
features"T
activations"T"
Ttype:
2	
V
ReluGrad
	gradients"T
features"T
	backprops"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
ŕ
ResourceApplyAdam
var
m
v
beta1_power"T
beta2_power"T
lr"T

beta1"T

beta2"T
epsilon"T	
grad"T" 
Ttype:
2	"
use_lockingbool( "
use_nesterovbool( 
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
e
ShapeN
input"T*N
output"out_type*N"
Nint(0"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
a
Slice

input"T
begin"Index
size"Index
output"T"	
Ttype"
Indextype:
2	
ź
SparseToDense
sparse_indices"Tindices
output_shape"Tindices
sparse_values"T
default_value"T

dense"T"
validate_indicesbool("	
Ttype"
Tindicestype:
2	
7
Square
x"T
y"T"
Ttype:
2	
@
StaticRegexFullMatch	
input

output
"
patternstring
÷
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	

Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 
9
VarIsInitializedOp
resource
is_initialized

&
	ZerosLike
x"T
y"T"	
Ttype"serve*2.11.02v2.11.0-rc2-17-gd5b57ca93e58ůý
y
inputsPlaceholder*/
_output_shapes
:˙˙˙˙˙˙˙˙˙^*
dtype0*$
shape:˙˙˙˙˙˙˙˙˙^
f
PlaceholderPlaceholder*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0	*
shape:˙˙˙˙˙˙˙˙˙
h
Placeholder_1Placeholder*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0*
shape:˙˙˙˙˙˙˙˙˙

Placeholder_2Placeholder*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
dtype0	*%
shape:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙

(conv1/w/Initializer/random_uniform/shapeConst*
_class
loc:@conv1/w*
_output_shapes
:*
dtype0*%
valueB"         @   

&conv1/w/Initializer/random_uniform/minConst*
_class
loc:@conv1/w*
_output_shapes
: *
dtype0*
valueB
 *8JĚ˝

&conv1/w/Initializer/random_uniform/maxConst*
_class
loc:@conv1/w*
_output_shapes
: *
dtype0*
valueB
 *8JĚ=
Ĺ
0conv1/w/Initializer/random_uniform/RandomUniformRandomUniform(conv1/w/Initializer/random_uniform/shape*
T0*
_class
loc:@conv1/w*&
_output_shapes
:@*
dtype0
ş
&conv1/w/Initializer/random_uniform/subSub&conv1/w/Initializer/random_uniform/max&conv1/w/Initializer/random_uniform/min*
T0*
_class
loc:@conv1/w*
_output_shapes
: 
Ô
&conv1/w/Initializer/random_uniform/mulMul0conv1/w/Initializer/random_uniform/RandomUniform&conv1/w/Initializer/random_uniform/sub*
T0*
_class
loc:@conv1/w*&
_output_shapes
:@
Č
"conv1/w/Initializer/random_uniformAddV2&conv1/w/Initializer/random_uniform/mul&conv1/w/Initializer/random_uniform/min*
T0*
_class
loc:@conv1/w*&
_output_shapes
:@

conv1/wVarHandleOp*
_class
loc:@conv1/w*
_output_shapes
: *
dtype0*
shape:@*
shared_name	conv1/w
_
(conv1/w/IsInitialized/VarIsInitializedOpVarIsInitializedOpconv1/w*
_output_shapes
: 

conv1/w/AssignAssignVariableOpconv1/w"conv1/w/Initializer/random_uniform*&
 _has_manual_control_dependencies(*
dtype0
k
conv1/w/Read/ReadVariableOpReadVariableOpconv1/w*&
_output_shapes
:@*
dtype0

conv1/b/Initializer/ConstConst*
_class
loc:@conv1/b*
_output_shapes
:@*
dtype0*
valueB@*    

conv1/bVarHandleOp*
_class
loc:@conv1/b*
_output_shapes
: *
dtype0*
shape:@*
shared_name	conv1/b
_
(conv1/b/IsInitialized/VarIsInitializedOpVarIsInitializedOpconv1/b*
_output_shapes
: 
{
conv1/b/AssignAssignVariableOpconv1/bconv1/b/Initializer/Const*&
 _has_manual_control_dependencies(*
dtype0
_
conv1/b/Read/ReadVariableOpReadVariableOpconv1/b*
_output_shapes
:@*
dtype0
k
conv1/Conv2D/ReadVariableOpReadVariableOpconv1/w*&
_output_shapes
:@*
dtype0

conv1/Conv2DConv2Dinputsconv1/Conv2D/ReadVariableOp*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙^@*
paddingSAME*
strides

`
conv1/BiasAdd/ReadVariableOpReadVariableOpconv1/b*
_output_shapes
:@*
dtype0
~
conv1/BiasAddBiasAddconv1/Conv2Dconv1/BiasAdd/ReadVariableOp*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙^@
Ľ
*batch_normalization/gamma/Initializer/onesConst*,
_class"
 loc:@batch_normalization/gamma*
_output_shapes
:@*
dtype0*
valueB@*  ?
¸
batch_normalization/gammaVarHandleOp*,
_class"
 loc:@batch_normalization/gamma*
_output_shapes
: *
dtype0*
shape:@**
shared_namebatch_normalization/gamma

:batch_normalization/gamma/IsInitialized/VarIsInitializedOpVarIsInitializedOpbatch_normalization/gamma*
_output_shapes
: 
°
 batch_normalization/gamma/AssignAssignVariableOpbatch_normalization/gamma*batch_normalization/gamma/Initializer/ones*&
 _has_manual_control_dependencies(*
dtype0

-batch_normalization/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization/gamma*
_output_shapes
:@*
dtype0
¤
*batch_normalization/beta/Initializer/zerosConst*+
_class!
loc:@batch_normalization/beta*
_output_shapes
:@*
dtype0*
valueB@*    
ľ
batch_normalization/betaVarHandleOp*+
_class!
loc:@batch_normalization/beta*
_output_shapes
: *
dtype0*
shape:@*)
shared_namebatch_normalization/beta

9batch_normalization/beta/IsInitialized/VarIsInitializedOpVarIsInitializedOpbatch_normalization/beta*
_output_shapes
: 
Ž
batch_normalization/beta/AssignAssignVariableOpbatch_normalization/beta*batch_normalization/beta/Initializer/zeros*&
 _has_manual_control_dependencies(*
dtype0

,batch_normalization/beta/Read/ReadVariableOpReadVariableOpbatch_normalization/beta*
_output_shapes
:@*
dtype0
˛
1batch_normalization/moving_mean/Initializer/zerosConst*2
_class(
&$loc:@batch_normalization/moving_mean*
_output_shapes
:@*
dtype0*
valueB@*    
Ę
batch_normalization/moving_meanVarHandleOp*2
_class(
&$loc:@batch_normalization/moving_mean*
_output_shapes
: *
dtype0*
shape:@*0
shared_name!batch_normalization/moving_mean

@batch_normalization/moving_mean/IsInitialized/VarIsInitializedOpVarIsInitializedOpbatch_normalization/moving_mean*
_output_shapes
: 
Ă
&batch_normalization/moving_mean/AssignAssignVariableOpbatch_normalization/moving_mean1batch_normalization/moving_mean/Initializer/zeros*&
 _has_manual_control_dependencies(*
dtype0

3batch_normalization/moving_mean/Read/ReadVariableOpReadVariableOpbatch_normalization/moving_mean*
_output_shapes
:@*
dtype0
š
4batch_normalization/moving_variance/Initializer/onesConst*6
_class,
*(loc:@batch_normalization/moving_variance*
_output_shapes
:@*
dtype0*
valueB@*  ?
Ö
#batch_normalization/moving_varianceVarHandleOp*6
_class,
*(loc:@batch_normalization/moving_variance*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#batch_normalization/moving_variance

Dbatch_normalization/moving_variance/IsInitialized/VarIsInitializedOpVarIsInitializedOp#batch_normalization/moving_variance*
_output_shapes
: 
Î
*batch_normalization/moving_variance/AssignAssignVariableOp#batch_normalization/moving_variance4batch_normalization/moving_variance/Initializer/ones*&
 _has_manual_control_dependencies(*
dtype0

7batch_normalization/moving_variance/Read/ReadVariableOpReadVariableOp#batch_normalization/moving_variance*
_output_shapes
:@*
dtype0
x
"batch_normalization/ReadVariableOpReadVariableOpbatch_normalization/gamma*
_output_shapes
:@*
dtype0
y
$batch_normalization/ReadVariableOp_1ReadVariableOpbatch_normalization/beta*
_output_shapes
:@*
dtype0

3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOpbatch_normalization/moving_mean*
_output_shapes
:@*
dtype0

5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp#batch_normalization/moving_variance*
_output_shapes
:@*
dtype0

$batch_normalization/FusedBatchNormV3FusedBatchNormV3conv1/BiasAdd"batch_normalization/ReadVariableOp$batch_normalization/ReadVariableOp_13batch_normalization/FusedBatchNormV3/ReadVariableOp5batch_normalization/FusedBatchNormV3/ReadVariableOp_1*
T0*
U0*K
_output_shapes9
7:˙˙˙˙˙˙˙˙˙^@:@:@:@:@:*
epsilon%o:*
is_training( 
l
ReluRelu$batch_normalization/FusedBatchNormV3*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙^@

MaxPoolMaxPoolRelu*/
_output_shapes
:˙˙˙˙˙˙˙˙˙^@*
ksize
*
paddingSAME*
strides

§
-sbb1/conv1/w/Initializer/random_uniform/shapeConst*
_class
loc:@sbb1/conv1/w*
_output_shapes
:*
dtype0*%
valueB"      @       

+sbb1/conv1/w/Initializer/random_uniform/minConst*
_class
loc:@sbb1/conv1/w*
_output_shapes
: *
dtype0*
valueB
 *  ž

+sbb1/conv1/w/Initializer/random_uniform/maxConst*
_class
loc:@sbb1/conv1/w*
_output_shapes
: *
dtype0*
valueB
 *  >
Ô
5sbb1/conv1/w/Initializer/random_uniform/RandomUniformRandomUniform-sbb1/conv1/w/Initializer/random_uniform/shape*
T0*
_class
loc:@sbb1/conv1/w*&
_output_shapes
:@ *
dtype0
Î
+sbb1/conv1/w/Initializer/random_uniform/subSub+sbb1/conv1/w/Initializer/random_uniform/max+sbb1/conv1/w/Initializer/random_uniform/min*
T0*
_class
loc:@sbb1/conv1/w*
_output_shapes
: 
č
+sbb1/conv1/w/Initializer/random_uniform/mulMul5sbb1/conv1/w/Initializer/random_uniform/RandomUniform+sbb1/conv1/w/Initializer/random_uniform/sub*
T0*
_class
loc:@sbb1/conv1/w*&
_output_shapes
:@ 
Ü
'sbb1/conv1/w/Initializer/random_uniformAddV2+sbb1/conv1/w/Initializer/random_uniform/mul+sbb1/conv1/w/Initializer/random_uniform/min*
T0*
_class
loc:@sbb1/conv1/w*&
_output_shapes
:@ 

sbb1/conv1/wVarHandleOp*
_class
loc:@sbb1/conv1/w*
_output_shapes
: *
dtype0*
shape:@ *
shared_namesbb1/conv1/w
i
-sbb1/conv1/w/IsInitialized/VarIsInitializedOpVarIsInitializedOpsbb1/conv1/w*
_output_shapes
: 

sbb1/conv1/w/AssignAssignVariableOpsbb1/conv1/w'sbb1/conv1/w/Initializer/random_uniform*&
 _has_manual_control_dependencies(*
dtype0
u
 sbb1/conv1/w/Read/ReadVariableOpReadVariableOpsbb1/conv1/w*&
_output_shapes
:@ *
dtype0

sbb1/conv1/b/Initializer/ConstConst*
_class
loc:@sbb1/conv1/b*
_output_shapes
: *
dtype0*
valueB *    

sbb1/conv1/bVarHandleOp*
_class
loc:@sbb1/conv1/b*
_output_shapes
: *
dtype0*
shape: *
shared_namesbb1/conv1/b
i
-sbb1/conv1/b/IsInitialized/VarIsInitializedOpVarIsInitializedOpsbb1/conv1/b*
_output_shapes
: 

sbb1/conv1/b/AssignAssignVariableOpsbb1/conv1/bsbb1/conv1/b/Initializer/Const*&
 _has_manual_control_dependencies(*
dtype0
i
 sbb1/conv1/b/Read/ReadVariableOpReadVariableOpsbb1/conv1/b*
_output_shapes
: *
dtype0
u
 sbb1/conv1/Conv2D/ReadVariableOpReadVariableOpsbb1/conv1/w*&
_output_shapes
:@ *
dtype0
¨
sbb1/conv1/Conv2DConv2DMaxPool sbb1/conv1/Conv2D/ReadVariableOp*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙^ *
paddingSAME*
strides

j
!sbb1/conv1/BiasAdd/ReadVariableOpReadVariableOpsbb1/conv1/b*
_output_shapes
: *
dtype0

sbb1/conv1/BiasAddBiasAddsbb1/conv1/Conv2D!sbb1/conv1/BiasAdd/ReadVariableOp*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙^ 
_
	sbb1/ReluRelusbb1/conv1/BiasAdd*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙^ 
§
-sbb1/conv2/w/Initializer/random_uniform/shapeConst*
_class
loc:@sbb1/conv2/w*
_output_shapes
:*
dtype0*%
valueB"              

+sbb1/conv2/w/Initializer/random_uniform/minConst*
_class
loc:@sbb1/conv2/w*
_output_shapes
: *
dtype0*
valueB
 *ó5ž

+sbb1/conv2/w/Initializer/random_uniform/maxConst*
_class
loc:@sbb1/conv2/w*
_output_shapes
: *
dtype0*
valueB
 *ó5>
Ô
5sbb1/conv2/w/Initializer/random_uniform/RandomUniformRandomUniform-sbb1/conv2/w/Initializer/random_uniform/shape*
T0*
_class
loc:@sbb1/conv2/w*&
_output_shapes
:  *
dtype0
Î
+sbb1/conv2/w/Initializer/random_uniform/subSub+sbb1/conv2/w/Initializer/random_uniform/max+sbb1/conv2/w/Initializer/random_uniform/min*
T0*
_class
loc:@sbb1/conv2/w*
_output_shapes
: 
č
+sbb1/conv2/w/Initializer/random_uniform/mulMul5sbb1/conv2/w/Initializer/random_uniform/RandomUniform+sbb1/conv2/w/Initializer/random_uniform/sub*
T0*
_class
loc:@sbb1/conv2/w*&
_output_shapes
:  
Ü
'sbb1/conv2/w/Initializer/random_uniformAddV2+sbb1/conv2/w/Initializer/random_uniform/mul+sbb1/conv2/w/Initializer/random_uniform/min*
T0*
_class
loc:@sbb1/conv2/w*&
_output_shapes
:  

sbb1/conv2/wVarHandleOp*
_class
loc:@sbb1/conv2/w*
_output_shapes
: *
dtype0*
shape:  *
shared_namesbb1/conv2/w
i
-sbb1/conv2/w/IsInitialized/VarIsInitializedOpVarIsInitializedOpsbb1/conv2/w*
_output_shapes
: 

sbb1/conv2/w/AssignAssignVariableOpsbb1/conv2/w'sbb1/conv2/w/Initializer/random_uniform*&
 _has_manual_control_dependencies(*
dtype0
u
 sbb1/conv2/w/Read/ReadVariableOpReadVariableOpsbb1/conv2/w*&
_output_shapes
:  *
dtype0

sbb1/conv2/b/Initializer/ConstConst*
_class
loc:@sbb1/conv2/b*
_output_shapes
: *
dtype0*
valueB *    

sbb1/conv2/bVarHandleOp*
_class
loc:@sbb1/conv2/b*
_output_shapes
: *
dtype0*
shape: *
shared_namesbb1/conv2/b
i
-sbb1/conv2/b/IsInitialized/VarIsInitializedOpVarIsInitializedOpsbb1/conv2/b*
_output_shapes
: 

sbb1/conv2/b/AssignAssignVariableOpsbb1/conv2/bsbb1/conv2/b/Initializer/Const*&
 _has_manual_control_dependencies(*
dtype0
i
 sbb1/conv2/b/Read/ReadVariableOpReadVariableOpsbb1/conv2/b*
_output_shapes
: *
dtype0
u
 sbb1/conv2/Conv2D/ReadVariableOpReadVariableOpsbb1/conv2/w*&
_output_shapes
:  *
dtype0
Ş
sbb1/conv2/Conv2DConv2D	sbb1/Relu sbb1/conv2/Conv2D/ReadVariableOp*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙^ *
paddingSAME*
strides

j
!sbb1/conv2/BiasAdd/ReadVariableOpReadVariableOpsbb1/conv2/b*
_output_shapes
: *
dtype0

sbb1/conv2/BiasAddBiasAddsbb1/conv2/Conv2D!sbb1/conv2/BiasAdd/ReadVariableOp*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙^ 
a
sbb1/Relu_1Relusbb1/conv2/BiasAdd*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙^ 
§
-sbb1/conv3/w/Initializer/random_uniform/shapeConst*
_class
loc:@sbb1/conv3/w*
_output_shapes
:*
dtype0*%
valueB"              

+sbb1/conv3/w/Initializer/random_uniform/minConst*
_class
loc:@sbb1/conv3/w*
_output_shapes
: *
dtype0*
valueB
 *ó5ž

+sbb1/conv3/w/Initializer/random_uniform/maxConst*
_class
loc:@sbb1/conv3/w*
_output_shapes
: *
dtype0*
valueB
 *ó5>
Ô
5sbb1/conv3/w/Initializer/random_uniform/RandomUniformRandomUniform-sbb1/conv3/w/Initializer/random_uniform/shape*
T0*
_class
loc:@sbb1/conv3/w*&
_output_shapes
:  *
dtype0
Î
+sbb1/conv3/w/Initializer/random_uniform/subSub+sbb1/conv3/w/Initializer/random_uniform/max+sbb1/conv3/w/Initializer/random_uniform/min*
T0*
_class
loc:@sbb1/conv3/w*
_output_shapes
: 
č
+sbb1/conv3/w/Initializer/random_uniform/mulMul5sbb1/conv3/w/Initializer/random_uniform/RandomUniform+sbb1/conv3/w/Initializer/random_uniform/sub*
T0*
_class
loc:@sbb1/conv3/w*&
_output_shapes
:  
Ü
'sbb1/conv3/w/Initializer/random_uniformAddV2+sbb1/conv3/w/Initializer/random_uniform/mul+sbb1/conv3/w/Initializer/random_uniform/min*
T0*
_class
loc:@sbb1/conv3/w*&
_output_shapes
:  

sbb1/conv3/wVarHandleOp*
_class
loc:@sbb1/conv3/w*
_output_shapes
: *
dtype0*
shape:  *
shared_namesbb1/conv3/w
i
-sbb1/conv3/w/IsInitialized/VarIsInitializedOpVarIsInitializedOpsbb1/conv3/w*
_output_shapes
: 

sbb1/conv3/w/AssignAssignVariableOpsbb1/conv3/w'sbb1/conv3/w/Initializer/random_uniform*&
 _has_manual_control_dependencies(*
dtype0
u
 sbb1/conv3/w/Read/ReadVariableOpReadVariableOpsbb1/conv3/w*&
_output_shapes
:  *
dtype0

sbb1/conv3/b/Initializer/ConstConst*
_class
loc:@sbb1/conv3/b*
_output_shapes
: *
dtype0*
valueB *    

sbb1/conv3/bVarHandleOp*
_class
loc:@sbb1/conv3/b*
_output_shapes
: *
dtype0*
shape: *
shared_namesbb1/conv3/b
i
-sbb1/conv3/b/IsInitialized/VarIsInitializedOpVarIsInitializedOpsbb1/conv3/b*
_output_shapes
: 

sbb1/conv3/b/AssignAssignVariableOpsbb1/conv3/bsbb1/conv3/b/Initializer/Const*&
 _has_manual_control_dependencies(*
dtype0
i
 sbb1/conv3/b/Read/ReadVariableOpReadVariableOpsbb1/conv3/b*
_output_shapes
: *
dtype0
u
 sbb1/conv3/Conv2D/ReadVariableOpReadVariableOpsbb1/conv3/w*&
_output_shapes
:  *
dtype0
Ź
sbb1/conv3/Conv2DConv2Dsbb1/Relu_1 sbb1/conv3/Conv2D/ReadVariableOp*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙^ *
paddingSAME*
strides

j
!sbb1/conv3/BiasAdd/ReadVariableOpReadVariableOpsbb1/conv3/b*
_output_shapes
: *
dtype0

sbb1/conv3/BiasAddBiasAddsbb1/conv3/Conv2D!sbb1/conv3/BiasAdd/ReadVariableOp*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙^ 
a
sbb1/Relu_2Relusbb1/conv3/BiasAdd*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙^ 
§
-sbb1/conv4/w/Initializer/random_uniform/shapeConst*
_class
loc:@sbb1/conv4/w*
_output_shapes
:*
dtype0*%
valueB"             

+sbb1/conv4/w/Initializer/random_uniform/minConst*
_class
loc:@sbb1/conv4/w*
_output_shapes
: *
dtype0*
valueB
 *řKFž

+sbb1/conv4/w/Initializer/random_uniform/maxConst*
_class
loc:@sbb1/conv4/w*
_output_shapes
: *
dtype0*
valueB
 *řKF>
Ő
5sbb1/conv4/w/Initializer/random_uniform/RandomUniformRandomUniform-sbb1/conv4/w/Initializer/random_uniform/shape*
T0*
_class
loc:@sbb1/conv4/w*'
_output_shapes
: *
dtype0
Î
+sbb1/conv4/w/Initializer/random_uniform/subSub+sbb1/conv4/w/Initializer/random_uniform/max+sbb1/conv4/w/Initializer/random_uniform/min*
T0*
_class
loc:@sbb1/conv4/w*
_output_shapes
: 
é
+sbb1/conv4/w/Initializer/random_uniform/mulMul5sbb1/conv4/w/Initializer/random_uniform/RandomUniform+sbb1/conv4/w/Initializer/random_uniform/sub*
T0*
_class
loc:@sbb1/conv4/w*'
_output_shapes
: 
Ý
'sbb1/conv4/w/Initializer/random_uniformAddV2+sbb1/conv4/w/Initializer/random_uniform/mul+sbb1/conv4/w/Initializer/random_uniform/min*
T0*
_class
loc:@sbb1/conv4/w*'
_output_shapes
: 

sbb1/conv4/wVarHandleOp*
_class
loc:@sbb1/conv4/w*
_output_shapes
: *
dtype0*
shape: *
shared_namesbb1/conv4/w
i
-sbb1/conv4/w/IsInitialized/VarIsInitializedOpVarIsInitializedOpsbb1/conv4/w*
_output_shapes
: 

sbb1/conv4/w/AssignAssignVariableOpsbb1/conv4/w'sbb1/conv4/w/Initializer/random_uniform*&
 _has_manual_control_dependencies(*
dtype0
v
 sbb1/conv4/w/Read/ReadVariableOpReadVariableOpsbb1/conv4/w*'
_output_shapes
: *
dtype0

sbb1/conv4/b/Initializer/ConstConst*
_class
loc:@sbb1/conv4/b*
_output_shapes	
:*
dtype0*
valueB*    

sbb1/conv4/bVarHandleOp*
_class
loc:@sbb1/conv4/b*
_output_shapes
: *
dtype0*
shape:*
shared_namesbb1/conv4/b
i
-sbb1/conv4/b/IsInitialized/VarIsInitializedOpVarIsInitializedOpsbb1/conv4/b*
_output_shapes
: 

sbb1/conv4/b/AssignAssignVariableOpsbb1/conv4/bsbb1/conv4/b/Initializer/Const*&
 _has_manual_control_dependencies(*
dtype0
j
 sbb1/conv4/b/Read/ReadVariableOpReadVariableOpsbb1/conv4/b*
_output_shapes	
:*
dtype0
v
 sbb1/conv4/Conv2D/ReadVariableOpReadVariableOpsbb1/conv4/w*'
_output_shapes
: *
dtype0
­
sbb1/conv4/Conv2DConv2Dsbb1/Relu_2 sbb1/conv4/Conv2D/ReadVariableOp*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙^*
paddingSAME*
strides

k
!sbb1/conv4/BiasAdd/ReadVariableOpReadVariableOpsbb1/conv4/b*
_output_shapes	
:*
dtype0

sbb1/conv4/BiasAddBiasAddsbb1/conv4/Conv2D!sbb1/conv4/BiasAdd/ReadVariableOp*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙^
ą
/sbb1/batch_normalization/gamma/Initializer/onesConst*1
_class'
%#loc:@sbb1/batch_normalization/gamma*
_output_shapes	
:*
dtype0*
valueB*  ?
Č
sbb1/batch_normalization/gammaVarHandleOp*1
_class'
%#loc:@sbb1/batch_normalization/gamma*
_output_shapes
: *
dtype0*
shape:*/
shared_name sbb1/batch_normalization/gamma

?sbb1/batch_normalization/gamma/IsInitialized/VarIsInitializedOpVarIsInitializedOpsbb1/batch_normalization/gamma*
_output_shapes
: 
ż
%sbb1/batch_normalization/gamma/AssignAssignVariableOpsbb1/batch_normalization/gamma/sbb1/batch_normalization/gamma/Initializer/ones*&
 _has_manual_control_dependencies(*
dtype0

2sbb1/batch_normalization/gamma/Read/ReadVariableOpReadVariableOpsbb1/batch_normalization/gamma*
_output_shapes	
:*
dtype0
°
/sbb1/batch_normalization/beta/Initializer/zerosConst*0
_class&
$"loc:@sbb1/batch_normalization/beta*
_output_shapes	
:*
dtype0*
valueB*    
Ĺ
sbb1/batch_normalization/betaVarHandleOp*0
_class&
$"loc:@sbb1/batch_normalization/beta*
_output_shapes
: *
dtype0*
shape:*.
shared_namesbb1/batch_normalization/beta

>sbb1/batch_normalization/beta/IsInitialized/VarIsInitializedOpVarIsInitializedOpsbb1/batch_normalization/beta*
_output_shapes
: 
˝
$sbb1/batch_normalization/beta/AssignAssignVariableOpsbb1/batch_normalization/beta/sbb1/batch_normalization/beta/Initializer/zeros*&
 _has_manual_control_dependencies(*
dtype0

1sbb1/batch_normalization/beta/Read/ReadVariableOpReadVariableOpsbb1/batch_normalization/beta*
_output_shapes	
:*
dtype0
ž
6sbb1/batch_normalization/moving_mean/Initializer/zerosConst*7
_class-
+)loc:@sbb1/batch_normalization/moving_mean*
_output_shapes	
:*
dtype0*
valueB*    
Ú
$sbb1/batch_normalization/moving_meanVarHandleOp*7
_class-
+)loc:@sbb1/batch_normalization/moving_mean*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$sbb1/batch_normalization/moving_mean

Esbb1/batch_normalization/moving_mean/IsInitialized/VarIsInitializedOpVarIsInitializedOp$sbb1/batch_normalization/moving_mean*
_output_shapes
: 
Ň
+sbb1/batch_normalization/moving_mean/AssignAssignVariableOp$sbb1/batch_normalization/moving_mean6sbb1/batch_normalization/moving_mean/Initializer/zeros*&
 _has_manual_control_dependencies(*
dtype0

8sbb1/batch_normalization/moving_mean/Read/ReadVariableOpReadVariableOp$sbb1/batch_normalization/moving_mean*
_output_shapes	
:*
dtype0
Ĺ
9sbb1/batch_normalization/moving_variance/Initializer/onesConst*;
_class1
/-loc:@sbb1/batch_normalization/moving_variance*
_output_shapes	
:*
dtype0*
valueB*  ?
ć
(sbb1/batch_normalization/moving_varianceVarHandleOp*;
_class1
/-loc:@sbb1/batch_normalization/moving_variance*
_output_shapes
: *
dtype0*
shape:*9
shared_name*(sbb1/batch_normalization/moving_variance
Ą
Isbb1/batch_normalization/moving_variance/IsInitialized/VarIsInitializedOpVarIsInitializedOp(sbb1/batch_normalization/moving_variance*
_output_shapes
: 
Ý
/sbb1/batch_normalization/moving_variance/AssignAssignVariableOp(sbb1/batch_normalization/moving_variance9sbb1/batch_normalization/moving_variance/Initializer/ones*&
 _has_manual_control_dependencies(*
dtype0
˘
<sbb1/batch_normalization/moving_variance/Read/ReadVariableOpReadVariableOp(sbb1/batch_normalization/moving_variance*
_output_shapes	
:*
dtype0

'sbb1/batch_normalization/ReadVariableOpReadVariableOpsbb1/batch_normalization/gamma*
_output_shapes	
:*
dtype0

)sbb1/batch_normalization/ReadVariableOp_1ReadVariableOpsbb1/batch_normalization/beta*
_output_shapes	
:*
dtype0

8sbb1/batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp$sbb1/batch_normalization/moving_mean*
_output_shapes	
:*
dtype0
 
:sbb1/batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp(sbb1/batch_normalization/moving_variance*
_output_shapes	
:*
dtype0
¤
)sbb1/batch_normalization/FusedBatchNormV3FusedBatchNormV3sbb1/conv4/BiasAdd'sbb1/batch_normalization/ReadVariableOp)sbb1/batch_normalization/ReadVariableOp_18sbb1/batch_normalization/FusedBatchNormV3/ReadVariableOp:sbb1/batch_normalization/FusedBatchNormV3/ReadVariableOp_1*
T0*
U0*P
_output_shapes>
<:˙˙˙˙˙˙˙˙˙^:::::*
epsilon%o:*
is_training( 
y
sbb1/Relu_3Relu)sbb1/batch_normalization/FusedBatchNormV3*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙^

	MaxPool_1MaxPoolsbb1/Relu_3*0
_output_shapes
:˙˙˙˙˙˙˙˙˙/*
ksize
*
paddingSAME*
strides

§
-sbb2/conv1/w/Initializer/random_uniform/shapeConst*
_class
loc:@sbb2/conv1/w*
_output_shapes
:*
dtype0*%
valueB"         @   

+sbb2/conv1/w/Initializer/random_uniform/minConst*
_class
loc:@sbb2/conv1/w*
_output_shapes
: *
dtype0*
valueB
 *ó5ž

+sbb2/conv1/w/Initializer/random_uniform/maxConst*
_class
loc:@sbb2/conv1/w*
_output_shapes
: *
dtype0*
valueB
 *ó5>
Ő
5sbb2/conv1/w/Initializer/random_uniform/RandomUniformRandomUniform-sbb2/conv1/w/Initializer/random_uniform/shape*
T0*
_class
loc:@sbb2/conv1/w*'
_output_shapes
:@*
dtype0
Î
+sbb2/conv1/w/Initializer/random_uniform/subSub+sbb2/conv1/w/Initializer/random_uniform/max+sbb2/conv1/w/Initializer/random_uniform/min*
T0*
_class
loc:@sbb2/conv1/w*
_output_shapes
: 
é
+sbb2/conv1/w/Initializer/random_uniform/mulMul5sbb2/conv1/w/Initializer/random_uniform/RandomUniform+sbb2/conv1/w/Initializer/random_uniform/sub*
T0*
_class
loc:@sbb2/conv1/w*'
_output_shapes
:@
Ý
'sbb2/conv1/w/Initializer/random_uniformAddV2+sbb2/conv1/w/Initializer/random_uniform/mul+sbb2/conv1/w/Initializer/random_uniform/min*
T0*
_class
loc:@sbb2/conv1/w*'
_output_shapes
:@

sbb2/conv1/wVarHandleOp*
_class
loc:@sbb2/conv1/w*
_output_shapes
: *
dtype0*
shape:@*
shared_namesbb2/conv1/w
i
-sbb2/conv1/w/IsInitialized/VarIsInitializedOpVarIsInitializedOpsbb2/conv1/w*
_output_shapes
: 

sbb2/conv1/w/AssignAssignVariableOpsbb2/conv1/w'sbb2/conv1/w/Initializer/random_uniform*&
 _has_manual_control_dependencies(*
dtype0
v
 sbb2/conv1/w/Read/ReadVariableOpReadVariableOpsbb2/conv1/w*'
_output_shapes
:@*
dtype0

sbb2/conv1/b/Initializer/ConstConst*
_class
loc:@sbb2/conv1/b*
_output_shapes
:@*
dtype0*
valueB@*    

sbb2/conv1/bVarHandleOp*
_class
loc:@sbb2/conv1/b*
_output_shapes
: *
dtype0*
shape:@*
shared_namesbb2/conv1/b
i
-sbb2/conv1/b/IsInitialized/VarIsInitializedOpVarIsInitializedOpsbb2/conv1/b*
_output_shapes
: 

sbb2/conv1/b/AssignAssignVariableOpsbb2/conv1/bsbb2/conv1/b/Initializer/Const*&
 _has_manual_control_dependencies(*
dtype0
i
 sbb2/conv1/b/Read/ReadVariableOpReadVariableOpsbb2/conv1/b*
_output_shapes
:@*
dtype0
v
 sbb2/conv1/Conv2D/ReadVariableOpReadVariableOpsbb2/conv1/w*'
_output_shapes
:@*
dtype0
Ş
sbb2/conv1/Conv2DConv2D	MaxPool_1 sbb2/conv1/Conv2D/ReadVariableOp*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙/@*
paddingSAME*
strides

j
!sbb2/conv1/BiasAdd/ReadVariableOpReadVariableOpsbb2/conv1/b*
_output_shapes
:@*
dtype0

sbb2/conv1/BiasAddBiasAddsbb2/conv1/Conv2D!sbb2/conv1/BiasAdd/ReadVariableOp*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙/@
_
	sbb2/ReluRelusbb2/conv1/BiasAdd*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙/@
§
-sbb2/conv2/w/Initializer/random_uniform/shapeConst*
_class
loc:@sbb2/conv2/w*
_output_shapes
:*
dtype0*%
valueB"      @   @   

+sbb2/conv2/w/Initializer/random_uniform/minConst*
_class
loc:@sbb2/conv2/w*
_output_shapes
: *
dtype0*
valueB
 *   ž

+sbb2/conv2/w/Initializer/random_uniform/maxConst*
_class
loc:@sbb2/conv2/w*
_output_shapes
: *
dtype0*
valueB
 *   >
Ô
5sbb2/conv2/w/Initializer/random_uniform/RandomUniformRandomUniform-sbb2/conv2/w/Initializer/random_uniform/shape*
T0*
_class
loc:@sbb2/conv2/w*&
_output_shapes
:@@*
dtype0
Î
+sbb2/conv2/w/Initializer/random_uniform/subSub+sbb2/conv2/w/Initializer/random_uniform/max+sbb2/conv2/w/Initializer/random_uniform/min*
T0*
_class
loc:@sbb2/conv2/w*
_output_shapes
: 
č
+sbb2/conv2/w/Initializer/random_uniform/mulMul5sbb2/conv2/w/Initializer/random_uniform/RandomUniform+sbb2/conv2/w/Initializer/random_uniform/sub*
T0*
_class
loc:@sbb2/conv2/w*&
_output_shapes
:@@
Ü
'sbb2/conv2/w/Initializer/random_uniformAddV2+sbb2/conv2/w/Initializer/random_uniform/mul+sbb2/conv2/w/Initializer/random_uniform/min*
T0*
_class
loc:@sbb2/conv2/w*&
_output_shapes
:@@

sbb2/conv2/wVarHandleOp*
_class
loc:@sbb2/conv2/w*
_output_shapes
: *
dtype0*
shape:@@*
shared_namesbb2/conv2/w
i
-sbb2/conv2/w/IsInitialized/VarIsInitializedOpVarIsInitializedOpsbb2/conv2/w*
_output_shapes
: 

sbb2/conv2/w/AssignAssignVariableOpsbb2/conv2/w'sbb2/conv2/w/Initializer/random_uniform*&
 _has_manual_control_dependencies(*
dtype0
u
 sbb2/conv2/w/Read/ReadVariableOpReadVariableOpsbb2/conv2/w*&
_output_shapes
:@@*
dtype0

sbb2/conv2/b/Initializer/ConstConst*
_class
loc:@sbb2/conv2/b*
_output_shapes
:@*
dtype0*
valueB@*    

sbb2/conv2/bVarHandleOp*
_class
loc:@sbb2/conv2/b*
_output_shapes
: *
dtype0*
shape:@*
shared_namesbb2/conv2/b
i
-sbb2/conv2/b/IsInitialized/VarIsInitializedOpVarIsInitializedOpsbb2/conv2/b*
_output_shapes
: 

sbb2/conv2/b/AssignAssignVariableOpsbb2/conv2/bsbb2/conv2/b/Initializer/Const*&
 _has_manual_control_dependencies(*
dtype0
i
 sbb2/conv2/b/Read/ReadVariableOpReadVariableOpsbb2/conv2/b*
_output_shapes
:@*
dtype0
u
 sbb2/conv2/Conv2D/ReadVariableOpReadVariableOpsbb2/conv2/w*&
_output_shapes
:@@*
dtype0
Ş
sbb2/conv2/Conv2DConv2D	sbb2/Relu sbb2/conv2/Conv2D/ReadVariableOp*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙/@*
paddingSAME*
strides

j
!sbb2/conv2/BiasAdd/ReadVariableOpReadVariableOpsbb2/conv2/b*
_output_shapes
:@*
dtype0

sbb2/conv2/BiasAddBiasAddsbb2/conv2/Conv2D!sbb2/conv2/BiasAdd/ReadVariableOp*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙/@
a
sbb2/Relu_1Relusbb2/conv2/BiasAdd*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙/@
§
-sbb2/conv3/w/Initializer/random_uniform/shapeConst*
_class
loc:@sbb2/conv3/w*
_output_shapes
:*
dtype0*%
valueB"      @   @   

+sbb2/conv3/w/Initializer/random_uniform/minConst*
_class
loc:@sbb2/conv3/w*
_output_shapes
: *
dtype0*
valueB
 *   ž

+sbb2/conv3/w/Initializer/random_uniform/maxConst*
_class
loc:@sbb2/conv3/w*
_output_shapes
: *
dtype0*
valueB
 *   >
Ô
5sbb2/conv3/w/Initializer/random_uniform/RandomUniformRandomUniform-sbb2/conv3/w/Initializer/random_uniform/shape*
T0*
_class
loc:@sbb2/conv3/w*&
_output_shapes
:@@*
dtype0
Î
+sbb2/conv3/w/Initializer/random_uniform/subSub+sbb2/conv3/w/Initializer/random_uniform/max+sbb2/conv3/w/Initializer/random_uniform/min*
T0*
_class
loc:@sbb2/conv3/w*
_output_shapes
: 
č
+sbb2/conv3/w/Initializer/random_uniform/mulMul5sbb2/conv3/w/Initializer/random_uniform/RandomUniform+sbb2/conv3/w/Initializer/random_uniform/sub*
T0*
_class
loc:@sbb2/conv3/w*&
_output_shapes
:@@
Ü
'sbb2/conv3/w/Initializer/random_uniformAddV2+sbb2/conv3/w/Initializer/random_uniform/mul+sbb2/conv3/w/Initializer/random_uniform/min*
T0*
_class
loc:@sbb2/conv3/w*&
_output_shapes
:@@

sbb2/conv3/wVarHandleOp*
_class
loc:@sbb2/conv3/w*
_output_shapes
: *
dtype0*
shape:@@*
shared_namesbb2/conv3/w
i
-sbb2/conv3/w/IsInitialized/VarIsInitializedOpVarIsInitializedOpsbb2/conv3/w*
_output_shapes
: 

sbb2/conv3/w/AssignAssignVariableOpsbb2/conv3/w'sbb2/conv3/w/Initializer/random_uniform*&
 _has_manual_control_dependencies(*
dtype0
u
 sbb2/conv3/w/Read/ReadVariableOpReadVariableOpsbb2/conv3/w*&
_output_shapes
:@@*
dtype0

sbb2/conv3/b/Initializer/ConstConst*
_class
loc:@sbb2/conv3/b*
_output_shapes
:@*
dtype0*
valueB@*    

sbb2/conv3/bVarHandleOp*
_class
loc:@sbb2/conv3/b*
_output_shapes
: *
dtype0*
shape:@*
shared_namesbb2/conv3/b
i
-sbb2/conv3/b/IsInitialized/VarIsInitializedOpVarIsInitializedOpsbb2/conv3/b*
_output_shapes
: 

sbb2/conv3/b/AssignAssignVariableOpsbb2/conv3/bsbb2/conv3/b/Initializer/Const*&
 _has_manual_control_dependencies(*
dtype0
i
 sbb2/conv3/b/Read/ReadVariableOpReadVariableOpsbb2/conv3/b*
_output_shapes
:@*
dtype0
u
 sbb2/conv3/Conv2D/ReadVariableOpReadVariableOpsbb2/conv3/w*&
_output_shapes
:@@*
dtype0
Ź
sbb2/conv3/Conv2DConv2Dsbb2/Relu_1 sbb2/conv3/Conv2D/ReadVariableOp*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙/@*
paddingSAME*
strides

j
!sbb2/conv3/BiasAdd/ReadVariableOpReadVariableOpsbb2/conv3/b*
_output_shapes
:@*
dtype0

sbb2/conv3/BiasAddBiasAddsbb2/conv3/Conv2D!sbb2/conv3/BiasAdd/ReadVariableOp*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙/@
a
sbb2/Relu_2Relusbb2/conv3/BiasAdd*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙/@
§
-sbb2/conv4/w/Initializer/random_uniform/shapeConst*
_class
loc:@sbb2/conv4/w*
_output_shapes
:*
dtype0*%
valueB"      @      

+sbb2/conv4/w/Initializer/random_uniform/minConst*
_class
loc:@sbb2/conv4/w*
_output_shapes
: *
dtype0*
valueB
 *7ž

+sbb2/conv4/w/Initializer/random_uniform/maxConst*
_class
loc:@sbb2/conv4/w*
_output_shapes
: *
dtype0*
valueB
 *7>
Ő
5sbb2/conv4/w/Initializer/random_uniform/RandomUniformRandomUniform-sbb2/conv4/w/Initializer/random_uniform/shape*
T0*
_class
loc:@sbb2/conv4/w*'
_output_shapes
:@*
dtype0
Î
+sbb2/conv4/w/Initializer/random_uniform/subSub+sbb2/conv4/w/Initializer/random_uniform/max+sbb2/conv4/w/Initializer/random_uniform/min*
T0*
_class
loc:@sbb2/conv4/w*
_output_shapes
: 
é
+sbb2/conv4/w/Initializer/random_uniform/mulMul5sbb2/conv4/w/Initializer/random_uniform/RandomUniform+sbb2/conv4/w/Initializer/random_uniform/sub*
T0*
_class
loc:@sbb2/conv4/w*'
_output_shapes
:@
Ý
'sbb2/conv4/w/Initializer/random_uniformAddV2+sbb2/conv4/w/Initializer/random_uniform/mul+sbb2/conv4/w/Initializer/random_uniform/min*
T0*
_class
loc:@sbb2/conv4/w*'
_output_shapes
:@

sbb2/conv4/wVarHandleOp*
_class
loc:@sbb2/conv4/w*
_output_shapes
: *
dtype0*
shape:@*
shared_namesbb2/conv4/w
i
-sbb2/conv4/w/IsInitialized/VarIsInitializedOpVarIsInitializedOpsbb2/conv4/w*
_output_shapes
: 

sbb2/conv4/w/AssignAssignVariableOpsbb2/conv4/w'sbb2/conv4/w/Initializer/random_uniform*&
 _has_manual_control_dependencies(*
dtype0
v
 sbb2/conv4/w/Read/ReadVariableOpReadVariableOpsbb2/conv4/w*'
_output_shapes
:@*
dtype0

sbb2/conv4/b/Initializer/ConstConst*
_class
loc:@sbb2/conv4/b*
_output_shapes	
:*
dtype0*
valueB*    

sbb2/conv4/bVarHandleOp*
_class
loc:@sbb2/conv4/b*
_output_shapes
: *
dtype0*
shape:*
shared_namesbb2/conv4/b
i
-sbb2/conv4/b/IsInitialized/VarIsInitializedOpVarIsInitializedOpsbb2/conv4/b*
_output_shapes
: 

sbb2/conv4/b/AssignAssignVariableOpsbb2/conv4/bsbb2/conv4/b/Initializer/Const*&
 _has_manual_control_dependencies(*
dtype0
j
 sbb2/conv4/b/Read/ReadVariableOpReadVariableOpsbb2/conv4/b*
_output_shapes	
:*
dtype0
v
 sbb2/conv4/Conv2D/ReadVariableOpReadVariableOpsbb2/conv4/w*'
_output_shapes
:@*
dtype0
­
sbb2/conv4/Conv2DConv2Dsbb2/Relu_2 sbb2/conv4/Conv2D/ReadVariableOp*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙/*
paddingSAME*
strides

k
!sbb2/conv4/BiasAdd/ReadVariableOpReadVariableOpsbb2/conv4/b*
_output_shapes	
:*
dtype0

sbb2/conv4/BiasAddBiasAddsbb2/conv4/Conv2D!sbb2/conv4/BiasAdd/ReadVariableOp*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙/
ą
/sbb2/batch_normalization/gamma/Initializer/onesConst*1
_class'
%#loc:@sbb2/batch_normalization/gamma*
_output_shapes	
:*
dtype0*
valueB*  ?
Č
sbb2/batch_normalization/gammaVarHandleOp*1
_class'
%#loc:@sbb2/batch_normalization/gamma*
_output_shapes
: *
dtype0*
shape:*/
shared_name sbb2/batch_normalization/gamma

?sbb2/batch_normalization/gamma/IsInitialized/VarIsInitializedOpVarIsInitializedOpsbb2/batch_normalization/gamma*
_output_shapes
: 
ż
%sbb2/batch_normalization/gamma/AssignAssignVariableOpsbb2/batch_normalization/gamma/sbb2/batch_normalization/gamma/Initializer/ones*&
 _has_manual_control_dependencies(*
dtype0

2sbb2/batch_normalization/gamma/Read/ReadVariableOpReadVariableOpsbb2/batch_normalization/gamma*
_output_shapes	
:*
dtype0
°
/sbb2/batch_normalization/beta/Initializer/zerosConst*0
_class&
$"loc:@sbb2/batch_normalization/beta*
_output_shapes	
:*
dtype0*
valueB*    
Ĺ
sbb2/batch_normalization/betaVarHandleOp*0
_class&
$"loc:@sbb2/batch_normalization/beta*
_output_shapes
: *
dtype0*
shape:*.
shared_namesbb2/batch_normalization/beta

>sbb2/batch_normalization/beta/IsInitialized/VarIsInitializedOpVarIsInitializedOpsbb2/batch_normalization/beta*
_output_shapes
: 
˝
$sbb2/batch_normalization/beta/AssignAssignVariableOpsbb2/batch_normalization/beta/sbb2/batch_normalization/beta/Initializer/zeros*&
 _has_manual_control_dependencies(*
dtype0

1sbb2/batch_normalization/beta/Read/ReadVariableOpReadVariableOpsbb2/batch_normalization/beta*
_output_shapes	
:*
dtype0
ž
6sbb2/batch_normalization/moving_mean/Initializer/zerosConst*7
_class-
+)loc:@sbb2/batch_normalization/moving_mean*
_output_shapes	
:*
dtype0*
valueB*    
Ú
$sbb2/batch_normalization/moving_meanVarHandleOp*7
_class-
+)loc:@sbb2/batch_normalization/moving_mean*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$sbb2/batch_normalization/moving_mean

Esbb2/batch_normalization/moving_mean/IsInitialized/VarIsInitializedOpVarIsInitializedOp$sbb2/batch_normalization/moving_mean*
_output_shapes
: 
Ň
+sbb2/batch_normalization/moving_mean/AssignAssignVariableOp$sbb2/batch_normalization/moving_mean6sbb2/batch_normalization/moving_mean/Initializer/zeros*&
 _has_manual_control_dependencies(*
dtype0

8sbb2/batch_normalization/moving_mean/Read/ReadVariableOpReadVariableOp$sbb2/batch_normalization/moving_mean*
_output_shapes	
:*
dtype0
Ĺ
9sbb2/batch_normalization/moving_variance/Initializer/onesConst*;
_class1
/-loc:@sbb2/batch_normalization/moving_variance*
_output_shapes	
:*
dtype0*
valueB*  ?
ć
(sbb2/batch_normalization/moving_varianceVarHandleOp*;
_class1
/-loc:@sbb2/batch_normalization/moving_variance*
_output_shapes
: *
dtype0*
shape:*9
shared_name*(sbb2/batch_normalization/moving_variance
Ą
Isbb2/batch_normalization/moving_variance/IsInitialized/VarIsInitializedOpVarIsInitializedOp(sbb2/batch_normalization/moving_variance*
_output_shapes
: 
Ý
/sbb2/batch_normalization/moving_variance/AssignAssignVariableOp(sbb2/batch_normalization/moving_variance9sbb2/batch_normalization/moving_variance/Initializer/ones*&
 _has_manual_control_dependencies(*
dtype0
˘
<sbb2/batch_normalization/moving_variance/Read/ReadVariableOpReadVariableOp(sbb2/batch_normalization/moving_variance*
_output_shapes	
:*
dtype0

'sbb2/batch_normalization/ReadVariableOpReadVariableOpsbb2/batch_normalization/gamma*
_output_shapes	
:*
dtype0

)sbb2/batch_normalization/ReadVariableOp_1ReadVariableOpsbb2/batch_normalization/beta*
_output_shapes	
:*
dtype0

8sbb2/batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp$sbb2/batch_normalization/moving_mean*
_output_shapes	
:*
dtype0
 
:sbb2/batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp(sbb2/batch_normalization/moving_variance*
_output_shapes	
:*
dtype0
¤
)sbb2/batch_normalization/FusedBatchNormV3FusedBatchNormV3sbb2/conv4/BiasAdd'sbb2/batch_normalization/ReadVariableOp)sbb2/batch_normalization/ReadVariableOp_18sbb2/batch_normalization/FusedBatchNormV3/ReadVariableOp:sbb2/batch_normalization/FusedBatchNormV3/ReadVariableOp_1*
T0*
U0*P
_output_shapes>
<:˙˙˙˙˙˙˙˙˙/:::::*
epsilon%o:*
is_training( 
y
sbb2/Relu_3Relu)sbb2/batch_normalization/FusedBatchNormV3*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙/
§
-sbb3/conv1/w/Initializer/random_uniform/shapeConst*
_class
loc:@sbb3/conv1/w*
_output_shapes
:*
dtype0*%
valueB"         @   

+sbb3/conv1/w/Initializer/random_uniform/minConst*
_class
loc:@sbb3/conv1/w*
_output_shapes
: *
dtype0*
valueB
 *7ž

+sbb3/conv1/w/Initializer/random_uniform/maxConst*
_class
loc:@sbb3/conv1/w*
_output_shapes
: *
dtype0*
valueB
 *7>
Ő
5sbb3/conv1/w/Initializer/random_uniform/RandomUniformRandomUniform-sbb3/conv1/w/Initializer/random_uniform/shape*
T0*
_class
loc:@sbb3/conv1/w*'
_output_shapes
:@*
dtype0
Î
+sbb3/conv1/w/Initializer/random_uniform/subSub+sbb3/conv1/w/Initializer/random_uniform/max+sbb3/conv1/w/Initializer/random_uniform/min*
T0*
_class
loc:@sbb3/conv1/w*
_output_shapes
: 
é
+sbb3/conv1/w/Initializer/random_uniform/mulMul5sbb3/conv1/w/Initializer/random_uniform/RandomUniform+sbb3/conv1/w/Initializer/random_uniform/sub*
T0*
_class
loc:@sbb3/conv1/w*'
_output_shapes
:@
Ý
'sbb3/conv1/w/Initializer/random_uniformAddV2+sbb3/conv1/w/Initializer/random_uniform/mul+sbb3/conv1/w/Initializer/random_uniform/min*
T0*
_class
loc:@sbb3/conv1/w*'
_output_shapes
:@

sbb3/conv1/wVarHandleOp*
_class
loc:@sbb3/conv1/w*
_output_shapes
: *
dtype0*
shape:@*
shared_namesbb3/conv1/w
i
-sbb3/conv1/w/IsInitialized/VarIsInitializedOpVarIsInitializedOpsbb3/conv1/w*
_output_shapes
: 

sbb3/conv1/w/AssignAssignVariableOpsbb3/conv1/w'sbb3/conv1/w/Initializer/random_uniform*&
 _has_manual_control_dependencies(*
dtype0
v
 sbb3/conv1/w/Read/ReadVariableOpReadVariableOpsbb3/conv1/w*'
_output_shapes
:@*
dtype0

sbb3/conv1/b/Initializer/ConstConst*
_class
loc:@sbb3/conv1/b*
_output_shapes
:@*
dtype0*
valueB@*    

sbb3/conv1/bVarHandleOp*
_class
loc:@sbb3/conv1/b*
_output_shapes
: *
dtype0*
shape:@*
shared_namesbb3/conv1/b
i
-sbb3/conv1/b/IsInitialized/VarIsInitializedOpVarIsInitializedOpsbb3/conv1/b*
_output_shapes
: 

sbb3/conv1/b/AssignAssignVariableOpsbb3/conv1/bsbb3/conv1/b/Initializer/Const*&
 _has_manual_control_dependencies(*
dtype0
i
 sbb3/conv1/b/Read/ReadVariableOpReadVariableOpsbb3/conv1/b*
_output_shapes
:@*
dtype0
v
 sbb3/conv1/Conv2D/ReadVariableOpReadVariableOpsbb3/conv1/w*'
_output_shapes
:@*
dtype0
Ź
sbb3/conv1/Conv2DConv2Dsbb2/Relu_3 sbb3/conv1/Conv2D/ReadVariableOp*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙/@*
paddingSAME*
strides

j
!sbb3/conv1/BiasAdd/ReadVariableOpReadVariableOpsbb3/conv1/b*
_output_shapes
:@*
dtype0

sbb3/conv1/BiasAddBiasAddsbb3/conv1/Conv2D!sbb3/conv1/BiasAdd/ReadVariableOp*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙/@
_
	sbb3/ReluRelusbb3/conv1/BiasAdd*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙/@
§
-sbb3/conv2/w/Initializer/random_uniform/shapeConst*
_class
loc:@sbb3/conv2/w*
_output_shapes
:*
dtype0*%
valueB"      @   @   

+sbb3/conv2/w/Initializer/random_uniform/minConst*
_class
loc:@sbb3/conv2/w*
_output_shapes
: *
dtype0*
valueB
 *   ž

+sbb3/conv2/w/Initializer/random_uniform/maxConst*
_class
loc:@sbb3/conv2/w*
_output_shapes
: *
dtype0*
valueB
 *   >
Ô
5sbb3/conv2/w/Initializer/random_uniform/RandomUniformRandomUniform-sbb3/conv2/w/Initializer/random_uniform/shape*
T0*
_class
loc:@sbb3/conv2/w*&
_output_shapes
:@@*
dtype0
Î
+sbb3/conv2/w/Initializer/random_uniform/subSub+sbb3/conv2/w/Initializer/random_uniform/max+sbb3/conv2/w/Initializer/random_uniform/min*
T0*
_class
loc:@sbb3/conv2/w*
_output_shapes
: 
č
+sbb3/conv2/w/Initializer/random_uniform/mulMul5sbb3/conv2/w/Initializer/random_uniform/RandomUniform+sbb3/conv2/w/Initializer/random_uniform/sub*
T0*
_class
loc:@sbb3/conv2/w*&
_output_shapes
:@@
Ü
'sbb3/conv2/w/Initializer/random_uniformAddV2+sbb3/conv2/w/Initializer/random_uniform/mul+sbb3/conv2/w/Initializer/random_uniform/min*
T0*
_class
loc:@sbb3/conv2/w*&
_output_shapes
:@@

sbb3/conv2/wVarHandleOp*
_class
loc:@sbb3/conv2/w*
_output_shapes
: *
dtype0*
shape:@@*
shared_namesbb3/conv2/w
i
-sbb3/conv2/w/IsInitialized/VarIsInitializedOpVarIsInitializedOpsbb3/conv2/w*
_output_shapes
: 

sbb3/conv2/w/AssignAssignVariableOpsbb3/conv2/w'sbb3/conv2/w/Initializer/random_uniform*&
 _has_manual_control_dependencies(*
dtype0
u
 sbb3/conv2/w/Read/ReadVariableOpReadVariableOpsbb3/conv2/w*&
_output_shapes
:@@*
dtype0

sbb3/conv2/b/Initializer/ConstConst*
_class
loc:@sbb3/conv2/b*
_output_shapes
:@*
dtype0*
valueB@*    

sbb3/conv2/bVarHandleOp*
_class
loc:@sbb3/conv2/b*
_output_shapes
: *
dtype0*
shape:@*
shared_namesbb3/conv2/b
i
-sbb3/conv2/b/IsInitialized/VarIsInitializedOpVarIsInitializedOpsbb3/conv2/b*
_output_shapes
: 

sbb3/conv2/b/AssignAssignVariableOpsbb3/conv2/bsbb3/conv2/b/Initializer/Const*&
 _has_manual_control_dependencies(*
dtype0
i
 sbb3/conv2/b/Read/ReadVariableOpReadVariableOpsbb3/conv2/b*
_output_shapes
:@*
dtype0
u
 sbb3/conv2/Conv2D/ReadVariableOpReadVariableOpsbb3/conv2/w*&
_output_shapes
:@@*
dtype0
Ş
sbb3/conv2/Conv2DConv2D	sbb3/Relu sbb3/conv2/Conv2D/ReadVariableOp*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙/@*
paddingSAME*
strides

j
!sbb3/conv2/BiasAdd/ReadVariableOpReadVariableOpsbb3/conv2/b*
_output_shapes
:@*
dtype0

sbb3/conv2/BiasAddBiasAddsbb3/conv2/Conv2D!sbb3/conv2/BiasAdd/ReadVariableOp*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙/@
a
sbb3/Relu_1Relusbb3/conv2/BiasAdd*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙/@
§
-sbb3/conv3/w/Initializer/random_uniform/shapeConst*
_class
loc:@sbb3/conv3/w*
_output_shapes
:*
dtype0*%
valueB"      @   @   

+sbb3/conv3/w/Initializer/random_uniform/minConst*
_class
loc:@sbb3/conv3/w*
_output_shapes
: *
dtype0*
valueB
 *   ž

+sbb3/conv3/w/Initializer/random_uniform/maxConst*
_class
loc:@sbb3/conv3/w*
_output_shapes
: *
dtype0*
valueB
 *   >
Ô
5sbb3/conv3/w/Initializer/random_uniform/RandomUniformRandomUniform-sbb3/conv3/w/Initializer/random_uniform/shape*
T0*
_class
loc:@sbb3/conv3/w*&
_output_shapes
:@@*
dtype0
Î
+sbb3/conv3/w/Initializer/random_uniform/subSub+sbb3/conv3/w/Initializer/random_uniform/max+sbb3/conv3/w/Initializer/random_uniform/min*
T0*
_class
loc:@sbb3/conv3/w*
_output_shapes
: 
č
+sbb3/conv3/w/Initializer/random_uniform/mulMul5sbb3/conv3/w/Initializer/random_uniform/RandomUniform+sbb3/conv3/w/Initializer/random_uniform/sub*
T0*
_class
loc:@sbb3/conv3/w*&
_output_shapes
:@@
Ü
'sbb3/conv3/w/Initializer/random_uniformAddV2+sbb3/conv3/w/Initializer/random_uniform/mul+sbb3/conv3/w/Initializer/random_uniform/min*
T0*
_class
loc:@sbb3/conv3/w*&
_output_shapes
:@@

sbb3/conv3/wVarHandleOp*
_class
loc:@sbb3/conv3/w*
_output_shapes
: *
dtype0*
shape:@@*
shared_namesbb3/conv3/w
i
-sbb3/conv3/w/IsInitialized/VarIsInitializedOpVarIsInitializedOpsbb3/conv3/w*
_output_shapes
: 

sbb3/conv3/w/AssignAssignVariableOpsbb3/conv3/w'sbb3/conv3/w/Initializer/random_uniform*&
 _has_manual_control_dependencies(*
dtype0
u
 sbb3/conv3/w/Read/ReadVariableOpReadVariableOpsbb3/conv3/w*&
_output_shapes
:@@*
dtype0

sbb3/conv3/b/Initializer/ConstConst*
_class
loc:@sbb3/conv3/b*
_output_shapes
:@*
dtype0*
valueB@*    

sbb3/conv3/bVarHandleOp*
_class
loc:@sbb3/conv3/b*
_output_shapes
: *
dtype0*
shape:@*
shared_namesbb3/conv3/b
i
-sbb3/conv3/b/IsInitialized/VarIsInitializedOpVarIsInitializedOpsbb3/conv3/b*
_output_shapes
: 

sbb3/conv3/b/AssignAssignVariableOpsbb3/conv3/bsbb3/conv3/b/Initializer/Const*&
 _has_manual_control_dependencies(*
dtype0
i
 sbb3/conv3/b/Read/ReadVariableOpReadVariableOpsbb3/conv3/b*
_output_shapes
:@*
dtype0
u
 sbb3/conv3/Conv2D/ReadVariableOpReadVariableOpsbb3/conv3/w*&
_output_shapes
:@@*
dtype0
Ź
sbb3/conv3/Conv2DConv2Dsbb3/Relu_1 sbb3/conv3/Conv2D/ReadVariableOp*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙/@*
paddingSAME*
strides

j
!sbb3/conv3/BiasAdd/ReadVariableOpReadVariableOpsbb3/conv3/b*
_output_shapes
:@*
dtype0

sbb3/conv3/BiasAddBiasAddsbb3/conv3/Conv2D!sbb3/conv3/BiasAdd/ReadVariableOp*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙/@
a
sbb3/Relu_2Relusbb3/conv3/BiasAdd*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙/@
§
-sbb3/conv4/w/Initializer/random_uniform/shapeConst*
_class
loc:@sbb3/conv4/w*
_output_shapes
:*
dtype0*%
valueB"      @      

+sbb3/conv4/w/Initializer/random_uniform/minConst*
_class
loc:@sbb3/conv4/w*
_output_shapes
: *
dtype0*
valueB
 *7ž

+sbb3/conv4/w/Initializer/random_uniform/maxConst*
_class
loc:@sbb3/conv4/w*
_output_shapes
: *
dtype0*
valueB
 *7>
Ő
5sbb3/conv4/w/Initializer/random_uniform/RandomUniformRandomUniform-sbb3/conv4/w/Initializer/random_uniform/shape*
T0*
_class
loc:@sbb3/conv4/w*'
_output_shapes
:@*
dtype0
Î
+sbb3/conv4/w/Initializer/random_uniform/subSub+sbb3/conv4/w/Initializer/random_uniform/max+sbb3/conv4/w/Initializer/random_uniform/min*
T0*
_class
loc:@sbb3/conv4/w*
_output_shapes
: 
é
+sbb3/conv4/w/Initializer/random_uniform/mulMul5sbb3/conv4/w/Initializer/random_uniform/RandomUniform+sbb3/conv4/w/Initializer/random_uniform/sub*
T0*
_class
loc:@sbb3/conv4/w*'
_output_shapes
:@
Ý
'sbb3/conv4/w/Initializer/random_uniformAddV2+sbb3/conv4/w/Initializer/random_uniform/mul+sbb3/conv4/w/Initializer/random_uniform/min*
T0*
_class
loc:@sbb3/conv4/w*'
_output_shapes
:@

sbb3/conv4/wVarHandleOp*
_class
loc:@sbb3/conv4/w*
_output_shapes
: *
dtype0*
shape:@*
shared_namesbb3/conv4/w
i
-sbb3/conv4/w/IsInitialized/VarIsInitializedOpVarIsInitializedOpsbb3/conv4/w*
_output_shapes
: 

sbb3/conv4/w/AssignAssignVariableOpsbb3/conv4/w'sbb3/conv4/w/Initializer/random_uniform*&
 _has_manual_control_dependencies(*
dtype0
v
 sbb3/conv4/w/Read/ReadVariableOpReadVariableOpsbb3/conv4/w*'
_output_shapes
:@*
dtype0

sbb3/conv4/b/Initializer/ConstConst*
_class
loc:@sbb3/conv4/b*
_output_shapes	
:*
dtype0*
valueB*    

sbb3/conv4/bVarHandleOp*
_class
loc:@sbb3/conv4/b*
_output_shapes
: *
dtype0*
shape:*
shared_namesbb3/conv4/b
i
-sbb3/conv4/b/IsInitialized/VarIsInitializedOpVarIsInitializedOpsbb3/conv4/b*
_output_shapes
: 

sbb3/conv4/b/AssignAssignVariableOpsbb3/conv4/bsbb3/conv4/b/Initializer/Const*&
 _has_manual_control_dependencies(*
dtype0
j
 sbb3/conv4/b/Read/ReadVariableOpReadVariableOpsbb3/conv4/b*
_output_shapes	
:*
dtype0
v
 sbb3/conv4/Conv2D/ReadVariableOpReadVariableOpsbb3/conv4/w*'
_output_shapes
:@*
dtype0
­
sbb3/conv4/Conv2DConv2Dsbb3/Relu_2 sbb3/conv4/Conv2D/ReadVariableOp*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙/*
paddingSAME*
strides

k
!sbb3/conv4/BiasAdd/ReadVariableOpReadVariableOpsbb3/conv4/b*
_output_shapes	
:*
dtype0

sbb3/conv4/BiasAddBiasAddsbb3/conv4/Conv2D!sbb3/conv4/BiasAdd/ReadVariableOp*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙/
ą
/sbb3/batch_normalization/gamma/Initializer/onesConst*1
_class'
%#loc:@sbb3/batch_normalization/gamma*
_output_shapes	
:*
dtype0*
valueB*  ?
Č
sbb3/batch_normalization/gammaVarHandleOp*1
_class'
%#loc:@sbb3/batch_normalization/gamma*
_output_shapes
: *
dtype0*
shape:*/
shared_name sbb3/batch_normalization/gamma

?sbb3/batch_normalization/gamma/IsInitialized/VarIsInitializedOpVarIsInitializedOpsbb3/batch_normalization/gamma*
_output_shapes
: 
ż
%sbb3/batch_normalization/gamma/AssignAssignVariableOpsbb3/batch_normalization/gamma/sbb3/batch_normalization/gamma/Initializer/ones*&
 _has_manual_control_dependencies(*
dtype0

2sbb3/batch_normalization/gamma/Read/ReadVariableOpReadVariableOpsbb3/batch_normalization/gamma*
_output_shapes	
:*
dtype0
°
/sbb3/batch_normalization/beta/Initializer/zerosConst*0
_class&
$"loc:@sbb3/batch_normalization/beta*
_output_shapes	
:*
dtype0*
valueB*    
Ĺ
sbb3/batch_normalization/betaVarHandleOp*0
_class&
$"loc:@sbb3/batch_normalization/beta*
_output_shapes
: *
dtype0*
shape:*.
shared_namesbb3/batch_normalization/beta

>sbb3/batch_normalization/beta/IsInitialized/VarIsInitializedOpVarIsInitializedOpsbb3/batch_normalization/beta*
_output_shapes
: 
˝
$sbb3/batch_normalization/beta/AssignAssignVariableOpsbb3/batch_normalization/beta/sbb3/batch_normalization/beta/Initializer/zeros*&
 _has_manual_control_dependencies(*
dtype0

1sbb3/batch_normalization/beta/Read/ReadVariableOpReadVariableOpsbb3/batch_normalization/beta*
_output_shapes	
:*
dtype0
ž
6sbb3/batch_normalization/moving_mean/Initializer/zerosConst*7
_class-
+)loc:@sbb3/batch_normalization/moving_mean*
_output_shapes	
:*
dtype0*
valueB*    
Ú
$sbb3/batch_normalization/moving_meanVarHandleOp*7
_class-
+)loc:@sbb3/batch_normalization/moving_mean*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$sbb3/batch_normalization/moving_mean

Esbb3/batch_normalization/moving_mean/IsInitialized/VarIsInitializedOpVarIsInitializedOp$sbb3/batch_normalization/moving_mean*
_output_shapes
: 
Ň
+sbb3/batch_normalization/moving_mean/AssignAssignVariableOp$sbb3/batch_normalization/moving_mean6sbb3/batch_normalization/moving_mean/Initializer/zeros*&
 _has_manual_control_dependencies(*
dtype0

8sbb3/batch_normalization/moving_mean/Read/ReadVariableOpReadVariableOp$sbb3/batch_normalization/moving_mean*
_output_shapes	
:*
dtype0
Ĺ
9sbb3/batch_normalization/moving_variance/Initializer/onesConst*;
_class1
/-loc:@sbb3/batch_normalization/moving_variance*
_output_shapes	
:*
dtype0*
valueB*  ?
ć
(sbb3/batch_normalization/moving_varianceVarHandleOp*;
_class1
/-loc:@sbb3/batch_normalization/moving_variance*
_output_shapes
: *
dtype0*
shape:*9
shared_name*(sbb3/batch_normalization/moving_variance
Ą
Isbb3/batch_normalization/moving_variance/IsInitialized/VarIsInitializedOpVarIsInitializedOp(sbb3/batch_normalization/moving_variance*
_output_shapes
: 
Ý
/sbb3/batch_normalization/moving_variance/AssignAssignVariableOp(sbb3/batch_normalization/moving_variance9sbb3/batch_normalization/moving_variance/Initializer/ones*&
 _has_manual_control_dependencies(*
dtype0
˘
<sbb3/batch_normalization/moving_variance/Read/ReadVariableOpReadVariableOp(sbb3/batch_normalization/moving_variance*
_output_shapes	
:*
dtype0

'sbb3/batch_normalization/ReadVariableOpReadVariableOpsbb3/batch_normalization/gamma*
_output_shapes	
:*
dtype0

)sbb3/batch_normalization/ReadVariableOp_1ReadVariableOpsbb3/batch_normalization/beta*
_output_shapes	
:*
dtype0

8sbb3/batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp$sbb3/batch_normalization/moving_mean*
_output_shapes	
:*
dtype0
 
:sbb3/batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp(sbb3/batch_normalization/moving_variance*
_output_shapes	
:*
dtype0
¤
)sbb3/batch_normalization/FusedBatchNormV3FusedBatchNormV3sbb3/conv4/BiasAdd'sbb3/batch_normalization/ReadVariableOp)sbb3/batch_normalization/ReadVariableOp_18sbb3/batch_normalization/FusedBatchNormV3/ReadVariableOp:sbb3/batch_normalization/FusedBatchNormV3/ReadVariableOp_1*
T0*
U0*P
_output_shapes>
<:˙˙˙˙˙˙˙˙˙/:::::*
epsilon%o:*
is_training( 
y
sbb3/Relu_3Relu)sbb3/batch_normalization/FusedBatchNormV3*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙/

	MaxPool_2MaxPoolsbb3/Relu_3*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
ksize
*
paddingSAME*
strides

b
dropout/IdentityIdentity	MaxPool_2*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ą
*conv_d1/w/Initializer/random_uniform/shapeConst*
_class
loc:@conv_d1/w*
_output_shapes
:*
dtype0*%
valueB"            

(conv_d1/w/Initializer/random_uniform/minConst*
_class
loc:@conv_d1/w*
_output_shapes
: *
dtype0*
valueB
 *×ł]˝

(conv_d1/w/Initializer/random_uniform/maxConst*
_class
loc:@conv_d1/w*
_output_shapes
: *
dtype0*
valueB
 *×ł]=
Í
2conv_d1/w/Initializer/random_uniform/RandomUniformRandomUniform*conv_d1/w/Initializer/random_uniform/shape*
T0*
_class
loc:@conv_d1/w*(
_output_shapes
:*
dtype0
Â
(conv_d1/w/Initializer/random_uniform/subSub(conv_d1/w/Initializer/random_uniform/max(conv_d1/w/Initializer/random_uniform/min*
T0*
_class
loc:@conv_d1/w*
_output_shapes
: 
Ţ
(conv_d1/w/Initializer/random_uniform/mulMul2conv_d1/w/Initializer/random_uniform/RandomUniform(conv_d1/w/Initializer/random_uniform/sub*
T0*
_class
loc:@conv_d1/w*(
_output_shapes
:
Ň
$conv_d1/w/Initializer/random_uniformAddV2(conv_d1/w/Initializer/random_uniform/mul(conv_d1/w/Initializer/random_uniform/min*
T0*
_class
loc:@conv_d1/w*(
_output_shapes
:

	conv_d1/wVarHandleOp*
_class
loc:@conv_d1/w*
_output_shapes
: *
dtype0*
shape:*
shared_name	conv_d1/w
c
*conv_d1/w/IsInitialized/VarIsInitializedOpVarIsInitializedOp	conv_d1/w*
_output_shapes
: 

conv_d1/w/AssignAssignVariableOp	conv_d1/w$conv_d1/w/Initializer/random_uniform*&
 _has_manual_control_dependencies(*
dtype0
q
conv_d1/w/Read/ReadVariableOpReadVariableOp	conv_d1/w*(
_output_shapes
:*
dtype0

conv_d1/b/Initializer/ConstConst*
_class
loc:@conv_d1/b*
_output_shapes	
:*
dtype0*
valueB*    

	conv_d1/bVarHandleOp*
_class
loc:@conv_d1/b*
_output_shapes
: *
dtype0*
shape:*
shared_name	conv_d1/b
c
*conv_d1/b/IsInitialized/VarIsInitializedOpVarIsInitializedOp	conv_d1/b*
_output_shapes
: 

conv_d1/b/AssignAssignVariableOp	conv_d1/bconv_d1/b/Initializer/Const*&
 _has_manual_control_dependencies(*
dtype0
d
conv_d1/b/Read/ReadVariableOpReadVariableOp	conv_d1/b*
_output_shapes	
:*
dtype0
q
conv_d1/Conv2D/ReadVariableOpReadVariableOp	conv_d1/w*(
_output_shapes
:*
dtype0
Ź
conv_d1/Conv2DConv2Ddropout/Identityconv_d1/Conv2D/ReadVariableOp*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
paddingSAME*
strides

e
conv_d1/BiasAdd/ReadVariableOpReadVariableOp	conv_d1/b*
_output_shapes	
:*
dtype0

conv_d1/BiasAddBiasAddconv_d1/Conv2Dconv_d1/BiasAdd/ReadVariableOp*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ť
,batch_normalization_1/gamma/Initializer/onesConst*.
_class$
" loc:@batch_normalization_1/gamma*
_output_shapes	
:*
dtype0*
valueB*  ?
ż
batch_normalization_1/gammaVarHandleOp*.
_class$
" loc:@batch_normalization_1/gamma*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_1/gamma

<batch_normalization_1/gamma/IsInitialized/VarIsInitializedOpVarIsInitializedOpbatch_normalization_1/gamma*
_output_shapes
: 
ś
"batch_normalization_1/gamma/AssignAssignVariableOpbatch_normalization_1/gamma,batch_normalization_1/gamma/Initializer/ones*&
 _has_manual_control_dependencies(*
dtype0

/batch_normalization_1/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_1/gamma*
_output_shapes	
:*
dtype0
Ş
,batch_normalization_1/beta/Initializer/zerosConst*-
_class#
!loc:@batch_normalization_1/beta*
_output_shapes	
:*
dtype0*
valueB*    
ź
batch_normalization_1/betaVarHandleOp*-
_class#
!loc:@batch_normalization_1/beta*
_output_shapes
: *
dtype0*
shape:*+
shared_namebatch_normalization_1/beta

;batch_normalization_1/beta/IsInitialized/VarIsInitializedOpVarIsInitializedOpbatch_normalization_1/beta*
_output_shapes
: 
´
!batch_normalization_1/beta/AssignAssignVariableOpbatch_normalization_1/beta,batch_normalization_1/beta/Initializer/zeros*&
 _has_manual_control_dependencies(*
dtype0

.batch_normalization_1/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_1/beta*
_output_shapes	
:*
dtype0
¸
3batch_normalization_1/moving_mean/Initializer/zerosConst*4
_class*
(&loc:@batch_normalization_1/moving_mean*
_output_shapes	
:*
dtype0*
valueB*    
Ń
!batch_normalization_1/moving_meanVarHandleOp*4
_class*
(&loc:@batch_normalization_1/moving_mean*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!batch_normalization_1/moving_mean

Bbatch_normalization_1/moving_mean/IsInitialized/VarIsInitializedOpVarIsInitializedOp!batch_normalization_1/moving_mean*
_output_shapes
: 
É
(batch_normalization_1/moving_mean/AssignAssignVariableOp!batch_normalization_1/moving_mean3batch_normalization_1/moving_mean/Initializer/zeros*&
 _has_manual_control_dependencies(*
dtype0

5batch_normalization_1/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_1/moving_mean*
_output_shapes	
:*
dtype0
ż
6batch_normalization_1/moving_variance/Initializer/onesConst*8
_class.
,*loc:@batch_normalization_1/moving_variance*
_output_shapes	
:*
dtype0*
valueB*  ?
Ý
%batch_normalization_1/moving_varianceVarHandleOp*8
_class.
,*loc:@batch_normalization_1/moving_variance*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%batch_normalization_1/moving_variance

Fbatch_normalization_1/moving_variance/IsInitialized/VarIsInitializedOpVarIsInitializedOp%batch_normalization_1/moving_variance*
_output_shapes
: 
Ô
,batch_normalization_1/moving_variance/AssignAssignVariableOp%batch_normalization_1/moving_variance6batch_normalization_1/moving_variance/Initializer/ones*&
 _has_manual_control_dependencies(*
dtype0

9batch_normalization_1/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_1/moving_variance*
_output_shapes	
:*
dtype0
}
$batch_normalization_1/ReadVariableOpReadVariableOpbatch_normalization_1/gamma*
_output_shapes	
:*
dtype0
~
&batch_normalization_1/ReadVariableOp_1ReadVariableOpbatch_normalization_1/beta*
_output_shapes	
:*
dtype0

5batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOp!batch_normalization_1/moving_mean*
_output_shapes	
:*
dtype0

7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp%batch_normalization_1/moving_variance*
_output_shapes	
:*
dtype0

&batch_normalization_1/FusedBatchNormV3FusedBatchNormV3conv_d1/BiasAdd$batch_normalization_1/ReadVariableOp&batch_normalization_1/ReadVariableOp_15batch_normalization_1/FusedBatchNormV3/ReadVariableOp7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1*
T0*
U0*P
_output_shapes>
<:˙˙˙˙˙˙˙˙˙:::::*
epsilon%o:*
is_training( 
q
Relu_1Relu&batch_normalization_1/FusedBatchNormV3*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
a
dropout_1/IdentityIdentityRelu_1*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ą
*conv_d2/w/Initializer/random_uniform/shapeConst*
_class
loc:@conv_d2/w*
_output_shapes
:*
dtype0*%
valueB"         #   

(conv_d2/w/Initializer/random_uniform/minConst*
_class
loc:@conv_d2/w*
_output_shapes
: *
dtype0*
valueB
 *ł#˝

(conv_d2/w/Initializer/random_uniform/maxConst*
_class
loc:@conv_d2/w*
_output_shapes
: *
dtype0*
valueB
 *ł#=
Ě
2conv_d2/w/Initializer/random_uniform/RandomUniformRandomUniform*conv_d2/w/Initializer/random_uniform/shape*
T0*
_class
loc:@conv_d2/w*'
_output_shapes
:#*
dtype0
Â
(conv_d2/w/Initializer/random_uniform/subSub(conv_d2/w/Initializer/random_uniform/max(conv_d2/w/Initializer/random_uniform/min*
T0*
_class
loc:@conv_d2/w*
_output_shapes
: 
Ý
(conv_d2/w/Initializer/random_uniform/mulMul2conv_d2/w/Initializer/random_uniform/RandomUniform(conv_d2/w/Initializer/random_uniform/sub*
T0*
_class
loc:@conv_d2/w*'
_output_shapes
:#
Ń
$conv_d2/w/Initializer/random_uniformAddV2(conv_d2/w/Initializer/random_uniform/mul(conv_d2/w/Initializer/random_uniform/min*
T0*
_class
loc:@conv_d2/w*'
_output_shapes
:#

	conv_d2/wVarHandleOp*
_class
loc:@conv_d2/w*
_output_shapes
: *
dtype0*
shape:#*
shared_name	conv_d2/w
c
*conv_d2/w/IsInitialized/VarIsInitializedOpVarIsInitializedOp	conv_d2/w*
_output_shapes
: 

conv_d2/w/AssignAssignVariableOp	conv_d2/w$conv_d2/w/Initializer/random_uniform*&
 _has_manual_control_dependencies(*
dtype0
p
conv_d2/w/Read/ReadVariableOpReadVariableOp	conv_d2/w*'
_output_shapes
:#*
dtype0

conv_d2/b/Initializer/ConstConst*
_class
loc:@conv_d2/b*
_output_shapes
:#*
dtype0*
valueB#*    

	conv_d2/bVarHandleOp*
_class
loc:@conv_d2/b*
_output_shapes
: *
dtype0*
shape:#*
shared_name	conv_d2/b
c
*conv_d2/b/IsInitialized/VarIsInitializedOpVarIsInitializedOp	conv_d2/b*
_output_shapes
: 

conv_d2/b/AssignAssignVariableOp	conv_d2/bconv_d2/b/Initializer/Const*&
 _has_manual_control_dependencies(*
dtype0
c
conv_d2/b/Read/ReadVariableOpReadVariableOp	conv_d2/b*
_output_shapes
:#*
dtype0
p
conv_d2/Conv2D/ReadVariableOpReadVariableOp	conv_d2/w*'
_output_shapes
:#*
dtype0
­
conv_d2/Conv2DConv2Ddropout_1/Identityconv_d2/Conv2D/ReadVariableOp*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙#*
paddingSAME*
strides

d
conv_d2/BiasAdd/ReadVariableOpReadVariableOp	conv_d2/b*
_output_shapes
:#*
dtype0

conv_d2/BiasAddBiasAddconv_d2/Conv2Dconv_d2/BiasAdd/ReadVariableOp*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙#
Š
,batch_normalization_2/gamma/Initializer/onesConst*.
_class$
" loc:@batch_normalization_2/gamma*
_output_shapes
:#*
dtype0*
valueB#*  ?
ž
batch_normalization_2/gammaVarHandleOp*.
_class$
" loc:@batch_normalization_2/gamma*
_output_shapes
: *
dtype0*
shape:#*,
shared_namebatch_normalization_2/gamma

<batch_normalization_2/gamma/IsInitialized/VarIsInitializedOpVarIsInitializedOpbatch_normalization_2/gamma*
_output_shapes
: 
ś
"batch_normalization_2/gamma/AssignAssignVariableOpbatch_normalization_2/gamma,batch_normalization_2/gamma/Initializer/ones*&
 _has_manual_control_dependencies(*
dtype0

/batch_normalization_2/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_2/gamma*
_output_shapes
:#*
dtype0
¨
,batch_normalization_2/beta/Initializer/zerosConst*-
_class#
!loc:@batch_normalization_2/beta*
_output_shapes
:#*
dtype0*
valueB#*    
ť
batch_normalization_2/betaVarHandleOp*-
_class#
!loc:@batch_normalization_2/beta*
_output_shapes
: *
dtype0*
shape:#*+
shared_namebatch_normalization_2/beta

;batch_normalization_2/beta/IsInitialized/VarIsInitializedOpVarIsInitializedOpbatch_normalization_2/beta*
_output_shapes
: 
´
!batch_normalization_2/beta/AssignAssignVariableOpbatch_normalization_2/beta,batch_normalization_2/beta/Initializer/zeros*&
 _has_manual_control_dependencies(*
dtype0

.batch_normalization_2/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_2/beta*
_output_shapes
:#*
dtype0
ś
3batch_normalization_2/moving_mean/Initializer/zerosConst*4
_class*
(&loc:@batch_normalization_2/moving_mean*
_output_shapes
:#*
dtype0*
valueB#*    
Đ
!batch_normalization_2/moving_meanVarHandleOp*4
_class*
(&loc:@batch_normalization_2/moving_mean*
_output_shapes
: *
dtype0*
shape:#*2
shared_name#!batch_normalization_2/moving_mean

Bbatch_normalization_2/moving_mean/IsInitialized/VarIsInitializedOpVarIsInitializedOp!batch_normalization_2/moving_mean*
_output_shapes
: 
É
(batch_normalization_2/moving_mean/AssignAssignVariableOp!batch_normalization_2/moving_mean3batch_normalization_2/moving_mean/Initializer/zeros*&
 _has_manual_control_dependencies(*
dtype0

5batch_normalization_2/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_2/moving_mean*
_output_shapes
:#*
dtype0
˝
6batch_normalization_2/moving_variance/Initializer/onesConst*8
_class.
,*loc:@batch_normalization_2/moving_variance*
_output_shapes
:#*
dtype0*
valueB#*  ?
Ü
%batch_normalization_2/moving_varianceVarHandleOp*8
_class.
,*loc:@batch_normalization_2/moving_variance*
_output_shapes
: *
dtype0*
shape:#*6
shared_name'%batch_normalization_2/moving_variance

Fbatch_normalization_2/moving_variance/IsInitialized/VarIsInitializedOpVarIsInitializedOp%batch_normalization_2/moving_variance*
_output_shapes
: 
Ô
,batch_normalization_2/moving_variance/AssignAssignVariableOp%batch_normalization_2/moving_variance6batch_normalization_2/moving_variance/Initializer/ones*&
 _has_manual_control_dependencies(*
dtype0

9batch_normalization_2/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_2/moving_variance*
_output_shapes
:#*
dtype0
|
$batch_normalization_2/ReadVariableOpReadVariableOpbatch_normalization_2/gamma*
_output_shapes
:#*
dtype0
}
&batch_normalization_2/ReadVariableOp_1ReadVariableOpbatch_normalization_2/beta*
_output_shapes
:#*
dtype0

5batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOp!batch_normalization_2/moving_mean*
_output_shapes
:#*
dtype0

7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp%batch_normalization_2/moving_variance*
_output_shapes
:#*
dtype0

&batch_normalization_2/FusedBatchNormV3FusedBatchNormV3conv_d2/BiasAdd$batch_normalization_2/ReadVariableOp&batch_normalization_2/ReadVariableOp_15batch_normalization_2/FusedBatchNormV3/ReadVariableOp7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1*
T0*
U0*K
_output_shapes9
7:˙˙˙˙˙˙˙˙˙#:#:#:#:#:*
epsilon%o:*
is_training( 
p
Relu_2Relu&batch_normalization_2/FusedBatchNormV3*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙#

gc1/AvgPoolAvgPoolconv1/BiasAdd*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@*
ksize
*
paddingSAME*
strides

[

gc1/SquareSquaregc1/AvgPool*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@
b
	gc1/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
H
gc1/MeanMean
gc1/Square	gc1/Const*
T0*
_output_shapes
: 
c
gc1/divRealDivgc1/AvgPoolgc1/Mean*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@

gc2/AvgPoolAvgPoolsbb1/Relu_3*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
ksize
*
paddingSAME*
strides

\

gc2/SquareSquaregc2/AvgPool*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
b
	gc2/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
H
gc2/MeanMean
gc2/Square	gc2/Const*
T0*
_output_shapes
: 
d
gc2/divRealDivgc2/AvgPoolgc2/Mean*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙

gc3/AvgPoolAvgPoolsbb3/Relu_3*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
ksize
*
paddingSAME*
strides

\

gc3/SquareSquaregc3/AvgPool*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
b
	gc3/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
H
gc3/MeanMean
gc3/Square	gc3/Const*
T0*
_output_shapes
: 
d
gc3/divRealDivgc3/AvgPoolgc3/Mean*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
R
SquareSquareRelu_2*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙#
^
ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
<
MeanMeanSquareConst*
T0*
_output_shapes
: 
V
divRealDivRelu_2Mean*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙#
M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :

concatConcatV2gc1/divgc2/divgc3/divdivconcat/axis*
N*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙ă
Ł
+conv_out/w/Initializer/random_uniform/shapeConst*
_class
loc:@conv_out/w*
_output_shapes
:*
dtype0*%
valueB"      ă  #   

)conv_out/w/Initializer/random_uniform/minConst*
_class
loc:@conv_out/w*
_output_shapes
: *
dtype0*
valueB
 */jÜ˝

)conv_out/w/Initializer/random_uniform/maxConst*
_class
loc:@conv_out/w*
_output_shapes
: *
dtype0*
valueB
 */jÜ=
Ď
3conv_out/w/Initializer/random_uniform/RandomUniformRandomUniform+conv_out/w/Initializer/random_uniform/shape*
T0*
_class
loc:@conv_out/w*'
_output_shapes
:ă#*
dtype0
Ć
)conv_out/w/Initializer/random_uniform/subSub)conv_out/w/Initializer/random_uniform/max)conv_out/w/Initializer/random_uniform/min*
T0*
_class
loc:@conv_out/w*
_output_shapes
: 
á
)conv_out/w/Initializer/random_uniform/mulMul3conv_out/w/Initializer/random_uniform/RandomUniform)conv_out/w/Initializer/random_uniform/sub*
T0*
_class
loc:@conv_out/w*'
_output_shapes
:ă#
Ő
%conv_out/w/Initializer/random_uniformAddV2)conv_out/w/Initializer/random_uniform/mul)conv_out/w/Initializer/random_uniform/min*
T0*
_class
loc:@conv_out/w*'
_output_shapes
:ă#


conv_out/wVarHandleOp*
_class
loc:@conv_out/w*
_output_shapes
: *
dtype0*
shape:ă#*
shared_name
conv_out/w
e
+conv_out/w/IsInitialized/VarIsInitializedOpVarIsInitializedOp
conv_out/w*
_output_shapes
: 

conv_out/w/AssignAssignVariableOp
conv_out/w%conv_out/w/Initializer/random_uniform*&
 _has_manual_control_dependencies(*
dtype0
r
conv_out/w/Read/ReadVariableOpReadVariableOp
conv_out/w*'
_output_shapes
:ă#*
dtype0

conv_out/b/Initializer/ConstConst*
_class
loc:@conv_out/b*
_output_shapes
:#*
dtype0*
valueB#*    


conv_out/bVarHandleOp*
_class
loc:@conv_out/b*
_output_shapes
: *
dtype0*
shape:#*
shared_name
conv_out/b
e
+conv_out/b/IsInitialized/VarIsInitializedOpVarIsInitializedOp
conv_out/b*
_output_shapes
: 

conv_out/b/AssignAssignVariableOp
conv_out/bconv_out/b/Initializer/Const*&
 _has_manual_control_dependencies(*
dtype0
e
conv_out/b/Read/ReadVariableOpReadVariableOp
conv_out/b*
_output_shapes
:#*
dtype0
r
conv_out/Conv2D/ReadVariableOpReadVariableOp
conv_out/w*'
_output_shapes
:ă#*
dtype0
Ł
conv_out/Conv2DConv2Dconcatconv_out/Conv2D/ReadVariableOp*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙#*
paddingSAME*
strides

f
conv_out/BiasAdd/ReadVariableOpReadVariableOp
conv_out/b*
_output_shapes
:#*
dtype0

conv_out/BiasAddBiasAddconv_out/Conv2Dconv_out/BiasAdd/ReadVariableOp*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙#
\
mean_out/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :
t
mean_outMeanconv_out/BiasAddmean_out/reduction_indices*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙#
=
ShapeShapemean_out*
T0*
_output_shapes
:
]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
­
strided_sliceStridedSliceShapestrided_slice/stackstrided_slice/stack_1strided_slice/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
ľ
strided_slice_1StridedSliceShapestrided_slice_1/stackstrided_slice_1/stack_1strided_slice_1/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
N
	Fill/dimsPackstrided_slice*
N*
T0*
_output_shapes
:
V
FillFill	Fill/dimsstrided_slice_1*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
f
	transpose	Transposemean_outtranspose/perm*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙#
Ç
CTCBeamSearchDecoderCTCBeamSearchDecoder	transposeFill*O
_output_shapes=
;:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙::˙˙˙˙˙˙˙˙˙*

beam_widthd*
merge_repeated( *
	top_paths
`
decoded/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
˙˙˙˙˙˙˙˙˙
Ŕ
decodedSparseToDenseCTCBeamSearchDecoderCTCBeamSearchDecoder:2CTCBeamSearchDecoder:1decoded/default_value*
T0	*
Tindices0	*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
a
CastCastCTCBeamSearchDecoder:1*

DstT0*

SrcT0	*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
°
edit_distanceEditDistanceCTCBeamSearchDecoderCastCTCBeamSearchDecoder:2Placeholder_2Placeholder_1Placeholder*
T0*
_output_shapes
:*
	normalize( 
<
RankRankedit_distance*
T0*
_output_shapes
: 
M
range/startConst*
_output_shapes
: *
dtype0*
value	B : 
M
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :
S
rangeRangerange/startRankrange/delta*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
A
SumSumedit_distancerange*
T0*
_output_shapes
: 
}
CTCLossCTCLoss	transposePlaceholder_2Placeholder_1Fill*:
_output_shapes(
&:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙#
Q
Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
A
Mean_1MeanCTCLossConst_1*
T0*
_output_shapes
: 

"Variable/Initializer/initial_valueConst*
_class
loc:@Variable*
_output_shapes
: *
dtype0*
value	B : 

VariableVarHandleOp*
_class
loc:@Variable*
_output_shapes
: *
dtype0*
shape: *
shared_name
Variable
a
)Variable/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable*
_output_shapes
: 

Variable/AssignAssignVariableOpVariable"Variable/Initializer/initial_value*&
 _has_manual_control_dependencies(*
dtype0
]
Variable/Read/ReadVariableOpReadVariableOpVariable*
_output_shapes
: *
dtype0
k
&ExponentialDecay/initial_learning_rateConst*
_output_shapes
: *
dtype0*
valueB
 *o:
Z
ExponentialDecay/Cast/xConst*
_output_shapes
: *
dtype0*
value
B :Đ
f
ExponentialDecay/CastCastExponentialDecay/Cast/x*

DstT0*

SrcT0*
_output_shapes
: 
^
ExponentialDecay/Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *fff?
g
&ExponentialDecay/Cast_2/ReadVariableOpReadVariableOpVariable*
_output_shapes
: *
dtype0
w
ExponentialDecay/Cast_2Cast&ExponentialDecay/Cast_2/ReadVariableOp*

DstT0*

SrcT0*
_output_shapes
: 
t
ExponentialDecay/truedivRealDivExponentialDecay/Cast_2ExponentialDecay/Cast*
T0*
_output_shapes
: 
Z
ExponentialDecay/FloorFloorExponentialDecay/truediv*
T0*
_output_shapes
: 
o
ExponentialDecay/PowPowExponentialDecay/Cast_1/xExponentialDecay/Floor*
T0*
_output_shapes
: 
v
ExponentialDecayMul&ExponentialDecay/initial_learning_rateExponentialDecay/Pow*
T0*
_output_shapes
: 
F
gradients/ShapeShapeCTCLoss*
T0*
_output_shapes
:
^
gradients/grad_ys_0/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
u
gradients/grad_ys_0Fillgradients/Shapegradients/grad_ys_0/Const*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
b
gradients/zeros_like	ZerosLike	CTCLoss:1*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙#

&gradients/CTCLoss_grad/PreventGradientPreventGradient	CTCLoss:1*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙#*
messageCurrently there is no way to take the second  derivative of ctc_loss due to the fused implementation's interaction  with tf.gradients()
p
%gradients/CTCLoss_grad/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙

!gradients/CTCLoss_grad/ExpandDims
ExpandDimsgradients/grad_ys_0%gradients/CTCLoss_grad/ExpandDims/dim*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
˘
gradients/CTCLoss_grad/mulMul!gradients/CTCLoss_grad/ExpandDims&gradients/CTCLoss_grad/PreventGradient*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙#
k
*gradients/transpose_grad/InvertPermutationInvertPermutationtranspose/perm*
_output_shapes
:
­
"gradients/transpose_grad/transpose	Transposegradients/CTCLoss_grad/mul*gradients/transpose_grad/InvertPermutation*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙#
]
gradients/mean_out_grad/ShapeShapeconv_out/BiasAdd*
T0*
_output_shapes
:

gradients/mean_out_grad/SizeConst*0
_class&
$"loc:@gradients/mean_out_grad/Shape*
_output_shapes
: *
dtype0*
value	B :
ą
gradients/mean_out_grad/addAddV2mean_out/reduction_indicesgradients/mean_out_grad/Size*
T0*0
_class&
$"loc:@gradients/mean_out_grad/Shape*
_output_shapes
: 
ľ
gradients/mean_out_grad/modFloorModgradients/mean_out_grad/addgradients/mean_out_grad/Size*
T0*0
_class&
$"loc:@gradients/mean_out_grad/Shape*
_output_shapes
: 

gradients/mean_out_grad/Shape_1Const*0
_class&
$"loc:@gradients/mean_out_grad/Shape*
_output_shapes
: *
dtype0*
valueB 

#gradients/mean_out_grad/range/startConst*0
_class&
$"loc:@gradients/mean_out_grad/Shape*
_output_shapes
: *
dtype0*
value	B : 

#gradients/mean_out_grad/range/deltaConst*0
_class&
$"loc:@gradients/mean_out_grad/Shape*
_output_shapes
: *
dtype0*
value	B :
Ü
gradients/mean_out_grad/rangeRange#gradients/mean_out_grad/range/startgradients/mean_out_grad/Size#gradients/mean_out_grad/range/delta*0
_class&
$"loc:@gradients/mean_out_grad/Shape*
_output_shapes
:

"gradients/mean_out_grad/ones/ConstConst*0
_class&
$"loc:@gradients/mean_out_grad/Shape*
_output_shapes
: *
dtype0*
value	B :
ź
gradients/mean_out_grad/onesFillgradients/mean_out_grad/Shape_1"gradients/mean_out_grad/ones/Const*
T0*0
_class&
$"loc:@gradients/mean_out_grad/Shape*
_output_shapes
: 

%gradients/mean_out_grad/DynamicStitchDynamicStitchgradients/mean_out_grad/rangegradients/mean_out_grad/modgradients/mean_out_grad/Shapegradients/mean_out_grad/ones*
N*
T0*0
_class&
$"loc:@gradients/mean_out_grad/Shape*
_output_shapes
:
Ę
gradients/mean_out_grad/ReshapeReshape"gradients/transpose_grad/transpose%gradients/mean_out_grad/DynamicStitch*
T0*J
_output_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
Ź
#gradients/mean_out_grad/BroadcastToBroadcastTogradients/mean_out_grad/Reshapegradients/mean_out_grad/Shape*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙#
_
gradients/mean_out_grad/Shape_2Shapeconv_out/BiasAdd*
T0*
_output_shapes
:
W
gradients/mean_out_grad/Shape_3Shapemean_out*
T0*
_output_shapes
:
g
gradients/mean_out_grad/ConstConst*
_output_shapes
:*
dtype0*
valueB: 

gradients/mean_out_grad/ProdProdgradients/mean_out_grad/Shape_2gradients/mean_out_grad/Const*
T0*
_output_shapes
: 
i
gradients/mean_out_grad/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 

gradients/mean_out_grad/Prod_1Prodgradients/mean_out_grad/Shape_3gradients/mean_out_grad/Const_1*
T0*
_output_shapes
: 
c
!gradients/mean_out_grad/Maximum/yConst*
_output_shapes
: *
dtype0*
value	B :

gradients/mean_out_grad/MaximumMaximumgradients/mean_out_grad/Prod_1!gradients/mean_out_grad/Maximum/y*
T0*
_output_shapes
: 

 gradients/mean_out_grad/floordivFloorDivgradients/mean_out_grad/Prodgradients/mean_out_grad/Maximum*
T0*
_output_shapes
: 
v
gradients/mean_out_grad/CastCast gradients/mean_out_grad/floordiv*

DstT0*

SrcT0*
_output_shapes
: 
Ď
gradients/mean_out_grad/truedivRealDiv#gradients/mean_out_grad/BroadcastTogradients/mean_out_grad/Cast*
T0*&
 _has_manual_control_dependencies(*/
_output_shapes
:˙˙˙˙˙˙˙˙˙#
¨
+gradients/conv_out/BiasAdd_grad/BiasAddGradBiasAddGradgradients/mean_out_grad/truediv*
T0*&
 _has_manual_control_dependencies(*
_output_shapes
:#
°
0gradients/conv_out/BiasAdd_grad/tuple/group_depsNoOp,^gradients/conv_out/BiasAdd_grad/BiasAddGrad ^gradients/mean_out_grad/truediv*&
 _has_manual_control_dependencies(

8gradients/conv_out/BiasAdd_grad/tuple/control_dependencyIdentitygradients/mean_out_grad/truediv1^gradients/conv_out/BiasAdd_grad/tuple/group_deps*
T0*2
_class(
&$loc:@gradients/mean_out_grad/truediv*/
_output_shapes
:˙˙˙˙˙˙˙˙˙#

:gradients/conv_out/BiasAdd_grad/tuple/control_dependency_1Identity+gradients/conv_out/BiasAdd_grad/BiasAddGrad1^gradients/conv_out/BiasAdd_grad/tuple/group_deps*
T0*>
_class4
20loc:@gradients/conv_out/BiasAdd_grad/BiasAddGrad*
_output_shapes
:#

%gradients/conv_out/Conv2D_grad/ShapeNShapeNconcatconv_out/Conv2D/ReadVariableOp*
N*
T0* 
_output_shapes
::
Ő
2gradients/conv_out/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput%gradients/conv_out/Conv2D_grad/ShapeNconv_out/Conv2D/ReadVariableOp8gradients/conv_out/BiasAdd_grad/tuple/control_dependency*
T0*&
 _has_manual_control_dependencies(*0
_output_shapes
:˙˙˙˙˙˙˙˙˙ă*
paddingSAME*
strides

¸
3gradients/conv_out/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterconcat'gradients/conv_out/Conv2D_grad/ShapeN:18gradients/conv_out/BiasAdd_grad/tuple/control_dependency*
T0*&
 _has_manual_control_dependencies(*'
_output_shapes
:ă#*
paddingSAME*
strides

Ę
/gradients/conv_out/Conv2D_grad/tuple/group_depsNoOp4^gradients/conv_out/Conv2D_grad/Conv2DBackpropFilter3^gradients/conv_out/Conv2D_grad/Conv2DBackpropInput*&
 _has_manual_control_dependencies(
Ť
7gradients/conv_out/Conv2D_grad/tuple/control_dependencyIdentity2gradients/conv_out/Conv2D_grad/Conv2DBackpropInput0^gradients/conv_out/Conv2D_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients/conv_out/Conv2D_grad/Conv2DBackpropInput*0
_output_shapes
:˙˙˙˙˙˙˙˙˙ă
Ś
9gradients/conv_out/Conv2D_grad/tuple/control_dependency_1Identity3gradients/conv_out/Conv2D_grad/Conv2DBackpropFilter0^gradients/conv_out/Conv2D_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/conv_out/Conv2D_grad/Conv2DBackpropFilter*'
_output_shapes
:ă#
\
gradients/concat_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :
o
gradients/concat_grad/modFloorModconcat/axisgradients/concat_grad/Rank*
T0*
_output_shapes
: 
R
gradients/concat_grad/ShapeShapegc1/div*
T0*
_output_shapes
:

gradients/concat_grad/ShapeNShapeNgc1/divgc2/divgc3/divdiv*
N*
T0*,
_output_shapes
::::

"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/modgradients/concat_grad/ShapeNgradients/concat_grad/ShapeN:1gradients/concat_grad/ShapeN:2gradients/concat_grad/ShapeN:3*
N*,
_output_shapes
::::

gradients/concat_grad/SliceSlice7gradients/conv_out/Conv2D_grad/tuple/control_dependency"gradients/concat_grad/ConcatOffsetgradients/concat_grad/ShapeN*
Index0*
T0*&
 _has_manual_control_dependencies(*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@

gradients/concat_grad/Slice_1Slice7gradients/conv_out/Conv2D_grad/tuple/control_dependency$gradients/concat_grad/ConcatOffset:1gradients/concat_grad/ShapeN:1*
Index0*
T0*&
 _has_manual_control_dependencies(*0
_output_shapes
:˙˙˙˙˙˙˙˙˙

gradients/concat_grad/Slice_2Slice7gradients/conv_out/Conv2D_grad/tuple/control_dependency$gradients/concat_grad/ConcatOffset:2gradients/concat_grad/ShapeN:2*
Index0*
T0*&
 _has_manual_control_dependencies(*0
_output_shapes
:˙˙˙˙˙˙˙˙˙

gradients/concat_grad/Slice_3Slice7gradients/conv_out/Conv2D_grad/tuple/control_dependency$gradients/concat_grad/ConcatOffset:3gradients/concat_grad/ShapeN:3*
Index0*
T0*&
 _has_manual_control_dependencies(*/
_output_shapes
:˙˙˙˙˙˙˙˙˙#
Ô
&gradients/concat_grad/tuple/group_depsNoOp^gradients/concat_grad/Slice^gradients/concat_grad/Slice_1^gradients/concat_grad/Slice_2^gradients/concat_grad/Slice_3*&
 _has_manual_control_dependencies(
ę
.gradients/concat_grad/tuple/control_dependencyIdentitygradients/concat_grad/Slice'^gradients/concat_grad/tuple/group_deps*
T0*.
_class$
" loc:@gradients/concat_grad/Slice*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@
ń
0gradients/concat_grad/tuple/control_dependency_1Identitygradients/concat_grad/Slice_1'^gradients/concat_grad/tuple/group_deps*
T0*0
_class&
$"loc:@gradients/concat_grad/Slice_1*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
ń
0gradients/concat_grad/tuple/control_dependency_2Identitygradients/concat_grad/Slice_2'^gradients/concat_grad/tuple/group_deps*
T0*0
_class&
$"loc:@gradients/concat_grad/Slice_2*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
đ
0gradients/concat_grad/tuple/control_dependency_3Identitygradients/concat_grad/Slice_3'^gradients/concat_grad/tuple/group_deps*
T0*0
_class&
$"loc:@gradients/concat_grad/Slice_3*/
_output_shapes
:˙˙˙˙˙˙˙˙˙#
W
gradients/gc1/div_grad/ShapeShapegc1/AvgPool*
T0*
_output_shapes
:
a
gradients/gc1/div_grad/Shape_1Const*
_output_shapes
: *
dtype0*
valueB 
ˇ
,gradients/gc1/div_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/gc1/div_grad/Shapegradients/gc1/div_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙

gradients/gc1/div_grad/RealDivRealDiv.gradients/concat_grad/tuple/control_dependencygc1/Mean*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@

gradients/gc1/div_grad/SumSumgradients/gc1/div_grad/RealDiv,gradients/gc1/div_grad/BroadcastGradientArgs*
T0*
_output_shapes
:
Ĺ
gradients/gc1/div_grad/ReshapeReshapegradients/gc1/div_grad/Sumgradients/gc1/div_grad/Shape*
T0*&
 _has_manual_control_dependencies(*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@
h
gradients/gc1/div_grad/NegNeggc1/AvgPool*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@

 gradients/gc1/div_grad/RealDiv_1RealDivgradients/gc1/div_grad/Neggc1/Mean*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@

 gradients/gc1/div_grad/RealDiv_2RealDiv gradients/gc1/div_grad/RealDiv_1gc1/Mean*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@
­
gradients/gc1/div_grad/mulMul.gradients/concat_grad/tuple/control_dependency gradients/gc1/div_grad/RealDiv_2*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@

gradients/gc1/div_grad/Sum_1Sumgradients/gc1/div_grad/mul.gradients/gc1/div_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:
˛
 gradients/gc1/div_grad/Reshape_1Reshapegradients/gc1/div_grad/Sum_1gradients/gc1/div_grad/Shape_1*
T0*&
 _has_manual_control_dependencies(*
_output_shapes
: 

'gradients/gc1/div_grad/tuple/group_depsNoOp^gradients/gc1/div_grad/Reshape!^gradients/gc1/div_grad/Reshape_1*&
 _has_manual_control_dependencies(
ň
/gradients/gc1/div_grad/tuple/control_dependencyIdentitygradients/gc1/div_grad/Reshape(^gradients/gc1/div_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/gc1/div_grad/Reshape*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@
ß
1gradients/gc1/div_grad/tuple/control_dependency_1Identity gradients/gc1/div_grad/Reshape_1(^gradients/gc1/div_grad/tuple/group_deps*
T0*3
_class)
'%loc:@gradients/gc1/div_grad/Reshape_1*
_output_shapes
: 
W
gradients/gc2/div_grad/ShapeShapegc2/AvgPool*
T0*
_output_shapes
:
a
gradients/gc2/div_grad/Shape_1Const*
_output_shapes
: *
dtype0*
valueB 
ˇ
,gradients/gc2/div_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/gc2/div_grad/Shapegradients/gc2/div_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
 
gradients/gc2/div_grad/RealDivRealDiv0gradients/concat_grad/tuple/control_dependency_1gc2/Mean*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙

gradients/gc2/div_grad/SumSumgradients/gc2/div_grad/RealDiv,gradients/gc2/div_grad/BroadcastGradientArgs*
T0*
_output_shapes
:
Ć
gradients/gc2/div_grad/ReshapeReshapegradients/gc2/div_grad/Sumgradients/gc2/div_grad/Shape*
T0*&
 _has_manual_control_dependencies(*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
i
gradients/gc2/div_grad/NegNeggc2/AvgPool*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙

 gradients/gc2/div_grad/RealDiv_1RealDivgradients/gc2/div_grad/Neggc2/Mean*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙

 gradients/gc2/div_grad/RealDiv_2RealDiv gradients/gc2/div_grad/RealDiv_1gc2/Mean*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
°
gradients/gc2/div_grad/mulMul0gradients/concat_grad/tuple/control_dependency_1 gradients/gc2/div_grad/RealDiv_2*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙

gradients/gc2/div_grad/Sum_1Sumgradients/gc2/div_grad/mul.gradients/gc2/div_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:
˛
 gradients/gc2/div_grad/Reshape_1Reshapegradients/gc2/div_grad/Sum_1gradients/gc2/div_grad/Shape_1*
T0*&
 _has_manual_control_dependencies(*
_output_shapes
: 

'gradients/gc2/div_grad/tuple/group_depsNoOp^gradients/gc2/div_grad/Reshape!^gradients/gc2/div_grad/Reshape_1*&
 _has_manual_control_dependencies(
ó
/gradients/gc2/div_grad/tuple/control_dependencyIdentitygradients/gc2/div_grad/Reshape(^gradients/gc2/div_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/gc2/div_grad/Reshape*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
ß
1gradients/gc2/div_grad/tuple/control_dependency_1Identity gradients/gc2/div_grad/Reshape_1(^gradients/gc2/div_grad/tuple/group_deps*
T0*3
_class)
'%loc:@gradients/gc2/div_grad/Reshape_1*
_output_shapes
: 
W
gradients/gc3/div_grad/ShapeShapegc3/AvgPool*
T0*
_output_shapes
:
a
gradients/gc3/div_grad/Shape_1Const*
_output_shapes
: *
dtype0*
valueB 
ˇ
,gradients/gc3/div_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/gc3/div_grad/Shapegradients/gc3/div_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
 
gradients/gc3/div_grad/RealDivRealDiv0gradients/concat_grad/tuple/control_dependency_2gc3/Mean*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙

gradients/gc3/div_grad/SumSumgradients/gc3/div_grad/RealDiv,gradients/gc3/div_grad/BroadcastGradientArgs*
T0*
_output_shapes
:
Ć
gradients/gc3/div_grad/ReshapeReshapegradients/gc3/div_grad/Sumgradients/gc3/div_grad/Shape*
T0*&
 _has_manual_control_dependencies(*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
i
gradients/gc3/div_grad/NegNeggc3/AvgPool*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙

 gradients/gc3/div_grad/RealDiv_1RealDivgradients/gc3/div_grad/Neggc3/Mean*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙

 gradients/gc3/div_grad/RealDiv_2RealDiv gradients/gc3/div_grad/RealDiv_1gc3/Mean*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
°
gradients/gc3/div_grad/mulMul0gradients/concat_grad/tuple/control_dependency_2 gradients/gc3/div_grad/RealDiv_2*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙

gradients/gc3/div_grad/Sum_1Sumgradients/gc3/div_grad/mul.gradients/gc3/div_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:
˛
 gradients/gc3/div_grad/Reshape_1Reshapegradients/gc3/div_grad/Sum_1gradients/gc3/div_grad/Shape_1*
T0*&
 _has_manual_control_dependencies(*
_output_shapes
: 

'gradients/gc3/div_grad/tuple/group_depsNoOp^gradients/gc3/div_grad/Reshape!^gradients/gc3/div_grad/Reshape_1*&
 _has_manual_control_dependencies(
ó
/gradients/gc3/div_grad/tuple/control_dependencyIdentitygradients/gc3/div_grad/Reshape(^gradients/gc3/div_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/gc3/div_grad/Reshape*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
ß
1gradients/gc3/div_grad/tuple/control_dependency_1Identity gradients/gc3/div_grad/Reshape_1(^gradients/gc3/div_grad/tuple/group_deps*
T0*3
_class)
'%loc:@gradients/gc3/div_grad/Reshape_1*
_output_shapes
: 
N
gradients/div_grad/ShapeShapeRelu_2*
T0*
_output_shapes
:
]
gradients/div_grad/Shape_1Const*
_output_shapes
: *
dtype0*
valueB 
Ť
(gradients/div_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/div_grad/Shapegradients/div_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙

gradients/div_grad/RealDivRealDiv0gradients/concat_grad/tuple/control_dependency_3Mean*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙#

gradients/div_grad/SumSumgradients/div_grad/RealDiv(gradients/div_grad/BroadcastGradientArgs*
T0*
_output_shapes
:
š
gradients/div_grad/ReshapeReshapegradients/div_grad/Sumgradients/div_grad/Shape*
T0*&
 _has_manual_control_dependencies(*/
_output_shapes
:˙˙˙˙˙˙˙˙˙#
_
gradients/div_grad/NegNegRelu_2*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙#

gradients/div_grad/RealDiv_1RealDivgradients/div_grad/NegMean*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙#

gradients/div_grad/RealDiv_2RealDivgradients/div_grad/RealDiv_1Mean*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙#
§
gradients/div_grad/mulMul0gradients/concat_grad/tuple/control_dependency_3gradients/div_grad/RealDiv_2*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙#

gradients/div_grad/Sum_1Sumgradients/div_grad/mul*gradients/div_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:
Ś
gradients/div_grad/Reshape_1Reshapegradients/div_grad/Sum_1gradients/div_grad/Shape_1*
T0*&
 _has_manual_control_dependencies(*
_output_shapes
: 

#gradients/div_grad/tuple/group_depsNoOp^gradients/div_grad/Reshape^gradients/div_grad/Reshape_1*&
 _has_manual_control_dependencies(
â
+gradients/div_grad/tuple/control_dependencyIdentitygradients/div_grad/Reshape$^gradients/div_grad/tuple/group_deps*
T0*-
_class#
!loc:@gradients/div_grad/Reshape*/
_output_shapes
:˙˙˙˙˙˙˙˙˙#
Ď
-gradients/div_grad/tuple/control_dependency_1Identitygradients/div_grad/Reshape_1$^gradients/div_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/div_grad/Reshape_1*
_output_shapes
: 
~
%gradients/gc1/Mean_grad/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            
ľ
gradients/gc1/Mean_grad/ReshapeReshape1gradients/gc1/div_grad/tuple/control_dependency_1%gradients/gc1/Mean_grad/Reshape/shape*
T0*&
_output_shapes
:
W
gradients/gc1/Mean_grad/ShapeShape
gc1/Square*
T0*
_output_shapes
:

gradients/gc1/Mean_grad/TileTilegradients/gc1/Mean_grad/Reshapegradients/gc1/Mean_grad/Shape*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@
Y
gradients/gc1/Mean_grad/Shape_1Shape
gc1/Square*
T0*
_output_shapes
:
b
gradients/gc1/Mean_grad/Shape_2Const*
_output_shapes
: *
dtype0*
valueB 
g
gradients/gc1/Mean_grad/ConstConst*
_output_shapes
:*
dtype0*
valueB: 

gradients/gc1/Mean_grad/ProdProdgradients/gc1/Mean_grad/Shape_1gradients/gc1/Mean_grad/Const*
T0*
_output_shapes
: 
i
gradients/gc1/Mean_grad/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 

gradients/gc1/Mean_grad/Prod_1Prodgradients/gc1/Mean_grad/Shape_2gradients/gc1/Mean_grad/Const_1*
T0*
_output_shapes
: 
c
!gradients/gc1/Mean_grad/Maximum/yConst*
_output_shapes
: *
dtype0*
value	B :

gradients/gc1/Mean_grad/MaximumMaximumgradients/gc1/Mean_grad/Prod_1!gradients/gc1/Mean_grad/Maximum/y*
T0*
_output_shapes
: 

 gradients/gc1/Mean_grad/floordivFloorDivgradients/gc1/Mean_grad/Prodgradients/gc1/Mean_grad/Maximum*
T0*
_output_shapes
: 
v
gradients/gc1/Mean_grad/CastCast gradients/gc1/Mean_grad/floordiv*

DstT0*

SrcT0*
_output_shapes
: 
Č
gradients/gc1/Mean_grad/truedivRealDivgradients/gc1/Mean_grad/Tilegradients/gc1/Mean_grad/Cast*
T0*&
 _has_manual_control_dependencies(*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@
~
%gradients/gc2/Mean_grad/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            
ľ
gradients/gc2/Mean_grad/ReshapeReshape1gradients/gc2/div_grad/tuple/control_dependency_1%gradients/gc2/Mean_grad/Reshape/shape*
T0*&
_output_shapes
:
W
gradients/gc2/Mean_grad/ShapeShape
gc2/Square*
T0*
_output_shapes
:

gradients/gc2/Mean_grad/TileTilegradients/gc2/Mean_grad/Reshapegradients/gc2/Mean_grad/Shape*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
Y
gradients/gc2/Mean_grad/Shape_1Shape
gc2/Square*
T0*
_output_shapes
:
b
gradients/gc2/Mean_grad/Shape_2Const*
_output_shapes
: *
dtype0*
valueB 
g
gradients/gc2/Mean_grad/ConstConst*
_output_shapes
:*
dtype0*
valueB: 

gradients/gc2/Mean_grad/ProdProdgradients/gc2/Mean_grad/Shape_1gradients/gc2/Mean_grad/Const*
T0*
_output_shapes
: 
i
gradients/gc2/Mean_grad/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 

gradients/gc2/Mean_grad/Prod_1Prodgradients/gc2/Mean_grad/Shape_2gradients/gc2/Mean_grad/Const_1*
T0*
_output_shapes
: 
c
!gradients/gc2/Mean_grad/Maximum/yConst*
_output_shapes
: *
dtype0*
value	B :

gradients/gc2/Mean_grad/MaximumMaximumgradients/gc2/Mean_grad/Prod_1!gradients/gc2/Mean_grad/Maximum/y*
T0*
_output_shapes
: 

 gradients/gc2/Mean_grad/floordivFloorDivgradients/gc2/Mean_grad/Prodgradients/gc2/Mean_grad/Maximum*
T0*
_output_shapes
: 
v
gradients/gc2/Mean_grad/CastCast gradients/gc2/Mean_grad/floordiv*

DstT0*

SrcT0*
_output_shapes
: 
É
gradients/gc2/Mean_grad/truedivRealDivgradients/gc2/Mean_grad/Tilegradients/gc2/Mean_grad/Cast*
T0*&
 _has_manual_control_dependencies(*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
~
%gradients/gc3/Mean_grad/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            
ľ
gradients/gc3/Mean_grad/ReshapeReshape1gradients/gc3/div_grad/tuple/control_dependency_1%gradients/gc3/Mean_grad/Reshape/shape*
T0*&
_output_shapes
:
W
gradients/gc3/Mean_grad/ShapeShape
gc3/Square*
T0*
_output_shapes
:

gradients/gc3/Mean_grad/TileTilegradients/gc3/Mean_grad/Reshapegradients/gc3/Mean_grad/Shape*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
Y
gradients/gc3/Mean_grad/Shape_1Shape
gc3/Square*
T0*
_output_shapes
:
b
gradients/gc3/Mean_grad/Shape_2Const*
_output_shapes
: *
dtype0*
valueB 
g
gradients/gc3/Mean_grad/ConstConst*
_output_shapes
:*
dtype0*
valueB: 

gradients/gc3/Mean_grad/ProdProdgradients/gc3/Mean_grad/Shape_1gradients/gc3/Mean_grad/Const*
T0*
_output_shapes
: 
i
gradients/gc3/Mean_grad/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 

gradients/gc3/Mean_grad/Prod_1Prodgradients/gc3/Mean_grad/Shape_2gradients/gc3/Mean_grad/Const_1*
T0*
_output_shapes
: 
c
!gradients/gc3/Mean_grad/Maximum/yConst*
_output_shapes
: *
dtype0*
value	B :

gradients/gc3/Mean_grad/MaximumMaximumgradients/gc3/Mean_grad/Prod_1!gradients/gc3/Mean_grad/Maximum/y*
T0*
_output_shapes
: 

 gradients/gc3/Mean_grad/floordivFloorDivgradients/gc3/Mean_grad/Prodgradients/gc3/Mean_grad/Maximum*
T0*
_output_shapes
: 
v
gradients/gc3/Mean_grad/CastCast gradients/gc3/Mean_grad/floordiv*

DstT0*

SrcT0*
_output_shapes
: 
É
gradients/gc3/Mean_grad/truedivRealDivgradients/gc3/Mean_grad/Tilegradients/gc3/Mean_grad/Cast*
T0*&
 _has_manual_control_dependencies(*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
z
!gradients/Mean_grad/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            
Š
gradients/Mean_grad/ReshapeReshape-gradients/div_grad/tuple/control_dependency_1!gradients/Mean_grad/Reshape/shape*
T0*&
_output_shapes
:
O
gradients/Mean_grad/ShapeShapeSquare*
T0*
_output_shapes
:

gradients/Mean_grad/TileTilegradients/Mean_grad/Reshapegradients/Mean_grad/Shape*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙#
Q
gradients/Mean_grad/Shape_1ShapeSquare*
T0*
_output_shapes
:
^
gradients/Mean_grad/Shape_2Const*
_output_shapes
: *
dtype0*
valueB 
c
gradients/Mean_grad/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
y
gradients/Mean_grad/ProdProdgradients/Mean_grad/Shape_1gradients/Mean_grad/Const*
T0*
_output_shapes
: 
e
gradients/Mean_grad/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
}
gradients/Mean_grad/Prod_1Prodgradients/Mean_grad/Shape_2gradients/Mean_grad/Const_1*
T0*
_output_shapes
: 
_
gradients/Mean_grad/Maximum/yConst*
_output_shapes
: *
dtype0*
value	B :

gradients/Mean_grad/MaximumMaximumgradients/Mean_grad/Prod_1gradients/Mean_grad/Maximum/y*
T0*
_output_shapes
: 

gradients/Mean_grad/floordivFloorDivgradients/Mean_grad/Prodgradients/Mean_grad/Maximum*
T0*
_output_shapes
: 
n
gradients/Mean_grad/CastCastgradients/Mean_grad/floordiv*

DstT0*

SrcT0*
_output_shapes
: 
ź
gradients/Mean_grad/truedivRealDivgradients/Mean_grad/Tilegradients/Mean_grad/Cast*
T0*&
 _has_manual_control_dependencies(*/
_output_shapes
:˙˙˙˙˙˙˙˙˙#

gradients/gc1/Square_grad/ConstConst ^gradients/gc1/Mean_grad/truediv*
_output_shapes
: *
dtype0*
valueB
 *   @

gradients/gc1/Square_grad/MulMulgc1/AvgPoolgradients/gc1/Square_grad/Const*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@
 
gradients/gc1/Square_grad/Mul_1Mulgradients/gc1/Mean_grad/truedivgradients/gc1/Square_grad/Mul*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@

gradients/gc2/Square_grad/ConstConst ^gradients/gc2/Mean_grad/truediv*
_output_shapes
: *
dtype0*
valueB
 *   @

gradients/gc2/Square_grad/MulMulgc2/AvgPoolgradients/gc2/Square_grad/Const*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ą
gradients/gc2/Square_grad/Mul_1Mulgradients/gc2/Mean_grad/truedivgradients/gc2/Square_grad/Mul*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙

gradients/gc3/Square_grad/ConstConst ^gradients/gc3/Mean_grad/truediv*
_output_shapes
: *
dtype0*
valueB
 *   @

gradients/gc3/Square_grad/MulMulgc3/AvgPoolgradients/gc3/Square_grad/Const*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ą
gradients/gc3/Square_grad/Mul_1Mulgradients/gc3/Mean_grad/truedivgradients/gc3/Square_grad/Mul*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
~
gradients/Square_grad/ConstConst^gradients/Mean_grad/truediv*
_output_shapes
: *
dtype0*
valueB
 *   @

gradients/Square_grad/MulMulRelu_2gradients/Square_grad/Const*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙#

gradients/Square_grad/Mul_1Mulgradients/Mean_grad/truedivgradients/Square_grad/Mul*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙#
Ţ
gradients/AddNAddN/gradients/gc1/div_grad/tuple/control_dependencygradients/gc1/Square_grad/Mul_1*
N*
T0*1
_class'
%#loc:@gradients/gc1/div_grad/Reshape*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@
]
 gradients/gc1/AvgPool_grad/ShapeShapeconv1/BiasAdd*
T0*
_output_shapes
:
Ü
&gradients/gc1/AvgPool_grad/AvgPoolGradAvgPoolGrad gradients/gc1/AvgPool_grad/Shapegradients/AddN*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙^@*
ksize
*
paddingSAME*
strides

á
gradients/AddN_1AddN/gradients/gc2/div_grad/tuple/control_dependencygradients/gc2/Square_grad/Mul_1*
N*
T0*1
_class'
%#loc:@gradients/gc2/div_grad/Reshape*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
[
 gradients/gc2/AvgPool_grad/ShapeShapesbb1/Relu_3*
T0*
_output_shapes
:
ß
&gradients/gc2/AvgPool_grad/AvgPoolGradAvgPoolGrad gradients/gc2/AvgPool_grad/Shapegradients/AddN_1*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙^*
ksize
*
paddingSAME*
strides

á
gradients/AddN_2AddN/gradients/gc3/div_grad/tuple/control_dependencygradients/gc3/Square_grad/Mul_1*
N*
T0*1
_class'
%#loc:@gradients/gc3/div_grad/Reshape*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
[
 gradients/gc3/AvgPool_grad/ShapeShapesbb3/Relu_3*
T0*
_output_shapes
:
ß
&gradients/gc3/AvgPool_grad/AvgPoolGradAvgPoolGrad gradients/gc3/AvgPool_grad/Shapegradients/AddN_2*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙/*
ksize
*
paddingSAME*
strides

Ô
gradients/AddN_3AddN+gradients/div_grad/tuple/control_dependencygradients/Square_grad/Mul_1*
N*
T0*-
_class#
!loc:@gradients/div_grad/Reshape*/
_output_shapes
:˙˙˙˙˙˙˙˙˙#
~
gradients/Relu_2_grad/ReluGradReluGradgradients/AddN_3Relu_2*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙#
r
gradients/zeros_like_1	ZerosLike(batch_normalization_2/FusedBatchNormV3:1*
T0*
_output_shapes
:#
r
gradients/zeros_like_2	ZerosLike(batch_normalization_2/FusedBatchNormV3:2*
T0*
_output_shapes
:#
r
gradients/zeros_like_3	ZerosLike(batch_normalization_2/FusedBatchNormV3:3*
T0*
_output_shapes
:#
r
gradients/zeros_like_4	ZerosLike(batch_normalization_2/FusedBatchNormV3:4*
T0*
_output_shapes
:#
p
gradients/zeros_like_5	ZerosLike(batch_normalization_2/FusedBatchNormV3:5*
T0*
_output_shapes
:
÷
Jgradients/batch_normalization_2/FusedBatchNormV3_grad/FusedBatchNormGradV3FusedBatchNormGradV3gradients/Relu_2_grad/ReluGradconv_d2/BiasAdd$batch_normalization_2/ReadVariableOp5batch_normalization_2/FusedBatchNormV3/ReadVariableOp7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1(batch_normalization_2/FusedBatchNormV3:5*
T0*
U0*&
 _has_manual_control_dependencies(*C
_output_shapes1
/:˙˙˙˙˙˙˙˙˙#:#:#: : *
epsilon%o:*
is_training( 
Ă
Fgradients/batch_normalization_2/FusedBatchNormV3_grad/tuple/group_depsNoOpK^gradients/batch_normalization_2/FusedBatchNormV3_grad/FusedBatchNormGradV3*&
 _has_manual_control_dependencies(
°
Ngradients/batch_normalization_2/FusedBatchNormV3_grad/tuple/control_dependencyIdentityJgradients/batch_normalization_2/FusedBatchNormV3_grad/FusedBatchNormGradV3G^gradients/batch_normalization_2/FusedBatchNormV3_grad/tuple/group_deps*
T0*]
_classS
QOloc:@gradients/batch_normalization_2/FusedBatchNormV3_grad/FusedBatchNormGradV3*&
 _has_manual_control_dependencies(*/
_output_shapes
:˙˙˙˙˙˙˙˙˙#
÷
Pgradients/batch_normalization_2/FusedBatchNormV3_grad/tuple/control_dependency_1IdentityLgradients/batch_normalization_2/FusedBatchNormV3_grad/FusedBatchNormGradV3:1G^gradients/batch_normalization_2/FusedBatchNormV3_grad/tuple/group_deps*
T0*]
_classS
QOloc:@gradients/batch_normalization_2/FusedBatchNormV3_grad/FusedBatchNormGradV3*
_output_shapes
:#
÷
Pgradients/batch_normalization_2/FusedBatchNormV3_grad/tuple/control_dependency_2IdentityLgradients/batch_normalization_2/FusedBatchNormV3_grad/FusedBatchNormGradV3:2G^gradients/batch_normalization_2/FusedBatchNormV3_grad/tuple/group_deps*
T0*]
_classS
QOloc:@gradients/batch_normalization_2/FusedBatchNormV3_grad/FusedBatchNormGradV3*
_output_shapes
:#
Ö
*gradients/conv_d2/BiasAdd_grad/BiasAddGradBiasAddGradNgradients/batch_normalization_2/FusedBatchNormV3_grad/tuple/control_dependency*
T0*&
 _has_manual_control_dependencies(*
_output_shapes
:#
Ý
/gradients/conv_d2/BiasAdd_grad/tuple/group_depsNoOpO^gradients/batch_normalization_2/FusedBatchNormV3_grad/tuple/control_dependency+^gradients/conv_d2/BiasAdd_grad/BiasAddGrad*&
 _has_manual_control_dependencies(
Ţ
7gradients/conv_d2/BiasAdd_grad/tuple/control_dependencyIdentityNgradients/batch_normalization_2/FusedBatchNormV3_grad/tuple/control_dependency0^gradients/conv_d2/BiasAdd_grad/tuple/group_deps*
T0*]
_classS
QOloc:@gradients/batch_normalization_2/FusedBatchNormV3_grad/FusedBatchNormGradV3*/
_output_shapes
:˙˙˙˙˙˙˙˙˙#

9gradients/conv_d2/BiasAdd_grad/tuple/control_dependency_1Identity*gradients/conv_d2/BiasAdd_grad/BiasAddGrad0^gradients/conv_d2/BiasAdd_grad/tuple/group_deps*
T0*=
_class3
1/loc:@gradients/conv_d2/BiasAdd_grad/BiasAddGrad*
_output_shapes
:#

$gradients/conv_d2/Conv2D_grad/ShapeNShapeNdropout_1/Identityconv_d2/Conv2D/ReadVariableOp*
N*
T0* 
_output_shapes
::
Ń
1gradients/conv_d2/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput$gradients/conv_d2/Conv2D_grad/ShapeNconv_d2/Conv2D/ReadVariableOp7gradients/conv_d2/BiasAdd_grad/tuple/control_dependency*
T0*&
 _has_manual_control_dependencies(*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
paddingSAME*
strides

Á
2gradients/conv_d2/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterdropout_1/Identity&gradients/conv_d2/Conv2D_grad/ShapeN:17gradients/conv_d2/BiasAdd_grad/tuple/control_dependency*
T0*&
 _has_manual_control_dependencies(*'
_output_shapes
:#*
paddingSAME*
strides

Ç
.gradients/conv_d2/Conv2D_grad/tuple/group_depsNoOp3^gradients/conv_d2/Conv2D_grad/Conv2DBackpropFilter2^gradients/conv_d2/Conv2D_grad/Conv2DBackpropInput*&
 _has_manual_control_dependencies(
§
6gradients/conv_d2/Conv2D_grad/tuple/control_dependencyIdentity1gradients/conv_d2/Conv2D_grad/Conv2DBackpropInput/^gradients/conv_d2/Conv2D_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/conv_d2/Conv2D_grad/Conv2DBackpropInput*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
˘
8gradients/conv_d2/Conv2D_grad/tuple/control_dependency_1Identity2gradients/conv_d2/Conv2D_grad/Conv2DBackpropFilter/^gradients/conv_d2/Conv2D_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients/conv_d2/Conv2D_grad/Conv2DBackpropFilter*'
_output_shapes
:#
Ľ
gradients/Relu_1_grad/ReluGradReluGrad6gradients/conv_d2/Conv2D_grad/tuple/control_dependencyRelu_1*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
s
gradients/zeros_like_6	ZerosLike(batch_normalization_1/FusedBatchNormV3:1*
T0*
_output_shapes	
:
s
gradients/zeros_like_7	ZerosLike(batch_normalization_1/FusedBatchNormV3:2*
T0*
_output_shapes	
:
s
gradients/zeros_like_8	ZerosLike(batch_normalization_1/FusedBatchNormV3:3*
T0*
_output_shapes	
:
s
gradients/zeros_like_9	ZerosLike(batch_normalization_1/FusedBatchNormV3:4*
T0*
_output_shapes	
:
q
gradients/zeros_like_10	ZerosLike(batch_normalization_1/FusedBatchNormV3:5*
T0*
_output_shapes
:
ú
Jgradients/batch_normalization_1/FusedBatchNormV3_grad/FusedBatchNormGradV3FusedBatchNormGradV3gradients/Relu_1_grad/ReluGradconv_d1/BiasAdd$batch_normalization_1/ReadVariableOp5batch_normalization_1/FusedBatchNormV3/ReadVariableOp7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1(batch_normalization_1/FusedBatchNormV3:5*
T0*
U0*&
 _has_manual_control_dependencies(*F
_output_shapes4
2:˙˙˙˙˙˙˙˙˙::: : *
epsilon%o:*
is_training( 
Ă
Fgradients/batch_normalization_1/FusedBatchNormV3_grad/tuple/group_depsNoOpK^gradients/batch_normalization_1/FusedBatchNormV3_grad/FusedBatchNormGradV3*&
 _has_manual_control_dependencies(
ą
Ngradients/batch_normalization_1/FusedBatchNormV3_grad/tuple/control_dependencyIdentityJgradients/batch_normalization_1/FusedBatchNormV3_grad/FusedBatchNormGradV3G^gradients/batch_normalization_1/FusedBatchNormV3_grad/tuple/group_deps*
T0*]
_classS
QOloc:@gradients/batch_normalization_1/FusedBatchNormV3_grad/FusedBatchNormGradV3*&
 _has_manual_control_dependencies(*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
ř
Pgradients/batch_normalization_1/FusedBatchNormV3_grad/tuple/control_dependency_1IdentityLgradients/batch_normalization_1/FusedBatchNormV3_grad/FusedBatchNormGradV3:1G^gradients/batch_normalization_1/FusedBatchNormV3_grad/tuple/group_deps*
T0*]
_classS
QOloc:@gradients/batch_normalization_1/FusedBatchNormV3_grad/FusedBatchNormGradV3*
_output_shapes	
:
ř
Pgradients/batch_normalization_1/FusedBatchNormV3_grad/tuple/control_dependency_2IdentityLgradients/batch_normalization_1/FusedBatchNormV3_grad/FusedBatchNormGradV3:2G^gradients/batch_normalization_1/FusedBatchNormV3_grad/tuple/group_deps*
T0*]
_classS
QOloc:@gradients/batch_normalization_1/FusedBatchNormV3_grad/FusedBatchNormGradV3*
_output_shapes	
:
×
*gradients/conv_d1/BiasAdd_grad/BiasAddGradBiasAddGradNgradients/batch_normalization_1/FusedBatchNormV3_grad/tuple/control_dependency*
T0*&
 _has_manual_control_dependencies(*
_output_shapes	
:
Ý
/gradients/conv_d1/BiasAdd_grad/tuple/group_depsNoOpO^gradients/batch_normalization_1/FusedBatchNormV3_grad/tuple/control_dependency+^gradients/conv_d1/BiasAdd_grad/BiasAddGrad*&
 _has_manual_control_dependencies(
ß
7gradients/conv_d1/BiasAdd_grad/tuple/control_dependencyIdentityNgradients/batch_normalization_1/FusedBatchNormV3_grad/tuple/control_dependency0^gradients/conv_d1/BiasAdd_grad/tuple/group_deps*
T0*]
_classS
QOloc:@gradients/batch_normalization_1/FusedBatchNormV3_grad/FusedBatchNormGradV3*0
_output_shapes
:˙˙˙˙˙˙˙˙˙

9gradients/conv_d1/BiasAdd_grad/tuple/control_dependency_1Identity*gradients/conv_d1/BiasAdd_grad/BiasAddGrad0^gradients/conv_d1/BiasAdd_grad/tuple/group_deps*
T0*=
_class3
1/loc:@gradients/conv_d1/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:

$gradients/conv_d1/Conv2D_grad/ShapeNShapeNdropout/Identityconv_d1/Conv2D/ReadVariableOp*
N*
T0* 
_output_shapes
::
Ń
1gradients/conv_d1/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput$gradients/conv_d1/Conv2D_grad/ShapeNconv_d1/Conv2D/ReadVariableOp7gradients/conv_d1/BiasAdd_grad/tuple/control_dependency*
T0*&
 _has_manual_control_dependencies(*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
paddingSAME*
strides

Ŕ
2gradients/conv_d1/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterdropout/Identity&gradients/conv_d1/Conv2D_grad/ShapeN:17gradients/conv_d1/BiasAdd_grad/tuple/control_dependency*
T0*&
 _has_manual_control_dependencies(*(
_output_shapes
:*
paddingSAME*
strides

Ç
.gradients/conv_d1/Conv2D_grad/tuple/group_depsNoOp3^gradients/conv_d1/Conv2D_grad/Conv2DBackpropFilter2^gradients/conv_d1/Conv2D_grad/Conv2DBackpropInput*&
 _has_manual_control_dependencies(
§
6gradients/conv_d1/Conv2D_grad/tuple/control_dependencyIdentity1gradients/conv_d1/Conv2D_grad/Conv2DBackpropInput/^gradients/conv_d1/Conv2D_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/conv_d1/Conv2D_grad/Conv2DBackpropInput*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ł
8gradients/conv_d1/Conv2D_grad/tuple/control_dependency_1Identity2gradients/conv_d1/Conv2D_grad/Conv2DBackpropFilter/^gradients/conv_d1/Conv2D_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients/conv_d1/Conv2D_grad/Conv2DBackpropFilter*(
_output_shapes
:
đ
$gradients/MaxPool_2_grad/MaxPoolGradMaxPoolGradsbb3/Relu_3	MaxPool_26gradients/conv_d1/Conv2D_grad/tuple/control_dependency*0
_output_shapes
:˙˙˙˙˙˙˙˙˙/*
ksize
*
paddingSAME*
strides

ĺ
gradients/AddN_4AddN&gradients/gc3/AvgPool_grad/AvgPoolGrad$gradients/MaxPool_2_grad/MaxPoolGrad*
N*
T0*9
_class/
-+loc:@gradients/gc3/AvgPool_grad/AvgPoolGrad*0
_output_shapes
:˙˙˙˙˙˙˙˙˙/

#gradients/sbb3/Relu_3_grad/ReluGradReluGradgradients/AddN_4sbb3/Relu_3*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙/
w
gradients/zeros_like_11	ZerosLike+sbb3/batch_normalization/FusedBatchNormV3:1*
T0*
_output_shapes	
:
w
gradients/zeros_like_12	ZerosLike+sbb3/batch_normalization/FusedBatchNormV3:2*
T0*
_output_shapes	
:
w
gradients/zeros_like_13	ZerosLike+sbb3/batch_normalization/FusedBatchNormV3:3*
T0*
_output_shapes	
:
w
gradients/zeros_like_14	ZerosLike+sbb3/batch_normalization/FusedBatchNormV3:4*
T0*
_output_shapes	
:
t
gradients/zeros_like_15	ZerosLike+sbb3/batch_normalization/FusedBatchNormV3:5*
T0*
_output_shapes
:

Mgradients/sbb3/batch_normalization/FusedBatchNormV3_grad/FusedBatchNormGradV3FusedBatchNormGradV3#gradients/sbb3/Relu_3_grad/ReluGradsbb3/conv4/BiasAdd'sbb3/batch_normalization/ReadVariableOp8sbb3/batch_normalization/FusedBatchNormV3/ReadVariableOp:sbb3/batch_normalization/FusedBatchNormV3/ReadVariableOp_1+sbb3/batch_normalization/FusedBatchNormV3:5*
T0*
U0*&
 _has_manual_control_dependencies(*F
_output_shapes4
2:˙˙˙˙˙˙˙˙˙/::: : *
epsilon%o:*
is_training( 
É
Igradients/sbb3/batch_normalization/FusedBatchNormV3_grad/tuple/group_depsNoOpN^gradients/sbb3/batch_normalization/FusedBatchNormV3_grad/FusedBatchNormGradV3*&
 _has_manual_control_dependencies(
˝
Qgradients/sbb3/batch_normalization/FusedBatchNormV3_grad/tuple/control_dependencyIdentityMgradients/sbb3/batch_normalization/FusedBatchNormV3_grad/FusedBatchNormGradV3J^gradients/sbb3/batch_normalization/FusedBatchNormV3_grad/tuple/group_deps*
T0*`
_classV
TRloc:@gradients/sbb3/batch_normalization/FusedBatchNormV3_grad/FusedBatchNormGradV3*&
 _has_manual_control_dependencies(*0
_output_shapes
:˙˙˙˙˙˙˙˙˙/

Sgradients/sbb3/batch_normalization/FusedBatchNormV3_grad/tuple/control_dependency_1IdentityOgradients/sbb3/batch_normalization/FusedBatchNormV3_grad/FusedBatchNormGradV3:1J^gradients/sbb3/batch_normalization/FusedBatchNormV3_grad/tuple/group_deps*
T0*`
_classV
TRloc:@gradients/sbb3/batch_normalization/FusedBatchNormV3_grad/FusedBatchNormGradV3*
_output_shapes	
:

Sgradients/sbb3/batch_normalization/FusedBatchNormV3_grad/tuple/control_dependency_2IdentityOgradients/sbb3/batch_normalization/FusedBatchNormV3_grad/FusedBatchNormGradV3:2J^gradients/sbb3/batch_normalization/FusedBatchNormV3_grad/tuple/group_deps*
T0*`
_classV
TRloc:@gradients/sbb3/batch_normalization/FusedBatchNormV3_grad/FusedBatchNormGradV3*
_output_shapes	
:
Ý
-gradients/sbb3/conv4/BiasAdd_grad/BiasAddGradBiasAddGradQgradients/sbb3/batch_normalization/FusedBatchNormV3_grad/tuple/control_dependency*
T0*&
 _has_manual_control_dependencies(*
_output_shapes	
:
ć
2gradients/sbb3/conv4/BiasAdd_grad/tuple/group_depsNoOpR^gradients/sbb3/batch_normalization/FusedBatchNormV3_grad/tuple/control_dependency.^gradients/sbb3/conv4/BiasAdd_grad/BiasAddGrad*&
 _has_manual_control_dependencies(
ë
:gradients/sbb3/conv4/BiasAdd_grad/tuple/control_dependencyIdentityQgradients/sbb3/batch_normalization/FusedBatchNormV3_grad/tuple/control_dependency3^gradients/sbb3/conv4/BiasAdd_grad/tuple/group_deps*
T0*`
_classV
TRloc:@gradients/sbb3/batch_normalization/FusedBatchNormV3_grad/FusedBatchNormGradV3*0
_output_shapes
:˙˙˙˙˙˙˙˙˙/

<gradients/sbb3/conv4/BiasAdd_grad/tuple/control_dependency_1Identity-gradients/sbb3/conv4/BiasAdd_grad/BiasAddGrad3^gradients/sbb3/conv4/BiasAdd_grad/tuple/group_deps*
T0*@
_class6
42loc:@gradients/sbb3/conv4/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:

'gradients/sbb3/conv4/Conv2D_grad/ShapeNShapeNsbb3/Relu_2 sbb3/conv4/Conv2D/ReadVariableOp*
N*
T0* 
_output_shapes
::
Ü
4gradients/sbb3/conv4/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput'gradients/sbb3/conv4/Conv2D_grad/ShapeN sbb3/conv4/Conv2D/ReadVariableOp:gradients/sbb3/conv4/BiasAdd_grad/tuple/control_dependency*
T0*&
 _has_manual_control_dependencies(*/
_output_shapes
:˙˙˙˙˙˙˙˙˙/@*
paddingSAME*
strides

Ă
5gradients/sbb3/conv4/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFiltersbb3/Relu_2)gradients/sbb3/conv4/Conv2D_grad/ShapeN:1:gradients/sbb3/conv4/BiasAdd_grad/tuple/control_dependency*
T0*&
 _has_manual_control_dependencies(*'
_output_shapes
:@*
paddingSAME*
strides

Đ
1gradients/sbb3/conv4/Conv2D_grad/tuple/group_depsNoOp6^gradients/sbb3/conv4/Conv2D_grad/Conv2DBackpropFilter5^gradients/sbb3/conv4/Conv2D_grad/Conv2DBackpropInput*&
 _has_manual_control_dependencies(
˛
9gradients/sbb3/conv4/Conv2D_grad/tuple/control_dependencyIdentity4gradients/sbb3/conv4/Conv2D_grad/Conv2DBackpropInput2^gradients/sbb3/conv4/Conv2D_grad/tuple/group_deps*
T0*G
_class=
;9loc:@gradients/sbb3/conv4/Conv2D_grad/Conv2DBackpropInput*/
_output_shapes
:˙˙˙˙˙˙˙˙˙/@
Ž
;gradients/sbb3/conv4/Conv2D_grad/tuple/control_dependency_1Identity5gradients/sbb3/conv4/Conv2D_grad/Conv2DBackpropFilter2^gradients/sbb3/conv4/Conv2D_grad/tuple/group_deps*
T0*H
_class>
<:loc:@gradients/sbb3/conv4/Conv2D_grad/Conv2DBackpropFilter*'
_output_shapes
:@
Ů
#gradients/sbb3/Relu_2_grad/ReluGradReluGrad9gradients/sbb3/conv4/Conv2D_grad/tuple/control_dependencysbb3/Relu_2*
T0*&
 _has_manual_control_dependencies(*/
_output_shapes
:˙˙˙˙˙˙˙˙˙/@
Ž
-gradients/sbb3/conv3/BiasAdd_grad/BiasAddGradBiasAddGrad#gradients/sbb3/Relu_2_grad/ReluGrad*
T0*&
 _has_manual_control_dependencies(*
_output_shapes
:@
¸
2gradients/sbb3/conv3/BiasAdd_grad/tuple/group_depsNoOp$^gradients/sbb3/Relu_2_grad/ReluGrad.^gradients/sbb3/conv3/BiasAdd_grad/BiasAddGrad*&
 _has_manual_control_dependencies(

:gradients/sbb3/conv3/BiasAdd_grad/tuple/control_dependencyIdentity#gradients/sbb3/Relu_2_grad/ReluGrad3^gradients/sbb3/conv3/BiasAdd_grad/tuple/group_deps*
T0*6
_class,
*(loc:@gradients/sbb3/Relu_2_grad/ReluGrad*/
_output_shapes
:˙˙˙˙˙˙˙˙˙/@

<gradients/sbb3/conv3/BiasAdd_grad/tuple/control_dependency_1Identity-gradients/sbb3/conv3/BiasAdd_grad/BiasAddGrad3^gradients/sbb3/conv3/BiasAdd_grad/tuple/group_deps*
T0*@
_class6
42loc:@gradients/sbb3/conv3/BiasAdd_grad/BiasAddGrad*
_output_shapes
:@

'gradients/sbb3/conv3/Conv2D_grad/ShapeNShapeNsbb3/Relu_1 sbb3/conv3/Conv2D/ReadVariableOp*
N*
T0* 
_output_shapes
::
Ü
4gradients/sbb3/conv3/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput'gradients/sbb3/conv3/Conv2D_grad/ShapeN sbb3/conv3/Conv2D/ReadVariableOp:gradients/sbb3/conv3/BiasAdd_grad/tuple/control_dependency*
T0*&
 _has_manual_control_dependencies(*/
_output_shapes
:˙˙˙˙˙˙˙˙˙/@*
paddingSAME*
strides

Â
5gradients/sbb3/conv3/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFiltersbb3/Relu_1)gradients/sbb3/conv3/Conv2D_grad/ShapeN:1:gradients/sbb3/conv3/BiasAdd_grad/tuple/control_dependency*
T0*&
 _has_manual_control_dependencies(*&
_output_shapes
:@@*
paddingSAME*
strides

Đ
1gradients/sbb3/conv3/Conv2D_grad/tuple/group_depsNoOp6^gradients/sbb3/conv3/Conv2D_grad/Conv2DBackpropFilter5^gradients/sbb3/conv3/Conv2D_grad/Conv2DBackpropInput*&
 _has_manual_control_dependencies(
˛
9gradients/sbb3/conv3/Conv2D_grad/tuple/control_dependencyIdentity4gradients/sbb3/conv3/Conv2D_grad/Conv2DBackpropInput2^gradients/sbb3/conv3/Conv2D_grad/tuple/group_deps*
T0*G
_class=
;9loc:@gradients/sbb3/conv3/Conv2D_grad/Conv2DBackpropInput*/
_output_shapes
:˙˙˙˙˙˙˙˙˙/@
­
;gradients/sbb3/conv3/Conv2D_grad/tuple/control_dependency_1Identity5gradients/sbb3/conv3/Conv2D_grad/Conv2DBackpropFilter2^gradients/sbb3/conv3/Conv2D_grad/tuple/group_deps*
T0*H
_class>
<:loc:@gradients/sbb3/conv3/Conv2D_grad/Conv2DBackpropFilter*&
_output_shapes
:@@
Ů
#gradients/sbb3/Relu_1_grad/ReluGradReluGrad9gradients/sbb3/conv3/Conv2D_grad/tuple/control_dependencysbb3/Relu_1*
T0*&
 _has_manual_control_dependencies(*/
_output_shapes
:˙˙˙˙˙˙˙˙˙/@
Ž
-gradients/sbb3/conv2/BiasAdd_grad/BiasAddGradBiasAddGrad#gradients/sbb3/Relu_1_grad/ReluGrad*
T0*&
 _has_manual_control_dependencies(*
_output_shapes
:@
¸
2gradients/sbb3/conv2/BiasAdd_grad/tuple/group_depsNoOp$^gradients/sbb3/Relu_1_grad/ReluGrad.^gradients/sbb3/conv2/BiasAdd_grad/BiasAddGrad*&
 _has_manual_control_dependencies(

:gradients/sbb3/conv2/BiasAdd_grad/tuple/control_dependencyIdentity#gradients/sbb3/Relu_1_grad/ReluGrad3^gradients/sbb3/conv2/BiasAdd_grad/tuple/group_deps*
T0*6
_class,
*(loc:@gradients/sbb3/Relu_1_grad/ReluGrad*/
_output_shapes
:˙˙˙˙˙˙˙˙˙/@

<gradients/sbb3/conv2/BiasAdd_grad/tuple/control_dependency_1Identity-gradients/sbb3/conv2/BiasAdd_grad/BiasAddGrad3^gradients/sbb3/conv2/BiasAdd_grad/tuple/group_deps*
T0*@
_class6
42loc:@gradients/sbb3/conv2/BiasAdd_grad/BiasAddGrad*
_output_shapes
:@

'gradients/sbb3/conv2/Conv2D_grad/ShapeNShapeN	sbb3/Relu sbb3/conv2/Conv2D/ReadVariableOp*
N*
T0* 
_output_shapes
::
Ü
4gradients/sbb3/conv2/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput'gradients/sbb3/conv2/Conv2D_grad/ShapeN sbb3/conv2/Conv2D/ReadVariableOp:gradients/sbb3/conv2/BiasAdd_grad/tuple/control_dependency*
T0*&
 _has_manual_control_dependencies(*/
_output_shapes
:˙˙˙˙˙˙˙˙˙/@*
paddingSAME*
strides

Ŕ
5gradients/sbb3/conv2/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilter	sbb3/Relu)gradients/sbb3/conv2/Conv2D_grad/ShapeN:1:gradients/sbb3/conv2/BiasAdd_grad/tuple/control_dependency*
T0*&
 _has_manual_control_dependencies(*&
_output_shapes
:@@*
paddingSAME*
strides

Đ
1gradients/sbb3/conv2/Conv2D_grad/tuple/group_depsNoOp6^gradients/sbb3/conv2/Conv2D_grad/Conv2DBackpropFilter5^gradients/sbb3/conv2/Conv2D_grad/Conv2DBackpropInput*&
 _has_manual_control_dependencies(
˛
9gradients/sbb3/conv2/Conv2D_grad/tuple/control_dependencyIdentity4gradients/sbb3/conv2/Conv2D_grad/Conv2DBackpropInput2^gradients/sbb3/conv2/Conv2D_grad/tuple/group_deps*
T0*G
_class=
;9loc:@gradients/sbb3/conv2/Conv2D_grad/Conv2DBackpropInput*/
_output_shapes
:˙˙˙˙˙˙˙˙˙/@
­
;gradients/sbb3/conv2/Conv2D_grad/tuple/control_dependency_1Identity5gradients/sbb3/conv2/Conv2D_grad/Conv2DBackpropFilter2^gradients/sbb3/conv2/Conv2D_grad/tuple/group_deps*
T0*H
_class>
<:loc:@gradients/sbb3/conv2/Conv2D_grad/Conv2DBackpropFilter*&
_output_shapes
:@@
Ő
!gradients/sbb3/Relu_grad/ReluGradReluGrad9gradients/sbb3/conv2/Conv2D_grad/tuple/control_dependency	sbb3/Relu*
T0*&
 _has_manual_control_dependencies(*/
_output_shapes
:˙˙˙˙˙˙˙˙˙/@
Ź
-gradients/sbb3/conv1/BiasAdd_grad/BiasAddGradBiasAddGrad!gradients/sbb3/Relu_grad/ReluGrad*
T0*&
 _has_manual_control_dependencies(*
_output_shapes
:@
ś
2gradients/sbb3/conv1/BiasAdd_grad/tuple/group_depsNoOp"^gradients/sbb3/Relu_grad/ReluGrad.^gradients/sbb3/conv1/BiasAdd_grad/BiasAddGrad*&
 _has_manual_control_dependencies(

:gradients/sbb3/conv1/BiasAdd_grad/tuple/control_dependencyIdentity!gradients/sbb3/Relu_grad/ReluGrad3^gradients/sbb3/conv1/BiasAdd_grad/tuple/group_deps*
T0*4
_class*
(&loc:@gradients/sbb3/Relu_grad/ReluGrad*/
_output_shapes
:˙˙˙˙˙˙˙˙˙/@

<gradients/sbb3/conv1/BiasAdd_grad/tuple/control_dependency_1Identity-gradients/sbb3/conv1/BiasAdd_grad/BiasAddGrad3^gradients/sbb3/conv1/BiasAdd_grad/tuple/group_deps*
T0*@
_class6
42loc:@gradients/sbb3/conv1/BiasAdd_grad/BiasAddGrad*
_output_shapes
:@

'gradients/sbb3/conv1/Conv2D_grad/ShapeNShapeNsbb2/Relu_3 sbb3/conv1/Conv2D/ReadVariableOp*
N*
T0* 
_output_shapes
::
Ý
4gradients/sbb3/conv1/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput'gradients/sbb3/conv1/Conv2D_grad/ShapeN sbb3/conv1/Conv2D/ReadVariableOp:gradients/sbb3/conv1/BiasAdd_grad/tuple/control_dependency*
T0*&
 _has_manual_control_dependencies(*0
_output_shapes
:˙˙˙˙˙˙˙˙˙/*
paddingSAME*
strides

Ă
5gradients/sbb3/conv1/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFiltersbb2/Relu_3)gradients/sbb3/conv1/Conv2D_grad/ShapeN:1:gradients/sbb3/conv1/BiasAdd_grad/tuple/control_dependency*
T0*&
 _has_manual_control_dependencies(*'
_output_shapes
:@*
paddingSAME*
strides

Đ
1gradients/sbb3/conv1/Conv2D_grad/tuple/group_depsNoOp6^gradients/sbb3/conv1/Conv2D_grad/Conv2DBackpropFilter5^gradients/sbb3/conv1/Conv2D_grad/Conv2DBackpropInput*&
 _has_manual_control_dependencies(
ł
9gradients/sbb3/conv1/Conv2D_grad/tuple/control_dependencyIdentity4gradients/sbb3/conv1/Conv2D_grad/Conv2DBackpropInput2^gradients/sbb3/conv1/Conv2D_grad/tuple/group_deps*
T0*G
_class=
;9loc:@gradients/sbb3/conv1/Conv2D_grad/Conv2DBackpropInput*0
_output_shapes
:˙˙˙˙˙˙˙˙˙/
Ž
;gradients/sbb3/conv1/Conv2D_grad/tuple/control_dependency_1Identity5gradients/sbb3/conv1/Conv2D_grad/Conv2DBackpropFilter2^gradients/sbb3/conv1/Conv2D_grad/tuple/group_deps*
T0*H
_class>
<:loc:@gradients/sbb3/conv1/Conv2D_grad/Conv2DBackpropFilter*'
_output_shapes
:@
˛
#gradients/sbb2/Relu_3_grad/ReluGradReluGrad9gradients/sbb3/conv1/Conv2D_grad/tuple/control_dependencysbb2/Relu_3*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙/
w
gradients/zeros_like_16	ZerosLike+sbb2/batch_normalization/FusedBatchNormV3:1*
T0*
_output_shapes	
:
w
gradients/zeros_like_17	ZerosLike+sbb2/batch_normalization/FusedBatchNormV3:2*
T0*
_output_shapes	
:
w
gradients/zeros_like_18	ZerosLike+sbb2/batch_normalization/FusedBatchNormV3:3*
T0*
_output_shapes	
:
w
gradients/zeros_like_19	ZerosLike+sbb2/batch_normalization/FusedBatchNormV3:4*
T0*
_output_shapes	
:
t
gradients/zeros_like_20	ZerosLike+sbb2/batch_normalization/FusedBatchNormV3:5*
T0*
_output_shapes
:

Mgradients/sbb2/batch_normalization/FusedBatchNormV3_grad/FusedBatchNormGradV3FusedBatchNormGradV3#gradients/sbb2/Relu_3_grad/ReluGradsbb2/conv4/BiasAdd'sbb2/batch_normalization/ReadVariableOp8sbb2/batch_normalization/FusedBatchNormV3/ReadVariableOp:sbb2/batch_normalization/FusedBatchNormV3/ReadVariableOp_1+sbb2/batch_normalization/FusedBatchNormV3:5*
T0*
U0*&
 _has_manual_control_dependencies(*F
_output_shapes4
2:˙˙˙˙˙˙˙˙˙/::: : *
epsilon%o:*
is_training( 
É
Igradients/sbb2/batch_normalization/FusedBatchNormV3_grad/tuple/group_depsNoOpN^gradients/sbb2/batch_normalization/FusedBatchNormV3_grad/FusedBatchNormGradV3*&
 _has_manual_control_dependencies(
˝
Qgradients/sbb2/batch_normalization/FusedBatchNormV3_grad/tuple/control_dependencyIdentityMgradients/sbb2/batch_normalization/FusedBatchNormV3_grad/FusedBatchNormGradV3J^gradients/sbb2/batch_normalization/FusedBatchNormV3_grad/tuple/group_deps*
T0*`
_classV
TRloc:@gradients/sbb2/batch_normalization/FusedBatchNormV3_grad/FusedBatchNormGradV3*&
 _has_manual_control_dependencies(*0
_output_shapes
:˙˙˙˙˙˙˙˙˙/

Sgradients/sbb2/batch_normalization/FusedBatchNormV3_grad/tuple/control_dependency_1IdentityOgradients/sbb2/batch_normalization/FusedBatchNormV3_grad/FusedBatchNormGradV3:1J^gradients/sbb2/batch_normalization/FusedBatchNormV3_grad/tuple/group_deps*
T0*`
_classV
TRloc:@gradients/sbb2/batch_normalization/FusedBatchNormV3_grad/FusedBatchNormGradV3*
_output_shapes	
:

Sgradients/sbb2/batch_normalization/FusedBatchNormV3_grad/tuple/control_dependency_2IdentityOgradients/sbb2/batch_normalization/FusedBatchNormV3_grad/FusedBatchNormGradV3:2J^gradients/sbb2/batch_normalization/FusedBatchNormV3_grad/tuple/group_deps*
T0*`
_classV
TRloc:@gradients/sbb2/batch_normalization/FusedBatchNormV3_grad/FusedBatchNormGradV3*
_output_shapes	
:
Ý
-gradients/sbb2/conv4/BiasAdd_grad/BiasAddGradBiasAddGradQgradients/sbb2/batch_normalization/FusedBatchNormV3_grad/tuple/control_dependency*
T0*&
 _has_manual_control_dependencies(*
_output_shapes	
:
ć
2gradients/sbb2/conv4/BiasAdd_grad/tuple/group_depsNoOpR^gradients/sbb2/batch_normalization/FusedBatchNormV3_grad/tuple/control_dependency.^gradients/sbb2/conv4/BiasAdd_grad/BiasAddGrad*&
 _has_manual_control_dependencies(
ë
:gradients/sbb2/conv4/BiasAdd_grad/tuple/control_dependencyIdentityQgradients/sbb2/batch_normalization/FusedBatchNormV3_grad/tuple/control_dependency3^gradients/sbb2/conv4/BiasAdd_grad/tuple/group_deps*
T0*`
_classV
TRloc:@gradients/sbb2/batch_normalization/FusedBatchNormV3_grad/FusedBatchNormGradV3*0
_output_shapes
:˙˙˙˙˙˙˙˙˙/

<gradients/sbb2/conv4/BiasAdd_grad/tuple/control_dependency_1Identity-gradients/sbb2/conv4/BiasAdd_grad/BiasAddGrad3^gradients/sbb2/conv4/BiasAdd_grad/tuple/group_deps*
T0*@
_class6
42loc:@gradients/sbb2/conv4/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:

'gradients/sbb2/conv4/Conv2D_grad/ShapeNShapeNsbb2/Relu_2 sbb2/conv4/Conv2D/ReadVariableOp*
N*
T0* 
_output_shapes
::
Ü
4gradients/sbb2/conv4/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput'gradients/sbb2/conv4/Conv2D_grad/ShapeN sbb2/conv4/Conv2D/ReadVariableOp:gradients/sbb2/conv4/BiasAdd_grad/tuple/control_dependency*
T0*&
 _has_manual_control_dependencies(*/
_output_shapes
:˙˙˙˙˙˙˙˙˙/@*
paddingSAME*
strides

Ă
5gradients/sbb2/conv4/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFiltersbb2/Relu_2)gradients/sbb2/conv4/Conv2D_grad/ShapeN:1:gradients/sbb2/conv4/BiasAdd_grad/tuple/control_dependency*
T0*&
 _has_manual_control_dependencies(*'
_output_shapes
:@*
paddingSAME*
strides

Đ
1gradients/sbb2/conv4/Conv2D_grad/tuple/group_depsNoOp6^gradients/sbb2/conv4/Conv2D_grad/Conv2DBackpropFilter5^gradients/sbb2/conv4/Conv2D_grad/Conv2DBackpropInput*&
 _has_manual_control_dependencies(
˛
9gradients/sbb2/conv4/Conv2D_grad/tuple/control_dependencyIdentity4gradients/sbb2/conv4/Conv2D_grad/Conv2DBackpropInput2^gradients/sbb2/conv4/Conv2D_grad/tuple/group_deps*
T0*G
_class=
;9loc:@gradients/sbb2/conv4/Conv2D_grad/Conv2DBackpropInput*/
_output_shapes
:˙˙˙˙˙˙˙˙˙/@
Ž
;gradients/sbb2/conv4/Conv2D_grad/tuple/control_dependency_1Identity5gradients/sbb2/conv4/Conv2D_grad/Conv2DBackpropFilter2^gradients/sbb2/conv4/Conv2D_grad/tuple/group_deps*
T0*H
_class>
<:loc:@gradients/sbb2/conv4/Conv2D_grad/Conv2DBackpropFilter*'
_output_shapes
:@
Ů
#gradients/sbb2/Relu_2_grad/ReluGradReluGrad9gradients/sbb2/conv4/Conv2D_grad/tuple/control_dependencysbb2/Relu_2*
T0*&
 _has_manual_control_dependencies(*/
_output_shapes
:˙˙˙˙˙˙˙˙˙/@
Ž
-gradients/sbb2/conv3/BiasAdd_grad/BiasAddGradBiasAddGrad#gradients/sbb2/Relu_2_grad/ReluGrad*
T0*&
 _has_manual_control_dependencies(*
_output_shapes
:@
¸
2gradients/sbb2/conv3/BiasAdd_grad/tuple/group_depsNoOp$^gradients/sbb2/Relu_2_grad/ReluGrad.^gradients/sbb2/conv3/BiasAdd_grad/BiasAddGrad*&
 _has_manual_control_dependencies(

:gradients/sbb2/conv3/BiasAdd_grad/tuple/control_dependencyIdentity#gradients/sbb2/Relu_2_grad/ReluGrad3^gradients/sbb2/conv3/BiasAdd_grad/tuple/group_deps*
T0*6
_class,
*(loc:@gradients/sbb2/Relu_2_grad/ReluGrad*/
_output_shapes
:˙˙˙˙˙˙˙˙˙/@

<gradients/sbb2/conv3/BiasAdd_grad/tuple/control_dependency_1Identity-gradients/sbb2/conv3/BiasAdd_grad/BiasAddGrad3^gradients/sbb2/conv3/BiasAdd_grad/tuple/group_deps*
T0*@
_class6
42loc:@gradients/sbb2/conv3/BiasAdd_grad/BiasAddGrad*
_output_shapes
:@

'gradients/sbb2/conv3/Conv2D_grad/ShapeNShapeNsbb2/Relu_1 sbb2/conv3/Conv2D/ReadVariableOp*
N*
T0* 
_output_shapes
::
Ü
4gradients/sbb2/conv3/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput'gradients/sbb2/conv3/Conv2D_grad/ShapeN sbb2/conv3/Conv2D/ReadVariableOp:gradients/sbb2/conv3/BiasAdd_grad/tuple/control_dependency*
T0*&
 _has_manual_control_dependencies(*/
_output_shapes
:˙˙˙˙˙˙˙˙˙/@*
paddingSAME*
strides

Â
5gradients/sbb2/conv3/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFiltersbb2/Relu_1)gradients/sbb2/conv3/Conv2D_grad/ShapeN:1:gradients/sbb2/conv3/BiasAdd_grad/tuple/control_dependency*
T0*&
 _has_manual_control_dependencies(*&
_output_shapes
:@@*
paddingSAME*
strides

Đ
1gradients/sbb2/conv3/Conv2D_grad/tuple/group_depsNoOp6^gradients/sbb2/conv3/Conv2D_grad/Conv2DBackpropFilter5^gradients/sbb2/conv3/Conv2D_grad/Conv2DBackpropInput*&
 _has_manual_control_dependencies(
˛
9gradients/sbb2/conv3/Conv2D_grad/tuple/control_dependencyIdentity4gradients/sbb2/conv3/Conv2D_grad/Conv2DBackpropInput2^gradients/sbb2/conv3/Conv2D_grad/tuple/group_deps*
T0*G
_class=
;9loc:@gradients/sbb2/conv3/Conv2D_grad/Conv2DBackpropInput*/
_output_shapes
:˙˙˙˙˙˙˙˙˙/@
­
;gradients/sbb2/conv3/Conv2D_grad/tuple/control_dependency_1Identity5gradients/sbb2/conv3/Conv2D_grad/Conv2DBackpropFilter2^gradients/sbb2/conv3/Conv2D_grad/tuple/group_deps*
T0*H
_class>
<:loc:@gradients/sbb2/conv3/Conv2D_grad/Conv2DBackpropFilter*&
_output_shapes
:@@
Ů
#gradients/sbb2/Relu_1_grad/ReluGradReluGrad9gradients/sbb2/conv3/Conv2D_grad/tuple/control_dependencysbb2/Relu_1*
T0*&
 _has_manual_control_dependencies(*/
_output_shapes
:˙˙˙˙˙˙˙˙˙/@
Ž
-gradients/sbb2/conv2/BiasAdd_grad/BiasAddGradBiasAddGrad#gradients/sbb2/Relu_1_grad/ReluGrad*
T0*&
 _has_manual_control_dependencies(*
_output_shapes
:@
¸
2gradients/sbb2/conv2/BiasAdd_grad/tuple/group_depsNoOp$^gradients/sbb2/Relu_1_grad/ReluGrad.^gradients/sbb2/conv2/BiasAdd_grad/BiasAddGrad*&
 _has_manual_control_dependencies(

:gradients/sbb2/conv2/BiasAdd_grad/tuple/control_dependencyIdentity#gradients/sbb2/Relu_1_grad/ReluGrad3^gradients/sbb2/conv2/BiasAdd_grad/tuple/group_deps*
T0*6
_class,
*(loc:@gradients/sbb2/Relu_1_grad/ReluGrad*/
_output_shapes
:˙˙˙˙˙˙˙˙˙/@

<gradients/sbb2/conv2/BiasAdd_grad/tuple/control_dependency_1Identity-gradients/sbb2/conv2/BiasAdd_grad/BiasAddGrad3^gradients/sbb2/conv2/BiasAdd_grad/tuple/group_deps*
T0*@
_class6
42loc:@gradients/sbb2/conv2/BiasAdd_grad/BiasAddGrad*
_output_shapes
:@

'gradients/sbb2/conv2/Conv2D_grad/ShapeNShapeN	sbb2/Relu sbb2/conv2/Conv2D/ReadVariableOp*
N*
T0* 
_output_shapes
::
Ü
4gradients/sbb2/conv2/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput'gradients/sbb2/conv2/Conv2D_grad/ShapeN sbb2/conv2/Conv2D/ReadVariableOp:gradients/sbb2/conv2/BiasAdd_grad/tuple/control_dependency*
T0*&
 _has_manual_control_dependencies(*/
_output_shapes
:˙˙˙˙˙˙˙˙˙/@*
paddingSAME*
strides

Ŕ
5gradients/sbb2/conv2/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilter	sbb2/Relu)gradients/sbb2/conv2/Conv2D_grad/ShapeN:1:gradients/sbb2/conv2/BiasAdd_grad/tuple/control_dependency*
T0*&
 _has_manual_control_dependencies(*&
_output_shapes
:@@*
paddingSAME*
strides

Đ
1gradients/sbb2/conv2/Conv2D_grad/tuple/group_depsNoOp6^gradients/sbb2/conv2/Conv2D_grad/Conv2DBackpropFilter5^gradients/sbb2/conv2/Conv2D_grad/Conv2DBackpropInput*&
 _has_manual_control_dependencies(
˛
9gradients/sbb2/conv2/Conv2D_grad/tuple/control_dependencyIdentity4gradients/sbb2/conv2/Conv2D_grad/Conv2DBackpropInput2^gradients/sbb2/conv2/Conv2D_grad/tuple/group_deps*
T0*G
_class=
;9loc:@gradients/sbb2/conv2/Conv2D_grad/Conv2DBackpropInput*/
_output_shapes
:˙˙˙˙˙˙˙˙˙/@
­
;gradients/sbb2/conv2/Conv2D_grad/tuple/control_dependency_1Identity5gradients/sbb2/conv2/Conv2D_grad/Conv2DBackpropFilter2^gradients/sbb2/conv2/Conv2D_grad/tuple/group_deps*
T0*H
_class>
<:loc:@gradients/sbb2/conv2/Conv2D_grad/Conv2DBackpropFilter*&
_output_shapes
:@@
Ő
!gradients/sbb2/Relu_grad/ReluGradReluGrad9gradients/sbb2/conv2/Conv2D_grad/tuple/control_dependency	sbb2/Relu*
T0*&
 _has_manual_control_dependencies(*/
_output_shapes
:˙˙˙˙˙˙˙˙˙/@
Ź
-gradients/sbb2/conv1/BiasAdd_grad/BiasAddGradBiasAddGrad!gradients/sbb2/Relu_grad/ReluGrad*
T0*&
 _has_manual_control_dependencies(*
_output_shapes
:@
ś
2gradients/sbb2/conv1/BiasAdd_grad/tuple/group_depsNoOp"^gradients/sbb2/Relu_grad/ReluGrad.^gradients/sbb2/conv1/BiasAdd_grad/BiasAddGrad*&
 _has_manual_control_dependencies(

:gradients/sbb2/conv1/BiasAdd_grad/tuple/control_dependencyIdentity!gradients/sbb2/Relu_grad/ReluGrad3^gradients/sbb2/conv1/BiasAdd_grad/tuple/group_deps*
T0*4
_class*
(&loc:@gradients/sbb2/Relu_grad/ReluGrad*/
_output_shapes
:˙˙˙˙˙˙˙˙˙/@

<gradients/sbb2/conv1/BiasAdd_grad/tuple/control_dependency_1Identity-gradients/sbb2/conv1/BiasAdd_grad/BiasAddGrad3^gradients/sbb2/conv1/BiasAdd_grad/tuple/group_deps*
T0*@
_class6
42loc:@gradients/sbb2/conv1/BiasAdd_grad/BiasAddGrad*
_output_shapes
:@

'gradients/sbb2/conv1/Conv2D_grad/ShapeNShapeN	MaxPool_1 sbb2/conv1/Conv2D/ReadVariableOp*
N*
T0* 
_output_shapes
::
Ý
4gradients/sbb2/conv1/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput'gradients/sbb2/conv1/Conv2D_grad/ShapeN sbb2/conv1/Conv2D/ReadVariableOp:gradients/sbb2/conv1/BiasAdd_grad/tuple/control_dependency*
T0*&
 _has_manual_control_dependencies(*0
_output_shapes
:˙˙˙˙˙˙˙˙˙/*
paddingSAME*
strides

Á
5gradients/sbb2/conv1/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilter	MaxPool_1)gradients/sbb2/conv1/Conv2D_grad/ShapeN:1:gradients/sbb2/conv1/BiasAdd_grad/tuple/control_dependency*
T0*&
 _has_manual_control_dependencies(*'
_output_shapes
:@*
paddingSAME*
strides

Đ
1gradients/sbb2/conv1/Conv2D_grad/tuple/group_depsNoOp6^gradients/sbb2/conv1/Conv2D_grad/Conv2DBackpropFilter5^gradients/sbb2/conv1/Conv2D_grad/Conv2DBackpropInput*&
 _has_manual_control_dependencies(
ł
9gradients/sbb2/conv1/Conv2D_grad/tuple/control_dependencyIdentity4gradients/sbb2/conv1/Conv2D_grad/Conv2DBackpropInput2^gradients/sbb2/conv1/Conv2D_grad/tuple/group_deps*
T0*G
_class=
;9loc:@gradients/sbb2/conv1/Conv2D_grad/Conv2DBackpropInput*0
_output_shapes
:˙˙˙˙˙˙˙˙˙/
Ž
;gradients/sbb2/conv1/Conv2D_grad/tuple/control_dependency_1Identity5gradients/sbb2/conv1/Conv2D_grad/Conv2DBackpropFilter2^gradients/sbb2/conv1/Conv2D_grad/tuple/group_deps*
T0*H
_class>
<:loc:@gradients/sbb2/conv1/Conv2D_grad/Conv2DBackpropFilter*'
_output_shapes
:@
ó
$gradients/MaxPool_1_grad/MaxPoolGradMaxPoolGradsbb1/Relu_3	MaxPool_19gradients/sbb2/conv1/Conv2D_grad/tuple/control_dependency*0
_output_shapes
:˙˙˙˙˙˙˙˙˙^*
ksize
*
paddingSAME*
strides

ĺ
gradients/AddN_5AddN&gradients/gc2/AvgPool_grad/AvgPoolGrad$gradients/MaxPool_1_grad/MaxPoolGrad*
N*
T0*9
_class/
-+loc:@gradients/gc2/AvgPool_grad/AvgPoolGrad*0
_output_shapes
:˙˙˙˙˙˙˙˙˙^

#gradients/sbb1/Relu_3_grad/ReluGradReluGradgradients/AddN_5sbb1/Relu_3*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙^
w
gradients/zeros_like_21	ZerosLike+sbb1/batch_normalization/FusedBatchNormV3:1*
T0*
_output_shapes	
:
w
gradients/zeros_like_22	ZerosLike+sbb1/batch_normalization/FusedBatchNormV3:2*
T0*
_output_shapes	
:
w
gradients/zeros_like_23	ZerosLike+sbb1/batch_normalization/FusedBatchNormV3:3*
T0*
_output_shapes	
:
w
gradients/zeros_like_24	ZerosLike+sbb1/batch_normalization/FusedBatchNormV3:4*
T0*
_output_shapes	
:
t
gradients/zeros_like_25	ZerosLike+sbb1/batch_normalization/FusedBatchNormV3:5*
T0*
_output_shapes
:

Mgradients/sbb1/batch_normalization/FusedBatchNormV3_grad/FusedBatchNormGradV3FusedBatchNormGradV3#gradients/sbb1/Relu_3_grad/ReluGradsbb1/conv4/BiasAdd'sbb1/batch_normalization/ReadVariableOp8sbb1/batch_normalization/FusedBatchNormV3/ReadVariableOp:sbb1/batch_normalization/FusedBatchNormV3/ReadVariableOp_1+sbb1/batch_normalization/FusedBatchNormV3:5*
T0*
U0*&
 _has_manual_control_dependencies(*F
_output_shapes4
2:˙˙˙˙˙˙˙˙˙^::: : *
epsilon%o:*
is_training( 
É
Igradients/sbb1/batch_normalization/FusedBatchNormV3_grad/tuple/group_depsNoOpN^gradients/sbb1/batch_normalization/FusedBatchNormV3_grad/FusedBatchNormGradV3*&
 _has_manual_control_dependencies(
˝
Qgradients/sbb1/batch_normalization/FusedBatchNormV3_grad/tuple/control_dependencyIdentityMgradients/sbb1/batch_normalization/FusedBatchNormV3_grad/FusedBatchNormGradV3J^gradients/sbb1/batch_normalization/FusedBatchNormV3_grad/tuple/group_deps*
T0*`
_classV
TRloc:@gradients/sbb1/batch_normalization/FusedBatchNormV3_grad/FusedBatchNormGradV3*&
 _has_manual_control_dependencies(*0
_output_shapes
:˙˙˙˙˙˙˙˙˙^

Sgradients/sbb1/batch_normalization/FusedBatchNormV3_grad/tuple/control_dependency_1IdentityOgradients/sbb1/batch_normalization/FusedBatchNormV3_grad/FusedBatchNormGradV3:1J^gradients/sbb1/batch_normalization/FusedBatchNormV3_grad/tuple/group_deps*
T0*`
_classV
TRloc:@gradients/sbb1/batch_normalization/FusedBatchNormV3_grad/FusedBatchNormGradV3*
_output_shapes	
:

Sgradients/sbb1/batch_normalization/FusedBatchNormV3_grad/tuple/control_dependency_2IdentityOgradients/sbb1/batch_normalization/FusedBatchNormV3_grad/FusedBatchNormGradV3:2J^gradients/sbb1/batch_normalization/FusedBatchNormV3_grad/tuple/group_deps*
T0*`
_classV
TRloc:@gradients/sbb1/batch_normalization/FusedBatchNormV3_grad/FusedBatchNormGradV3*
_output_shapes	
:
Ý
-gradients/sbb1/conv4/BiasAdd_grad/BiasAddGradBiasAddGradQgradients/sbb1/batch_normalization/FusedBatchNormV3_grad/tuple/control_dependency*
T0*&
 _has_manual_control_dependencies(*
_output_shapes	
:
ć
2gradients/sbb1/conv4/BiasAdd_grad/tuple/group_depsNoOpR^gradients/sbb1/batch_normalization/FusedBatchNormV3_grad/tuple/control_dependency.^gradients/sbb1/conv4/BiasAdd_grad/BiasAddGrad*&
 _has_manual_control_dependencies(
ë
:gradients/sbb1/conv4/BiasAdd_grad/tuple/control_dependencyIdentityQgradients/sbb1/batch_normalization/FusedBatchNormV3_grad/tuple/control_dependency3^gradients/sbb1/conv4/BiasAdd_grad/tuple/group_deps*
T0*`
_classV
TRloc:@gradients/sbb1/batch_normalization/FusedBatchNormV3_grad/FusedBatchNormGradV3*0
_output_shapes
:˙˙˙˙˙˙˙˙˙^

<gradients/sbb1/conv4/BiasAdd_grad/tuple/control_dependency_1Identity-gradients/sbb1/conv4/BiasAdd_grad/BiasAddGrad3^gradients/sbb1/conv4/BiasAdd_grad/tuple/group_deps*
T0*@
_class6
42loc:@gradients/sbb1/conv4/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:

'gradients/sbb1/conv4/Conv2D_grad/ShapeNShapeNsbb1/Relu_2 sbb1/conv4/Conv2D/ReadVariableOp*
N*
T0* 
_output_shapes
::
Ü
4gradients/sbb1/conv4/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput'gradients/sbb1/conv4/Conv2D_grad/ShapeN sbb1/conv4/Conv2D/ReadVariableOp:gradients/sbb1/conv4/BiasAdd_grad/tuple/control_dependency*
T0*&
 _has_manual_control_dependencies(*/
_output_shapes
:˙˙˙˙˙˙˙˙˙^ *
paddingSAME*
strides

Ă
5gradients/sbb1/conv4/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFiltersbb1/Relu_2)gradients/sbb1/conv4/Conv2D_grad/ShapeN:1:gradients/sbb1/conv4/BiasAdd_grad/tuple/control_dependency*
T0*&
 _has_manual_control_dependencies(*'
_output_shapes
: *
paddingSAME*
strides

Đ
1gradients/sbb1/conv4/Conv2D_grad/tuple/group_depsNoOp6^gradients/sbb1/conv4/Conv2D_grad/Conv2DBackpropFilter5^gradients/sbb1/conv4/Conv2D_grad/Conv2DBackpropInput*&
 _has_manual_control_dependencies(
˛
9gradients/sbb1/conv4/Conv2D_grad/tuple/control_dependencyIdentity4gradients/sbb1/conv4/Conv2D_grad/Conv2DBackpropInput2^gradients/sbb1/conv4/Conv2D_grad/tuple/group_deps*
T0*G
_class=
;9loc:@gradients/sbb1/conv4/Conv2D_grad/Conv2DBackpropInput*/
_output_shapes
:˙˙˙˙˙˙˙˙˙^ 
Ž
;gradients/sbb1/conv4/Conv2D_grad/tuple/control_dependency_1Identity5gradients/sbb1/conv4/Conv2D_grad/Conv2DBackpropFilter2^gradients/sbb1/conv4/Conv2D_grad/tuple/group_deps*
T0*H
_class>
<:loc:@gradients/sbb1/conv4/Conv2D_grad/Conv2DBackpropFilter*'
_output_shapes
: 
Ů
#gradients/sbb1/Relu_2_grad/ReluGradReluGrad9gradients/sbb1/conv4/Conv2D_grad/tuple/control_dependencysbb1/Relu_2*
T0*&
 _has_manual_control_dependencies(*/
_output_shapes
:˙˙˙˙˙˙˙˙˙^ 
Ž
-gradients/sbb1/conv3/BiasAdd_grad/BiasAddGradBiasAddGrad#gradients/sbb1/Relu_2_grad/ReluGrad*
T0*&
 _has_manual_control_dependencies(*
_output_shapes
: 
¸
2gradients/sbb1/conv3/BiasAdd_grad/tuple/group_depsNoOp$^gradients/sbb1/Relu_2_grad/ReluGrad.^gradients/sbb1/conv3/BiasAdd_grad/BiasAddGrad*&
 _has_manual_control_dependencies(

:gradients/sbb1/conv3/BiasAdd_grad/tuple/control_dependencyIdentity#gradients/sbb1/Relu_2_grad/ReluGrad3^gradients/sbb1/conv3/BiasAdd_grad/tuple/group_deps*
T0*6
_class,
*(loc:@gradients/sbb1/Relu_2_grad/ReluGrad*/
_output_shapes
:˙˙˙˙˙˙˙˙˙^ 

<gradients/sbb1/conv3/BiasAdd_grad/tuple/control_dependency_1Identity-gradients/sbb1/conv3/BiasAdd_grad/BiasAddGrad3^gradients/sbb1/conv3/BiasAdd_grad/tuple/group_deps*
T0*@
_class6
42loc:@gradients/sbb1/conv3/BiasAdd_grad/BiasAddGrad*
_output_shapes
: 

'gradients/sbb1/conv3/Conv2D_grad/ShapeNShapeNsbb1/Relu_1 sbb1/conv3/Conv2D/ReadVariableOp*
N*
T0* 
_output_shapes
::
Ü
4gradients/sbb1/conv3/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput'gradients/sbb1/conv3/Conv2D_grad/ShapeN sbb1/conv3/Conv2D/ReadVariableOp:gradients/sbb1/conv3/BiasAdd_grad/tuple/control_dependency*
T0*&
 _has_manual_control_dependencies(*/
_output_shapes
:˙˙˙˙˙˙˙˙˙^ *
paddingSAME*
strides

Â
5gradients/sbb1/conv3/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFiltersbb1/Relu_1)gradients/sbb1/conv3/Conv2D_grad/ShapeN:1:gradients/sbb1/conv3/BiasAdd_grad/tuple/control_dependency*
T0*&
 _has_manual_control_dependencies(*&
_output_shapes
:  *
paddingSAME*
strides

Đ
1gradients/sbb1/conv3/Conv2D_grad/tuple/group_depsNoOp6^gradients/sbb1/conv3/Conv2D_grad/Conv2DBackpropFilter5^gradients/sbb1/conv3/Conv2D_grad/Conv2DBackpropInput*&
 _has_manual_control_dependencies(
˛
9gradients/sbb1/conv3/Conv2D_grad/tuple/control_dependencyIdentity4gradients/sbb1/conv3/Conv2D_grad/Conv2DBackpropInput2^gradients/sbb1/conv3/Conv2D_grad/tuple/group_deps*
T0*G
_class=
;9loc:@gradients/sbb1/conv3/Conv2D_grad/Conv2DBackpropInput*/
_output_shapes
:˙˙˙˙˙˙˙˙˙^ 
­
;gradients/sbb1/conv3/Conv2D_grad/tuple/control_dependency_1Identity5gradients/sbb1/conv3/Conv2D_grad/Conv2DBackpropFilter2^gradients/sbb1/conv3/Conv2D_grad/tuple/group_deps*
T0*H
_class>
<:loc:@gradients/sbb1/conv3/Conv2D_grad/Conv2DBackpropFilter*&
_output_shapes
:  
Ů
#gradients/sbb1/Relu_1_grad/ReluGradReluGrad9gradients/sbb1/conv3/Conv2D_grad/tuple/control_dependencysbb1/Relu_1*
T0*&
 _has_manual_control_dependencies(*/
_output_shapes
:˙˙˙˙˙˙˙˙˙^ 
Ž
-gradients/sbb1/conv2/BiasAdd_grad/BiasAddGradBiasAddGrad#gradients/sbb1/Relu_1_grad/ReluGrad*
T0*&
 _has_manual_control_dependencies(*
_output_shapes
: 
¸
2gradients/sbb1/conv2/BiasAdd_grad/tuple/group_depsNoOp$^gradients/sbb1/Relu_1_grad/ReluGrad.^gradients/sbb1/conv2/BiasAdd_grad/BiasAddGrad*&
 _has_manual_control_dependencies(

:gradients/sbb1/conv2/BiasAdd_grad/tuple/control_dependencyIdentity#gradients/sbb1/Relu_1_grad/ReluGrad3^gradients/sbb1/conv2/BiasAdd_grad/tuple/group_deps*
T0*6
_class,
*(loc:@gradients/sbb1/Relu_1_grad/ReluGrad*/
_output_shapes
:˙˙˙˙˙˙˙˙˙^ 

<gradients/sbb1/conv2/BiasAdd_grad/tuple/control_dependency_1Identity-gradients/sbb1/conv2/BiasAdd_grad/BiasAddGrad3^gradients/sbb1/conv2/BiasAdd_grad/tuple/group_deps*
T0*@
_class6
42loc:@gradients/sbb1/conv2/BiasAdd_grad/BiasAddGrad*
_output_shapes
: 

'gradients/sbb1/conv2/Conv2D_grad/ShapeNShapeN	sbb1/Relu sbb1/conv2/Conv2D/ReadVariableOp*
N*
T0* 
_output_shapes
::
Ü
4gradients/sbb1/conv2/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput'gradients/sbb1/conv2/Conv2D_grad/ShapeN sbb1/conv2/Conv2D/ReadVariableOp:gradients/sbb1/conv2/BiasAdd_grad/tuple/control_dependency*
T0*&
 _has_manual_control_dependencies(*/
_output_shapes
:˙˙˙˙˙˙˙˙˙^ *
paddingSAME*
strides

Ŕ
5gradients/sbb1/conv2/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilter	sbb1/Relu)gradients/sbb1/conv2/Conv2D_grad/ShapeN:1:gradients/sbb1/conv2/BiasAdd_grad/tuple/control_dependency*
T0*&
 _has_manual_control_dependencies(*&
_output_shapes
:  *
paddingSAME*
strides

Đ
1gradients/sbb1/conv2/Conv2D_grad/tuple/group_depsNoOp6^gradients/sbb1/conv2/Conv2D_grad/Conv2DBackpropFilter5^gradients/sbb1/conv2/Conv2D_grad/Conv2DBackpropInput*&
 _has_manual_control_dependencies(
˛
9gradients/sbb1/conv2/Conv2D_grad/tuple/control_dependencyIdentity4gradients/sbb1/conv2/Conv2D_grad/Conv2DBackpropInput2^gradients/sbb1/conv2/Conv2D_grad/tuple/group_deps*
T0*G
_class=
;9loc:@gradients/sbb1/conv2/Conv2D_grad/Conv2DBackpropInput*/
_output_shapes
:˙˙˙˙˙˙˙˙˙^ 
­
;gradients/sbb1/conv2/Conv2D_grad/tuple/control_dependency_1Identity5gradients/sbb1/conv2/Conv2D_grad/Conv2DBackpropFilter2^gradients/sbb1/conv2/Conv2D_grad/tuple/group_deps*
T0*H
_class>
<:loc:@gradients/sbb1/conv2/Conv2D_grad/Conv2DBackpropFilter*&
_output_shapes
:  
Ő
!gradients/sbb1/Relu_grad/ReluGradReluGrad9gradients/sbb1/conv2/Conv2D_grad/tuple/control_dependency	sbb1/Relu*
T0*&
 _has_manual_control_dependencies(*/
_output_shapes
:˙˙˙˙˙˙˙˙˙^ 
Ź
-gradients/sbb1/conv1/BiasAdd_grad/BiasAddGradBiasAddGrad!gradients/sbb1/Relu_grad/ReluGrad*
T0*&
 _has_manual_control_dependencies(*
_output_shapes
: 
ś
2gradients/sbb1/conv1/BiasAdd_grad/tuple/group_depsNoOp"^gradients/sbb1/Relu_grad/ReluGrad.^gradients/sbb1/conv1/BiasAdd_grad/BiasAddGrad*&
 _has_manual_control_dependencies(

:gradients/sbb1/conv1/BiasAdd_grad/tuple/control_dependencyIdentity!gradients/sbb1/Relu_grad/ReluGrad3^gradients/sbb1/conv1/BiasAdd_grad/tuple/group_deps*
T0*4
_class*
(&loc:@gradients/sbb1/Relu_grad/ReluGrad*/
_output_shapes
:˙˙˙˙˙˙˙˙˙^ 

<gradients/sbb1/conv1/BiasAdd_grad/tuple/control_dependency_1Identity-gradients/sbb1/conv1/BiasAdd_grad/BiasAddGrad3^gradients/sbb1/conv1/BiasAdd_grad/tuple/group_deps*
T0*@
_class6
42loc:@gradients/sbb1/conv1/BiasAdd_grad/BiasAddGrad*
_output_shapes
: 

'gradients/sbb1/conv1/Conv2D_grad/ShapeNShapeNMaxPool sbb1/conv1/Conv2D/ReadVariableOp*
N*
T0* 
_output_shapes
::
Ü
4gradients/sbb1/conv1/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput'gradients/sbb1/conv1/Conv2D_grad/ShapeN sbb1/conv1/Conv2D/ReadVariableOp:gradients/sbb1/conv1/BiasAdd_grad/tuple/control_dependency*
T0*&
 _has_manual_control_dependencies(*/
_output_shapes
:˙˙˙˙˙˙˙˙˙^@*
paddingSAME*
strides

ž
5gradients/sbb1/conv1/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterMaxPool)gradients/sbb1/conv1/Conv2D_grad/ShapeN:1:gradients/sbb1/conv1/BiasAdd_grad/tuple/control_dependency*
T0*&
 _has_manual_control_dependencies(*&
_output_shapes
:@ *
paddingSAME*
strides

Đ
1gradients/sbb1/conv1/Conv2D_grad/tuple/group_depsNoOp6^gradients/sbb1/conv1/Conv2D_grad/Conv2DBackpropFilter5^gradients/sbb1/conv1/Conv2D_grad/Conv2DBackpropInput*&
 _has_manual_control_dependencies(
˛
9gradients/sbb1/conv1/Conv2D_grad/tuple/control_dependencyIdentity4gradients/sbb1/conv1/Conv2D_grad/Conv2DBackpropInput2^gradients/sbb1/conv1/Conv2D_grad/tuple/group_deps*
T0*G
_class=
;9loc:@gradients/sbb1/conv1/Conv2D_grad/Conv2DBackpropInput*/
_output_shapes
:˙˙˙˙˙˙˙˙˙^@
­
;gradients/sbb1/conv1/Conv2D_grad/tuple/control_dependency_1Identity5gradients/sbb1/conv1/Conv2D_grad/Conv2DBackpropFilter2^gradients/sbb1/conv1/Conv2D_grad/tuple/group_deps*
T0*H
_class>
<:loc:@gradients/sbb1/conv1/Conv2D_grad/Conv2DBackpropFilter*&
_output_shapes
:@ 
ç
"gradients/MaxPool_grad/MaxPoolGradMaxPoolGradReluMaxPool9gradients/sbb1/conv1/Conv2D_grad/tuple/control_dependency*/
_output_shapes
:˙˙˙˙˙˙˙˙˙^@*
ksize
*
paddingSAME*
strides


gradients/Relu_grad/ReluGradReluGrad"gradients/MaxPool_grad/MaxPoolGradRelu*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙^@
q
gradients/zeros_like_26	ZerosLike&batch_normalization/FusedBatchNormV3:1*
T0*
_output_shapes
:@
q
gradients/zeros_like_27	ZerosLike&batch_normalization/FusedBatchNormV3:2*
T0*
_output_shapes
:@
q
gradients/zeros_like_28	ZerosLike&batch_normalization/FusedBatchNormV3:3*
T0*
_output_shapes
:@
q
gradients/zeros_like_29	ZerosLike&batch_normalization/FusedBatchNormV3:4*
T0*
_output_shapes
:@
o
gradients/zeros_like_30	ZerosLike&batch_normalization/FusedBatchNormV3:5*
T0*
_output_shapes
:
é
Hgradients/batch_normalization/FusedBatchNormV3_grad/FusedBatchNormGradV3FusedBatchNormGradV3gradients/Relu_grad/ReluGradconv1/BiasAdd"batch_normalization/ReadVariableOp3batch_normalization/FusedBatchNormV3/ReadVariableOp5batch_normalization/FusedBatchNormV3/ReadVariableOp_1&batch_normalization/FusedBatchNormV3:5*
T0*
U0*&
 _has_manual_control_dependencies(*C
_output_shapes1
/:˙˙˙˙˙˙˙˙˙^@:@:@: : *
epsilon%o:*
is_training( 
ż
Dgradients/batch_normalization/FusedBatchNormV3_grad/tuple/group_depsNoOpI^gradients/batch_normalization/FusedBatchNormV3_grad/FusedBatchNormGradV3*&
 _has_manual_control_dependencies(

Lgradients/batch_normalization/FusedBatchNormV3_grad/tuple/control_dependencyIdentityHgradients/batch_normalization/FusedBatchNormV3_grad/FusedBatchNormGradV3E^gradients/batch_normalization/FusedBatchNormV3_grad/tuple/group_deps*
T0*[
_classQ
OMloc:@gradients/batch_normalization/FusedBatchNormV3_grad/FusedBatchNormGradV3*/
_output_shapes
:˙˙˙˙˙˙˙˙˙^@
ď
Ngradients/batch_normalization/FusedBatchNormV3_grad/tuple/control_dependency_1IdentityJgradients/batch_normalization/FusedBatchNormV3_grad/FusedBatchNormGradV3:1E^gradients/batch_normalization/FusedBatchNormV3_grad/tuple/group_deps*
T0*[
_classQ
OMloc:@gradients/batch_normalization/FusedBatchNormV3_grad/FusedBatchNormGradV3*
_output_shapes
:@
ď
Ngradients/batch_normalization/FusedBatchNormV3_grad/tuple/control_dependency_2IdentityJgradients/batch_normalization/FusedBatchNormV3_grad/FusedBatchNormGradV3:2E^gradients/batch_normalization/FusedBatchNormV3_grad/tuple/group_deps*
T0*[
_classQ
OMloc:@gradients/batch_normalization/FusedBatchNormV3_grad/FusedBatchNormGradV3*
_output_shapes
:@
´
gradients/AddN_6AddN&gradients/gc1/AvgPool_grad/AvgPoolGradLgradients/batch_normalization/FusedBatchNormV3_grad/tuple/control_dependency*
N*
T0*9
_class/
-+loc:@gradients/gc1/AvgPool_grad/AvgPoolGrad*&
 _has_manual_control_dependencies(*/
_output_shapes
:˙˙˙˙˙˙˙˙˙^@

(gradients/conv1/BiasAdd_grad/BiasAddGradBiasAddGradgradients/AddN_6*
T0*&
 _has_manual_control_dependencies(*
_output_shapes
:@

-gradients/conv1/BiasAdd_grad/tuple/group_depsNoOp^gradients/AddN_6)^gradients/conv1/BiasAdd_grad/BiasAddGrad*&
 _has_manual_control_dependencies(
ř
5gradients/conv1/BiasAdd_grad/tuple/control_dependencyIdentitygradients/AddN_6.^gradients/conv1/BiasAdd_grad/tuple/group_deps*
T0*9
_class/
-+loc:@gradients/gc1/AvgPool_grad/AvgPoolGrad*/
_output_shapes
:˙˙˙˙˙˙˙˙˙^@
˙
7gradients/conv1/BiasAdd_grad/tuple/control_dependency_1Identity(gradients/conv1/BiasAdd_grad/BiasAddGrad.^gradients/conv1/BiasAdd_grad/tuple/group_deps*
T0*;
_class1
/-loc:@gradients/conv1/BiasAdd_grad/BiasAddGrad*
_output_shapes
:@

"gradients/conv1/Conv2D_grad/ShapeNShapeNinputsconv1/Conv2D/ReadVariableOp*
N*
T0* 
_output_shapes
::
Č
/gradients/conv1/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput"gradients/conv1/Conv2D_grad/ShapeNconv1/Conv2D/ReadVariableOp5gradients/conv1/BiasAdd_grad/tuple/control_dependency*
T0*&
 _has_manual_control_dependencies(*/
_output_shapes
:˙˙˙˙˙˙˙˙˙^*
paddingSAME*
strides

Ž
0gradients/conv1/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterinputs$gradients/conv1/Conv2D_grad/ShapeN:15gradients/conv1/BiasAdd_grad/tuple/control_dependency*
T0*&
 _has_manual_control_dependencies(*&
_output_shapes
:@*
paddingSAME*
strides

Á
,gradients/conv1/Conv2D_grad/tuple/group_depsNoOp1^gradients/conv1/Conv2D_grad/Conv2DBackpropFilter0^gradients/conv1/Conv2D_grad/Conv2DBackpropInput*&
 _has_manual_control_dependencies(

4gradients/conv1/Conv2D_grad/tuple/control_dependencyIdentity/gradients/conv1/Conv2D_grad/Conv2DBackpropInput-^gradients/conv1/Conv2D_grad/tuple/group_deps*
T0*B
_class8
64loc:@gradients/conv1/Conv2D_grad/Conv2DBackpropInput*/
_output_shapes
:˙˙˙˙˙˙˙˙˙^

6gradients/conv1/Conv2D_grad/tuple/control_dependency_1Identity0gradients/conv1/Conv2D_grad/Conv2DBackpropFilter-^gradients/conv1/Conv2D_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/conv1/Conv2D_grad/Conv2DBackpropFilter*&
_output_shapes
:@

%beta1_power/Initializer/initial_valueConst*+
_class!
loc:@batch_normalization/beta*
_output_shapes
: *
dtype0*
valueB
 *fff?

beta1_powerVarHandleOp*+
_class!
loc:@batch_normalization/beta*
_output_shapes
: *
dtype0*
shape: *
shared_namebeta1_power

,beta1_power/IsInitialized/VarIsInitializedOpVarIsInitializedOpbeta1_power*+
_class!
loc:@batch_normalization/beta*
_output_shapes
: 

beta1_power/AssignAssignVariableOpbeta1_power%beta1_power/Initializer/initial_value*&
 _has_manual_control_dependencies(*
dtype0

beta1_power/Read/ReadVariableOpReadVariableOpbeta1_power*+
_class!
loc:@batch_normalization/beta*
_output_shapes
: *
dtype0

%beta2_power/Initializer/initial_valueConst*+
_class!
loc:@batch_normalization/beta*
_output_shapes
: *
dtype0*
valueB
 *wž?

beta2_powerVarHandleOp*+
_class!
loc:@batch_normalization/beta*
_output_shapes
: *
dtype0*
shape: *
shared_namebeta2_power

,beta2_power/IsInitialized/VarIsInitializedOpVarIsInitializedOpbeta2_power*+
_class!
loc:@batch_normalization/beta*
_output_shapes
: 

beta2_power/AssignAssignVariableOpbeta2_power%beta2_power/Initializer/initial_value*&
 _has_manual_control_dependencies(*
dtype0

beta2_power/Read/ReadVariableOpReadVariableOpbeta2_power*+
_class!
loc:@batch_normalization/beta*
_output_shapes
: *
dtype0
Ł
.conv1/w/Adam/Initializer/zeros/shape_as_tensorConst*
_class
loc:@conv1/w*
_output_shapes
:*
dtype0*%
valueB"         @   

$conv1/w/Adam/Initializer/zeros/ConstConst*
_class
loc:@conv1/w*
_output_shapes
: *
dtype0*
valueB
 *    
É
conv1/w/Adam/Initializer/zerosFill.conv1/w/Adam/Initializer/zeros/shape_as_tensor$conv1/w/Adam/Initializer/zeros/Const*
T0*
_class
loc:@conv1/w*&
_output_shapes
:@

conv1/w/AdamVarHandleOp*
_class
loc:@conv1/w*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv1/w/Adam

-conv1/w/Adam/IsInitialized/VarIsInitializedOpVarIsInitializedOpconv1/w/Adam*
_class
loc:@conv1/w*
_output_shapes
: 

conv1/w/Adam/AssignAssignVariableOpconv1/w/Adamconv1/w/Adam/Initializer/zeros*&
 _has_manual_control_dependencies(*
dtype0

 conv1/w/Adam/Read/ReadVariableOpReadVariableOpconv1/w/Adam*
_class
loc:@conv1/w*&
_output_shapes
:@*
dtype0
Ľ
0conv1/w/Adam_1/Initializer/zeros/shape_as_tensorConst*
_class
loc:@conv1/w*
_output_shapes
:*
dtype0*%
valueB"         @   

&conv1/w/Adam_1/Initializer/zeros/ConstConst*
_class
loc:@conv1/w*
_output_shapes
: *
dtype0*
valueB
 *    
Ď
 conv1/w/Adam_1/Initializer/zerosFill0conv1/w/Adam_1/Initializer/zeros/shape_as_tensor&conv1/w/Adam_1/Initializer/zeros/Const*
T0*
_class
loc:@conv1/w*&
_output_shapes
:@

conv1/w/Adam_1VarHandleOp*
_class
loc:@conv1/w*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv1/w/Adam_1

/conv1/w/Adam_1/IsInitialized/VarIsInitializedOpVarIsInitializedOpconv1/w/Adam_1*
_class
loc:@conv1/w*
_output_shapes
: 

conv1/w/Adam_1/AssignAssignVariableOpconv1/w/Adam_1 conv1/w/Adam_1/Initializer/zeros*&
 _has_manual_control_dependencies(*
dtype0

"conv1/w/Adam_1/Read/ReadVariableOpReadVariableOpconv1/w/Adam_1*
_class
loc:@conv1/w*&
_output_shapes
:@*
dtype0

conv1/b/Adam/Initializer/zerosConst*
_class
loc:@conv1/b*
_output_shapes
:@*
dtype0*
valueB@*    

conv1/b/AdamVarHandleOp*
_class
loc:@conv1/b*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv1/b/Adam

-conv1/b/Adam/IsInitialized/VarIsInitializedOpVarIsInitializedOpconv1/b/Adam*
_class
loc:@conv1/b*
_output_shapes
: 

conv1/b/Adam/AssignAssignVariableOpconv1/b/Adamconv1/b/Adam/Initializer/zeros*&
 _has_manual_control_dependencies(*
dtype0

 conv1/b/Adam/Read/ReadVariableOpReadVariableOpconv1/b/Adam*
_class
loc:@conv1/b*
_output_shapes
:@*
dtype0

 conv1/b/Adam_1/Initializer/zerosConst*
_class
loc:@conv1/b*
_output_shapes
:@*
dtype0*
valueB@*    

conv1/b/Adam_1VarHandleOp*
_class
loc:@conv1/b*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv1/b/Adam_1

/conv1/b/Adam_1/IsInitialized/VarIsInitializedOpVarIsInitializedOpconv1/b/Adam_1*
_class
loc:@conv1/b*
_output_shapes
: 

conv1/b/Adam_1/AssignAssignVariableOpconv1/b/Adam_1 conv1/b/Adam_1/Initializer/zeros*&
 _has_manual_control_dependencies(*
dtype0

"conv1/b/Adam_1/Read/ReadVariableOpReadVariableOpconv1/b/Adam_1*
_class
loc:@conv1/b*
_output_shapes
:@*
dtype0
Ť
0batch_normalization/gamma/Adam/Initializer/zerosConst*,
_class"
 loc:@batch_normalization/gamma*
_output_shapes
:@*
dtype0*
valueB@*    
Â
batch_normalization/gamma/AdamVarHandleOp*,
_class"
 loc:@batch_normalization/gamma*
_output_shapes
: *
dtype0*
shape:@*/
shared_name batch_normalization/gamma/Adam
ť
?batch_normalization/gamma/Adam/IsInitialized/VarIsInitializedOpVarIsInitializedOpbatch_normalization/gamma/Adam*,
_class"
 loc:@batch_normalization/gamma*
_output_shapes
: 
Ŕ
%batch_normalization/gamma/Adam/AssignAssignVariableOpbatch_normalization/gamma/Adam0batch_normalization/gamma/Adam/Initializer/zeros*&
 _has_manual_control_dependencies(*
dtype0
ť
2batch_normalization/gamma/Adam/Read/ReadVariableOpReadVariableOpbatch_normalization/gamma/Adam*,
_class"
 loc:@batch_normalization/gamma*
_output_shapes
:@*
dtype0
­
2batch_normalization/gamma/Adam_1/Initializer/zerosConst*,
_class"
 loc:@batch_normalization/gamma*
_output_shapes
:@*
dtype0*
valueB@*    
Ć
 batch_normalization/gamma/Adam_1VarHandleOp*,
_class"
 loc:@batch_normalization/gamma*
_output_shapes
: *
dtype0*
shape:@*1
shared_name" batch_normalization/gamma/Adam_1
ż
Abatch_normalization/gamma/Adam_1/IsInitialized/VarIsInitializedOpVarIsInitializedOp batch_normalization/gamma/Adam_1*,
_class"
 loc:@batch_normalization/gamma*
_output_shapes
: 
Ć
'batch_normalization/gamma/Adam_1/AssignAssignVariableOp batch_normalization/gamma/Adam_12batch_normalization/gamma/Adam_1/Initializer/zeros*&
 _has_manual_control_dependencies(*
dtype0
ż
4batch_normalization/gamma/Adam_1/Read/ReadVariableOpReadVariableOp batch_normalization/gamma/Adam_1*,
_class"
 loc:@batch_normalization/gamma*
_output_shapes
:@*
dtype0
Š
/batch_normalization/beta/Adam/Initializer/zerosConst*+
_class!
loc:@batch_normalization/beta*
_output_shapes
:@*
dtype0*
valueB@*    
ż
batch_normalization/beta/AdamVarHandleOp*+
_class!
loc:@batch_normalization/beta*
_output_shapes
: *
dtype0*
shape:@*.
shared_namebatch_normalization/beta/Adam
¸
>batch_normalization/beta/Adam/IsInitialized/VarIsInitializedOpVarIsInitializedOpbatch_normalization/beta/Adam*+
_class!
loc:@batch_normalization/beta*
_output_shapes
: 
˝
$batch_normalization/beta/Adam/AssignAssignVariableOpbatch_normalization/beta/Adam/batch_normalization/beta/Adam/Initializer/zeros*&
 _has_manual_control_dependencies(*
dtype0
¸
1batch_normalization/beta/Adam/Read/ReadVariableOpReadVariableOpbatch_normalization/beta/Adam*+
_class!
loc:@batch_normalization/beta*
_output_shapes
:@*
dtype0
Ť
1batch_normalization/beta/Adam_1/Initializer/zerosConst*+
_class!
loc:@batch_normalization/beta*
_output_shapes
:@*
dtype0*
valueB@*    
Ă
batch_normalization/beta/Adam_1VarHandleOp*+
_class!
loc:@batch_normalization/beta*
_output_shapes
: *
dtype0*
shape:@*0
shared_name!batch_normalization/beta/Adam_1
ź
@batch_normalization/beta/Adam_1/IsInitialized/VarIsInitializedOpVarIsInitializedOpbatch_normalization/beta/Adam_1*+
_class!
loc:@batch_normalization/beta*
_output_shapes
: 
Ă
&batch_normalization/beta/Adam_1/AssignAssignVariableOpbatch_normalization/beta/Adam_11batch_normalization/beta/Adam_1/Initializer/zeros*&
 _has_manual_control_dependencies(*
dtype0
ź
3batch_normalization/beta/Adam_1/Read/ReadVariableOpReadVariableOpbatch_normalization/beta/Adam_1*+
_class!
loc:@batch_normalization/beta*
_output_shapes
:@*
dtype0
­
3sbb1/conv1/w/Adam/Initializer/zeros/shape_as_tensorConst*
_class
loc:@sbb1/conv1/w*
_output_shapes
:*
dtype0*%
valueB"      @       

)sbb1/conv1/w/Adam/Initializer/zeros/ConstConst*
_class
loc:@sbb1/conv1/w*
_output_shapes
: *
dtype0*
valueB
 *    
Ý
#sbb1/conv1/w/Adam/Initializer/zerosFill3sbb1/conv1/w/Adam/Initializer/zeros/shape_as_tensor)sbb1/conv1/w/Adam/Initializer/zeros/Const*
T0*
_class
loc:@sbb1/conv1/w*&
_output_shapes
:@ 
§
sbb1/conv1/w/AdamVarHandleOp*
_class
loc:@sbb1/conv1/w*
_output_shapes
: *
dtype0*
shape:@ *"
shared_namesbb1/conv1/w/Adam

2sbb1/conv1/w/Adam/IsInitialized/VarIsInitializedOpVarIsInitializedOpsbb1/conv1/w/Adam*
_class
loc:@sbb1/conv1/w*
_output_shapes
: 

sbb1/conv1/w/Adam/AssignAssignVariableOpsbb1/conv1/w/Adam#sbb1/conv1/w/Adam/Initializer/zeros*&
 _has_manual_control_dependencies(*
dtype0
 
%sbb1/conv1/w/Adam/Read/ReadVariableOpReadVariableOpsbb1/conv1/w/Adam*
_class
loc:@sbb1/conv1/w*&
_output_shapes
:@ *
dtype0
Ż
5sbb1/conv1/w/Adam_1/Initializer/zeros/shape_as_tensorConst*
_class
loc:@sbb1/conv1/w*
_output_shapes
:*
dtype0*%
valueB"      @       

+sbb1/conv1/w/Adam_1/Initializer/zeros/ConstConst*
_class
loc:@sbb1/conv1/w*
_output_shapes
: *
dtype0*
valueB
 *    
ă
%sbb1/conv1/w/Adam_1/Initializer/zerosFill5sbb1/conv1/w/Adam_1/Initializer/zeros/shape_as_tensor+sbb1/conv1/w/Adam_1/Initializer/zeros/Const*
T0*
_class
loc:@sbb1/conv1/w*&
_output_shapes
:@ 
Ť
sbb1/conv1/w/Adam_1VarHandleOp*
_class
loc:@sbb1/conv1/w*
_output_shapes
: *
dtype0*
shape:@ *$
shared_namesbb1/conv1/w/Adam_1

4sbb1/conv1/w/Adam_1/IsInitialized/VarIsInitializedOpVarIsInitializedOpsbb1/conv1/w/Adam_1*
_class
loc:@sbb1/conv1/w*
_output_shapes
: 

sbb1/conv1/w/Adam_1/AssignAssignVariableOpsbb1/conv1/w/Adam_1%sbb1/conv1/w/Adam_1/Initializer/zeros*&
 _has_manual_control_dependencies(*
dtype0
¤
'sbb1/conv1/w/Adam_1/Read/ReadVariableOpReadVariableOpsbb1/conv1/w/Adam_1*
_class
loc:@sbb1/conv1/w*&
_output_shapes
:@ *
dtype0

#sbb1/conv1/b/Adam/Initializer/zerosConst*
_class
loc:@sbb1/conv1/b*
_output_shapes
: *
dtype0*
valueB *    

sbb1/conv1/b/AdamVarHandleOp*
_class
loc:@sbb1/conv1/b*
_output_shapes
: *
dtype0*
shape: *"
shared_namesbb1/conv1/b/Adam

2sbb1/conv1/b/Adam/IsInitialized/VarIsInitializedOpVarIsInitializedOpsbb1/conv1/b/Adam*
_class
loc:@sbb1/conv1/b*
_output_shapes
: 

sbb1/conv1/b/Adam/AssignAssignVariableOpsbb1/conv1/b/Adam#sbb1/conv1/b/Adam/Initializer/zeros*&
 _has_manual_control_dependencies(*
dtype0

%sbb1/conv1/b/Adam/Read/ReadVariableOpReadVariableOpsbb1/conv1/b/Adam*
_class
loc:@sbb1/conv1/b*
_output_shapes
: *
dtype0

%sbb1/conv1/b/Adam_1/Initializer/zerosConst*
_class
loc:@sbb1/conv1/b*
_output_shapes
: *
dtype0*
valueB *    

sbb1/conv1/b/Adam_1VarHandleOp*
_class
loc:@sbb1/conv1/b*
_output_shapes
: *
dtype0*
shape: *$
shared_namesbb1/conv1/b/Adam_1

4sbb1/conv1/b/Adam_1/IsInitialized/VarIsInitializedOpVarIsInitializedOpsbb1/conv1/b/Adam_1*
_class
loc:@sbb1/conv1/b*
_output_shapes
: 

sbb1/conv1/b/Adam_1/AssignAssignVariableOpsbb1/conv1/b/Adam_1%sbb1/conv1/b/Adam_1/Initializer/zeros*&
 _has_manual_control_dependencies(*
dtype0

'sbb1/conv1/b/Adam_1/Read/ReadVariableOpReadVariableOpsbb1/conv1/b/Adam_1*
_class
loc:@sbb1/conv1/b*
_output_shapes
: *
dtype0
­
3sbb1/conv2/w/Adam/Initializer/zeros/shape_as_tensorConst*
_class
loc:@sbb1/conv2/w*
_output_shapes
:*
dtype0*%
valueB"              

)sbb1/conv2/w/Adam/Initializer/zeros/ConstConst*
_class
loc:@sbb1/conv2/w*
_output_shapes
: *
dtype0*
valueB
 *    
Ý
#sbb1/conv2/w/Adam/Initializer/zerosFill3sbb1/conv2/w/Adam/Initializer/zeros/shape_as_tensor)sbb1/conv2/w/Adam/Initializer/zeros/Const*
T0*
_class
loc:@sbb1/conv2/w*&
_output_shapes
:  
§
sbb1/conv2/w/AdamVarHandleOp*
_class
loc:@sbb1/conv2/w*
_output_shapes
: *
dtype0*
shape:  *"
shared_namesbb1/conv2/w/Adam

2sbb1/conv2/w/Adam/IsInitialized/VarIsInitializedOpVarIsInitializedOpsbb1/conv2/w/Adam*
_class
loc:@sbb1/conv2/w*
_output_shapes
: 

sbb1/conv2/w/Adam/AssignAssignVariableOpsbb1/conv2/w/Adam#sbb1/conv2/w/Adam/Initializer/zeros*&
 _has_manual_control_dependencies(*
dtype0
 
%sbb1/conv2/w/Adam/Read/ReadVariableOpReadVariableOpsbb1/conv2/w/Adam*
_class
loc:@sbb1/conv2/w*&
_output_shapes
:  *
dtype0
Ż
5sbb1/conv2/w/Adam_1/Initializer/zeros/shape_as_tensorConst*
_class
loc:@sbb1/conv2/w*
_output_shapes
:*
dtype0*%
valueB"              

+sbb1/conv2/w/Adam_1/Initializer/zeros/ConstConst*
_class
loc:@sbb1/conv2/w*
_output_shapes
: *
dtype0*
valueB
 *    
ă
%sbb1/conv2/w/Adam_1/Initializer/zerosFill5sbb1/conv2/w/Adam_1/Initializer/zeros/shape_as_tensor+sbb1/conv2/w/Adam_1/Initializer/zeros/Const*
T0*
_class
loc:@sbb1/conv2/w*&
_output_shapes
:  
Ť
sbb1/conv2/w/Adam_1VarHandleOp*
_class
loc:@sbb1/conv2/w*
_output_shapes
: *
dtype0*
shape:  *$
shared_namesbb1/conv2/w/Adam_1

4sbb1/conv2/w/Adam_1/IsInitialized/VarIsInitializedOpVarIsInitializedOpsbb1/conv2/w/Adam_1*
_class
loc:@sbb1/conv2/w*
_output_shapes
: 

sbb1/conv2/w/Adam_1/AssignAssignVariableOpsbb1/conv2/w/Adam_1%sbb1/conv2/w/Adam_1/Initializer/zeros*&
 _has_manual_control_dependencies(*
dtype0
¤
'sbb1/conv2/w/Adam_1/Read/ReadVariableOpReadVariableOpsbb1/conv2/w/Adam_1*
_class
loc:@sbb1/conv2/w*&
_output_shapes
:  *
dtype0

#sbb1/conv2/b/Adam/Initializer/zerosConst*
_class
loc:@sbb1/conv2/b*
_output_shapes
: *
dtype0*
valueB *    

sbb1/conv2/b/AdamVarHandleOp*
_class
loc:@sbb1/conv2/b*
_output_shapes
: *
dtype0*
shape: *"
shared_namesbb1/conv2/b/Adam

2sbb1/conv2/b/Adam/IsInitialized/VarIsInitializedOpVarIsInitializedOpsbb1/conv2/b/Adam*
_class
loc:@sbb1/conv2/b*
_output_shapes
: 

sbb1/conv2/b/Adam/AssignAssignVariableOpsbb1/conv2/b/Adam#sbb1/conv2/b/Adam/Initializer/zeros*&
 _has_manual_control_dependencies(*
dtype0

%sbb1/conv2/b/Adam/Read/ReadVariableOpReadVariableOpsbb1/conv2/b/Adam*
_class
loc:@sbb1/conv2/b*
_output_shapes
: *
dtype0

%sbb1/conv2/b/Adam_1/Initializer/zerosConst*
_class
loc:@sbb1/conv2/b*
_output_shapes
: *
dtype0*
valueB *    

sbb1/conv2/b/Adam_1VarHandleOp*
_class
loc:@sbb1/conv2/b*
_output_shapes
: *
dtype0*
shape: *$
shared_namesbb1/conv2/b/Adam_1

4sbb1/conv2/b/Adam_1/IsInitialized/VarIsInitializedOpVarIsInitializedOpsbb1/conv2/b/Adam_1*
_class
loc:@sbb1/conv2/b*
_output_shapes
: 

sbb1/conv2/b/Adam_1/AssignAssignVariableOpsbb1/conv2/b/Adam_1%sbb1/conv2/b/Adam_1/Initializer/zeros*&
 _has_manual_control_dependencies(*
dtype0

'sbb1/conv2/b/Adam_1/Read/ReadVariableOpReadVariableOpsbb1/conv2/b/Adam_1*
_class
loc:@sbb1/conv2/b*
_output_shapes
: *
dtype0
­
3sbb1/conv3/w/Adam/Initializer/zeros/shape_as_tensorConst*
_class
loc:@sbb1/conv3/w*
_output_shapes
:*
dtype0*%
valueB"              

)sbb1/conv3/w/Adam/Initializer/zeros/ConstConst*
_class
loc:@sbb1/conv3/w*
_output_shapes
: *
dtype0*
valueB
 *    
Ý
#sbb1/conv3/w/Adam/Initializer/zerosFill3sbb1/conv3/w/Adam/Initializer/zeros/shape_as_tensor)sbb1/conv3/w/Adam/Initializer/zeros/Const*
T0*
_class
loc:@sbb1/conv3/w*&
_output_shapes
:  
§
sbb1/conv3/w/AdamVarHandleOp*
_class
loc:@sbb1/conv3/w*
_output_shapes
: *
dtype0*
shape:  *"
shared_namesbb1/conv3/w/Adam

2sbb1/conv3/w/Adam/IsInitialized/VarIsInitializedOpVarIsInitializedOpsbb1/conv3/w/Adam*
_class
loc:@sbb1/conv3/w*
_output_shapes
: 

sbb1/conv3/w/Adam/AssignAssignVariableOpsbb1/conv3/w/Adam#sbb1/conv3/w/Adam/Initializer/zeros*&
 _has_manual_control_dependencies(*
dtype0
 
%sbb1/conv3/w/Adam/Read/ReadVariableOpReadVariableOpsbb1/conv3/w/Adam*
_class
loc:@sbb1/conv3/w*&
_output_shapes
:  *
dtype0
Ż
5sbb1/conv3/w/Adam_1/Initializer/zeros/shape_as_tensorConst*
_class
loc:@sbb1/conv3/w*
_output_shapes
:*
dtype0*%
valueB"              

+sbb1/conv3/w/Adam_1/Initializer/zeros/ConstConst*
_class
loc:@sbb1/conv3/w*
_output_shapes
: *
dtype0*
valueB
 *    
ă
%sbb1/conv3/w/Adam_1/Initializer/zerosFill5sbb1/conv3/w/Adam_1/Initializer/zeros/shape_as_tensor+sbb1/conv3/w/Adam_1/Initializer/zeros/Const*
T0*
_class
loc:@sbb1/conv3/w*&
_output_shapes
:  
Ť
sbb1/conv3/w/Adam_1VarHandleOp*
_class
loc:@sbb1/conv3/w*
_output_shapes
: *
dtype0*
shape:  *$
shared_namesbb1/conv3/w/Adam_1

4sbb1/conv3/w/Adam_1/IsInitialized/VarIsInitializedOpVarIsInitializedOpsbb1/conv3/w/Adam_1*
_class
loc:@sbb1/conv3/w*
_output_shapes
: 

sbb1/conv3/w/Adam_1/AssignAssignVariableOpsbb1/conv3/w/Adam_1%sbb1/conv3/w/Adam_1/Initializer/zeros*&
 _has_manual_control_dependencies(*
dtype0
¤
'sbb1/conv3/w/Adam_1/Read/ReadVariableOpReadVariableOpsbb1/conv3/w/Adam_1*
_class
loc:@sbb1/conv3/w*&
_output_shapes
:  *
dtype0

#sbb1/conv3/b/Adam/Initializer/zerosConst*
_class
loc:@sbb1/conv3/b*
_output_shapes
: *
dtype0*
valueB *    

sbb1/conv3/b/AdamVarHandleOp*
_class
loc:@sbb1/conv3/b*
_output_shapes
: *
dtype0*
shape: *"
shared_namesbb1/conv3/b/Adam

2sbb1/conv3/b/Adam/IsInitialized/VarIsInitializedOpVarIsInitializedOpsbb1/conv3/b/Adam*
_class
loc:@sbb1/conv3/b*
_output_shapes
: 

sbb1/conv3/b/Adam/AssignAssignVariableOpsbb1/conv3/b/Adam#sbb1/conv3/b/Adam/Initializer/zeros*&
 _has_manual_control_dependencies(*
dtype0

%sbb1/conv3/b/Adam/Read/ReadVariableOpReadVariableOpsbb1/conv3/b/Adam*
_class
loc:@sbb1/conv3/b*
_output_shapes
: *
dtype0

%sbb1/conv3/b/Adam_1/Initializer/zerosConst*
_class
loc:@sbb1/conv3/b*
_output_shapes
: *
dtype0*
valueB *    

sbb1/conv3/b/Adam_1VarHandleOp*
_class
loc:@sbb1/conv3/b*
_output_shapes
: *
dtype0*
shape: *$
shared_namesbb1/conv3/b/Adam_1

4sbb1/conv3/b/Adam_1/IsInitialized/VarIsInitializedOpVarIsInitializedOpsbb1/conv3/b/Adam_1*
_class
loc:@sbb1/conv3/b*
_output_shapes
: 

sbb1/conv3/b/Adam_1/AssignAssignVariableOpsbb1/conv3/b/Adam_1%sbb1/conv3/b/Adam_1/Initializer/zeros*&
 _has_manual_control_dependencies(*
dtype0

'sbb1/conv3/b/Adam_1/Read/ReadVariableOpReadVariableOpsbb1/conv3/b/Adam_1*
_class
loc:@sbb1/conv3/b*
_output_shapes
: *
dtype0
­
3sbb1/conv4/w/Adam/Initializer/zeros/shape_as_tensorConst*
_class
loc:@sbb1/conv4/w*
_output_shapes
:*
dtype0*%
valueB"             

)sbb1/conv4/w/Adam/Initializer/zeros/ConstConst*
_class
loc:@sbb1/conv4/w*
_output_shapes
: *
dtype0*
valueB
 *    
Ţ
#sbb1/conv4/w/Adam/Initializer/zerosFill3sbb1/conv4/w/Adam/Initializer/zeros/shape_as_tensor)sbb1/conv4/w/Adam/Initializer/zeros/Const*
T0*
_class
loc:@sbb1/conv4/w*'
_output_shapes
: 
¨
sbb1/conv4/w/AdamVarHandleOp*
_class
loc:@sbb1/conv4/w*
_output_shapes
: *
dtype0*
shape: *"
shared_namesbb1/conv4/w/Adam

2sbb1/conv4/w/Adam/IsInitialized/VarIsInitializedOpVarIsInitializedOpsbb1/conv4/w/Adam*
_class
loc:@sbb1/conv4/w*
_output_shapes
: 

sbb1/conv4/w/Adam/AssignAssignVariableOpsbb1/conv4/w/Adam#sbb1/conv4/w/Adam/Initializer/zeros*&
 _has_manual_control_dependencies(*
dtype0
Ą
%sbb1/conv4/w/Adam/Read/ReadVariableOpReadVariableOpsbb1/conv4/w/Adam*
_class
loc:@sbb1/conv4/w*'
_output_shapes
: *
dtype0
Ż
5sbb1/conv4/w/Adam_1/Initializer/zeros/shape_as_tensorConst*
_class
loc:@sbb1/conv4/w*
_output_shapes
:*
dtype0*%
valueB"             

+sbb1/conv4/w/Adam_1/Initializer/zeros/ConstConst*
_class
loc:@sbb1/conv4/w*
_output_shapes
: *
dtype0*
valueB
 *    
ä
%sbb1/conv4/w/Adam_1/Initializer/zerosFill5sbb1/conv4/w/Adam_1/Initializer/zeros/shape_as_tensor+sbb1/conv4/w/Adam_1/Initializer/zeros/Const*
T0*
_class
loc:@sbb1/conv4/w*'
_output_shapes
: 
Ź
sbb1/conv4/w/Adam_1VarHandleOp*
_class
loc:@sbb1/conv4/w*
_output_shapes
: *
dtype0*
shape: *$
shared_namesbb1/conv4/w/Adam_1

4sbb1/conv4/w/Adam_1/IsInitialized/VarIsInitializedOpVarIsInitializedOpsbb1/conv4/w/Adam_1*
_class
loc:@sbb1/conv4/w*
_output_shapes
: 

sbb1/conv4/w/Adam_1/AssignAssignVariableOpsbb1/conv4/w/Adam_1%sbb1/conv4/w/Adam_1/Initializer/zeros*&
 _has_manual_control_dependencies(*
dtype0
Ľ
'sbb1/conv4/w/Adam_1/Read/ReadVariableOpReadVariableOpsbb1/conv4/w/Adam_1*
_class
loc:@sbb1/conv4/w*'
_output_shapes
: *
dtype0

#sbb1/conv4/b/Adam/Initializer/zerosConst*
_class
loc:@sbb1/conv4/b*
_output_shapes	
:*
dtype0*
valueB*    

sbb1/conv4/b/AdamVarHandleOp*
_class
loc:@sbb1/conv4/b*
_output_shapes
: *
dtype0*
shape:*"
shared_namesbb1/conv4/b/Adam

2sbb1/conv4/b/Adam/IsInitialized/VarIsInitializedOpVarIsInitializedOpsbb1/conv4/b/Adam*
_class
loc:@sbb1/conv4/b*
_output_shapes
: 

sbb1/conv4/b/Adam/AssignAssignVariableOpsbb1/conv4/b/Adam#sbb1/conv4/b/Adam/Initializer/zeros*&
 _has_manual_control_dependencies(*
dtype0

%sbb1/conv4/b/Adam/Read/ReadVariableOpReadVariableOpsbb1/conv4/b/Adam*
_class
loc:@sbb1/conv4/b*
_output_shapes	
:*
dtype0

%sbb1/conv4/b/Adam_1/Initializer/zerosConst*
_class
loc:@sbb1/conv4/b*
_output_shapes	
:*
dtype0*
valueB*    
 
sbb1/conv4/b/Adam_1VarHandleOp*
_class
loc:@sbb1/conv4/b*
_output_shapes
: *
dtype0*
shape:*$
shared_namesbb1/conv4/b/Adam_1

4sbb1/conv4/b/Adam_1/IsInitialized/VarIsInitializedOpVarIsInitializedOpsbb1/conv4/b/Adam_1*
_class
loc:@sbb1/conv4/b*
_output_shapes
: 

sbb1/conv4/b/Adam_1/AssignAssignVariableOpsbb1/conv4/b/Adam_1%sbb1/conv4/b/Adam_1/Initializer/zeros*&
 _has_manual_control_dependencies(*
dtype0

'sbb1/conv4/b/Adam_1/Read/ReadVariableOpReadVariableOpsbb1/conv4/b/Adam_1*
_class
loc:@sbb1/conv4/b*
_output_shapes	
:*
dtype0
ˇ
5sbb1/batch_normalization/gamma/Adam/Initializer/zerosConst*1
_class'
%#loc:@sbb1/batch_normalization/gamma*
_output_shapes	
:*
dtype0*
valueB*    
Ň
#sbb1/batch_normalization/gamma/AdamVarHandleOp*1
_class'
%#loc:@sbb1/batch_normalization/gamma*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#sbb1/batch_normalization/gamma/Adam
Ę
Dsbb1/batch_normalization/gamma/Adam/IsInitialized/VarIsInitializedOpVarIsInitializedOp#sbb1/batch_normalization/gamma/Adam*1
_class'
%#loc:@sbb1/batch_normalization/gamma*
_output_shapes
: 
Ď
*sbb1/batch_normalization/gamma/Adam/AssignAssignVariableOp#sbb1/batch_normalization/gamma/Adam5sbb1/batch_normalization/gamma/Adam/Initializer/zeros*&
 _has_manual_control_dependencies(*
dtype0
Ë
7sbb1/batch_normalization/gamma/Adam/Read/ReadVariableOpReadVariableOp#sbb1/batch_normalization/gamma/Adam*1
_class'
%#loc:@sbb1/batch_normalization/gamma*
_output_shapes	
:*
dtype0
š
7sbb1/batch_normalization/gamma/Adam_1/Initializer/zerosConst*1
_class'
%#loc:@sbb1/batch_normalization/gamma*
_output_shapes	
:*
dtype0*
valueB*    
Ö
%sbb1/batch_normalization/gamma/Adam_1VarHandleOp*1
_class'
%#loc:@sbb1/batch_normalization/gamma*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%sbb1/batch_normalization/gamma/Adam_1
Î
Fsbb1/batch_normalization/gamma/Adam_1/IsInitialized/VarIsInitializedOpVarIsInitializedOp%sbb1/batch_normalization/gamma/Adam_1*1
_class'
%#loc:@sbb1/batch_normalization/gamma*
_output_shapes
: 
Ő
,sbb1/batch_normalization/gamma/Adam_1/AssignAssignVariableOp%sbb1/batch_normalization/gamma/Adam_17sbb1/batch_normalization/gamma/Adam_1/Initializer/zeros*&
 _has_manual_control_dependencies(*
dtype0
Ď
9sbb1/batch_normalization/gamma/Adam_1/Read/ReadVariableOpReadVariableOp%sbb1/batch_normalization/gamma/Adam_1*1
_class'
%#loc:@sbb1/batch_normalization/gamma*
_output_shapes	
:*
dtype0
ľ
4sbb1/batch_normalization/beta/Adam/Initializer/zerosConst*0
_class&
$"loc:@sbb1/batch_normalization/beta*
_output_shapes	
:*
dtype0*
valueB*    
Ď
"sbb1/batch_normalization/beta/AdamVarHandleOp*0
_class&
$"loc:@sbb1/batch_normalization/beta*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"sbb1/batch_normalization/beta/Adam
Ç
Csbb1/batch_normalization/beta/Adam/IsInitialized/VarIsInitializedOpVarIsInitializedOp"sbb1/batch_normalization/beta/Adam*0
_class&
$"loc:@sbb1/batch_normalization/beta*
_output_shapes
: 
Ě
)sbb1/batch_normalization/beta/Adam/AssignAssignVariableOp"sbb1/batch_normalization/beta/Adam4sbb1/batch_normalization/beta/Adam/Initializer/zeros*&
 _has_manual_control_dependencies(*
dtype0
Č
6sbb1/batch_normalization/beta/Adam/Read/ReadVariableOpReadVariableOp"sbb1/batch_normalization/beta/Adam*0
_class&
$"loc:@sbb1/batch_normalization/beta*
_output_shapes	
:*
dtype0
ˇ
6sbb1/batch_normalization/beta/Adam_1/Initializer/zerosConst*0
_class&
$"loc:@sbb1/batch_normalization/beta*
_output_shapes	
:*
dtype0*
valueB*    
Ó
$sbb1/batch_normalization/beta/Adam_1VarHandleOp*0
_class&
$"loc:@sbb1/batch_normalization/beta*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$sbb1/batch_normalization/beta/Adam_1
Ë
Esbb1/batch_normalization/beta/Adam_1/IsInitialized/VarIsInitializedOpVarIsInitializedOp$sbb1/batch_normalization/beta/Adam_1*0
_class&
$"loc:@sbb1/batch_normalization/beta*
_output_shapes
: 
Ň
+sbb1/batch_normalization/beta/Adam_1/AssignAssignVariableOp$sbb1/batch_normalization/beta/Adam_16sbb1/batch_normalization/beta/Adam_1/Initializer/zeros*&
 _has_manual_control_dependencies(*
dtype0
Ě
8sbb1/batch_normalization/beta/Adam_1/Read/ReadVariableOpReadVariableOp$sbb1/batch_normalization/beta/Adam_1*0
_class&
$"loc:@sbb1/batch_normalization/beta*
_output_shapes	
:*
dtype0
­
3sbb2/conv1/w/Adam/Initializer/zeros/shape_as_tensorConst*
_class
loc:@sbb2/conv1/w*
_output_shapes
:*
dtype0*%
valueB"         @   

)sbb2/conv1/w/Adam/Initializer/zeros/ConstConst*
_class
loc:@sbb2/conv1/w*
_output_shapes
: *
dtype0*
valueB
 *    
Ţ
#sbb2/conv1/w/Adam/Initializer/zerosFill3sbb2/conv1/w/Adam/Initializer/zeros/shape_as_tensor)sbb2/conv1/w/Adam/Initializer/zeros/Const*
T0*
_class
loc:@sbb2/conv1/w*'
_output_shapes
:@
¨
sbb2/conv1/w/AdamVarHandleOp*
_class
loc:@sbb2/conv1/w*
_output_shapes
: *
dtype0*
shape:@*"
shared_namesbb2/conv1/w/Adam

2sbb2/conv1/w/Adam/IsInitialized/VarIsInitializedOpVarIsInitializedOpsbb2/conv1/w/Adam*
_class
loc:@sbb2/conv1/w*
_output_shapes
: 

sbb2/conv1/w/Adam/AssignAssignVariableOpsbb2/conv1/w/Adam#sbb2/conv1/w/Adam/Initializer/zeros*&
 _has_manual_control_dependencies(*
dtype0
Ą
%sbb2/conv1/w/Adam/Read/ReadVariableOpReadVariableOpsbb2/conv1/w/Adam*
_class
loc:@sbb2/conv1/w*'
_output_shapes
:@*
dtype0
Ż
5sbb2/conv1/w/Adam_1/Initializer/zeros/shape_as_tensorConst*
_class
loc:@sbb2/conv1/w*
_output_shapes
:*
dtype0*%
valueB"         @   

+sbb2/conv1/w/Adam_1/Initializer/zeros/ConstConst*
_class
loc:@sbb2/conv1/w*
_output_shapes
: *
dtype0*
valueB
 *    
ä
%sbb2/conv1/w/Adam_1/Initializer/zerosFill5sbb2/conv1/w/Adam_1/Initializer/zeros/shape_as_tensor+sbb2/conv1/w/Adam_1/Initializer/zeros/Const*
T0*
_class
loc:@sbb2/conv1/w*'
_output_shapes
:@
Ź
sbb2/conv1/w/Adam_1VarHandleOp*
_class
loc:@sbb2/conv1/w*
_output_shapes
: *
dtype0*
shape:@*$
shared_namesbb2/conv1/w/Adam_1

4sbb2/conv1/w/Adam_1/IsInitialized/VarIsInitializedOpVarIsInitializedOpsbb2/conv1/w/Adam_1*
_class
loc:@sbb2/conv1/w*
_output_shapes
: 

sbb2/conv1/w/Adam_1/AssignAssignVariableOpsbb2/conv1/w/Adam_1%sbb2/conv1/w/Adam_1/Initializer/zeros*&
 _has_manual_control_dependencies(*
dtype0
Ľ
'sbb2/conv1/w/Adam_1/Read/ReadVariableOpReadVariableOpsbb2/conv1/w/Adam_1*
_class
loc:@sbb2/conv1/w*'
_output_shapes
:@*
dtype0

#sbb2/conv1/b/Adam/Initializer/zerosConst*
_class
loc:@sbb2/conv1/b*
_output_shapes
:@*
dtype0*
valueB@*    

sbb2/conv1/b/AdamVarHandleOp*
_class
loc:@sbb2/conv1/b*
_output_shapes
: *
dtype0*
shape:@*"
shared_namesbb2/conv1/b/Adam

2sbb2/conv1/b/Adam/IsInitialized/VarIsInitializedOpVarIsInitializedOpsbb2/conv1/b/Adam*
_class
loc:@sbb2/conv1/b*
_output_shapes
: 

sbb2/conv1/b/Adam/AssignAssignVariableOpsbb2/conv1/b/Adam#sbb2/conv1/b/Adam/Initializer/zeros*&
 _has_manual_control_dependencies(*
dtype0

%sbb2/conv1/b/Adam/Read/ReadVariableOpReadVariableOpsbb2/conv1/b/Adam*
_class
loc:@sbb2/conv1/b*
_output_shapes
:@*
dtype0

%sbb2/conv1/b/Adam_1/Initializer/zerosConst*
_class
loc:@sbb2/conv1/b*
_output_shapes
:@*
dtype0*
valueB@*    

sbb2/conv1/b/Adam_1VarHandleOp*
_class
loc:@sbb2/conv1/b*
_output_shapes
: *
dtype0*
shape:@*$
shared_namesbb2/conv1/b/Adam_1

4sbb2/conv1/b/Adam_1/IsInitialized/VarIsInitializedOpVarIsInitializedOpsbb2/conv1/b/Adam_1*
_class
loc:@sbb2/conv1/b*
_output_shapes
: 

sbb2/conv1/b/Adam_1/AssignAssignVariableOpsbb2/conv1/b/Adam_1%sbb2/conv1/b/Adam_1/Initializer/zeros*&
 _has_manual_control_dependencies(*
dtype0

'sbb2/conv1/b/Adam_1/Read/ReadVariableOpReadVariableOpsbb2/conv1/b/Adam_1*
_class
loc:@sbb2/conv1/b*
_output_shapes
:@*
dtype0
­
3sbb2/conv2/w/Adam/Initializer/zeros/shape_as_tensorConst*
_class
loc:@sbb2/conv2/w*
_output_shapes
:*
dtype0*%
valueB"      @   @   

)sbb2/conv2/w/Adam/Initializer/zeros/ConstConst*
_class
loc:@sbb2/conv2/w*
_output_shapes
: *
dtype0*
valueB
 *    
Ý
#sbb2/conv2/w/Adam/Initializer/zerosFill3sbb2/conv2/w/Adam/Initializer/zeros/shape_as_tensor)sbb2/conv2/w/Adam/Initializer/zeros/Const*
T0*
_class
loc:@sbb2/conv2/w*&
_output_shapes
:@@
§
sbb2/conv2/w/AdamVarHandleOp*
_class
loc:@sbb2/conv2/w*
_output_shapes
: *
dtype0*
shape:@@*"
shared_namesbb2/conv2/w/Adam

2sbb2/conv2/w/Adam/IsInitialized/VarIsInitializedOpVarIsInitializedOpsbb2/conv2/w/Adam*
_class
loc:@sbb2/conv2/w*
_output_shapes
: 

sbb2/conv2/w/Adam/AssignAssignVariableOpsbb2/conv2/w/Adam#sbb2/conv2/w/Adam/Initializer/zeros*&
 _has_manual_control_dependencies(*
dtype0
 
%sbb2/conv2/w/Adam/Read/ReadVariableOpReadVariableOpsbb2/conv2/w/Adam*
_class
loc:@sbb2/conv2/w*&
_output_shapes
:@@*
dtype0
Ż
5sbb2/conv2/w/Adam_1/Initializer/zeros/shape_as_tensorConst*
_class
loc:@sbb2/conv2/w*
_output_shapes
:*
dtype0*%
valueB"      @   @   

+sbb2/conv2/w/Adam_1/Initializer/zeros/ConstConst*
_class
loc:@sbb2/conv2/w*
_output_shapes
: *
dtype0*
valueB
 *    
ă
%sbb2/conv2/w/Adam_1/Initializer/zerosFill5sbb2/conv2/w/Adam_1/Initializer/zeros/shape_as_tensor+sbb2/conv2/w/Adam_1/Initializer/zeros/Const*
T0*
_class
loc:@sbb2/conv2/w*&
_output_shapes
:@@
Ť
sbb2/conv2/w/Adam_1VarHandleOp*
_class
loc:@sbb2/conv2/w*
_output_shapes
: *
dtype0*
shape:@@*$
shared_namesbb2/conv2/w/Adam_1

4sbb2/conv2/w/Adam_1/IsInitialized/VarIsInitializedOpVarIsInitializedOpsbb2/conv2/w/Adam_1*
_class
loc:@sbb2/conv2/w*
_output_shapes
: 

sbb2/conv2/w/Adam_1/AssignAssignVariableOpsbb2/conv2/w/Adam_1%sbb2/conv2/w/Adam_1/Initializer/zeros*&
 _has_manual_control_dependencies(*
dtype0
¤
'sbb2/conv2/w/Adam_1/Read/ReadVariableOpReadVariableOpsbb2/conv2/w/Adam_1*
_class
loc:@sbb2/conv2/w*&
_output_shapes
:@@*
dtype0

#sbb2/conv2/b/Adam/Initializer/zerosConst*
_class
loc:@sbb2/conv2/b*
_output_shapes
:@*
dtype0*
valueB@*    

sbb2/conv2/b/AdamVarHandleOp*
_class
loc:@sbb2/conv2/b*
_output_shapes
: *
dtype0*
shape:@*"
shared_namesbb2/conv2/b/Adam

2sbb2/conv2/b/Adam/IsInitialized/VarIsInitializedOpVarIsInitializedOpsbb2/conv2/b/Adam*
_class
loc:@sbb2/conv2/b*
_output_shapes
: 

sbb2/conv2/b/Adam/AssignAssignVariableOpsbb2/conv2/b/Adam#sbb2/conv2/b/Adam/Initializer/zeros*&
 _has_manual_control_dependencies(*
dtype0

%sbb2/conv2/b/Adam/Read/ReadVariableOpReadVariableOpsbb2/conv2/b/Adam*
_class
loc:@sbb2/conv2/b*
_output_shapes
:@*
dtype0

%sbb2/conv2/b/Adam_1/Initializer/zerosConst*
_class
loc:@sbb2/conv2/b*
_output_shapes
:@*
dtype0*
valueB@*    

sbb2/conv2/b/Adam_1VarHandleOp*
_class
loc:@sbb2/conv2/b*
_output_shapes
: *
dtype0*
shape:@*$
shared_namesbb2/conv2/b/Adam_1

4sbb2/conv2/b/Adam_1/IsInitialized/VarIsInitializedOpVarIsInitializedOpsbb2/conv2/b/Adam_1*
_class
loc:@sbb2/conv2/b*
_output_shapes
: 

sbb2/conv2/b/Adam_1/AssignAssignVariableOpsbb2/conv2/b/Adam_1%sbb2/conv2/b/Adam_1/Initializer/zeros*&
 _has_manual_control_dependencies(*
dtype0

'sbb2/conv2/b/Adam_1/Read/ReadVariableOpReadVariableOpsbb2/conv2/b/Adam_1*
_class
loc:@sbb2/conv2/b*
_output_shapes
:@*
dtype0
­
3sbb2/conv3/w/Adam/Initializer/zeros/shape_as_tensorConst*
_class
loc:@sbb2/conv3/w*
_output_shapes
:*
dtype0*%
valueB"      @   @   

)sbb2/conv3/w/Adam/Initializer/zeros/ConstConst*
_class
loc:@sbb2/conv3/w*
_output_shapes
: *
dtype0*
valueB
 *    
Ý
#sbb2/conv3/w/Adam/Initializer/zerosFill3sbb2/conv3/w/Adam/Initializer/zeros/shape_as_tensor)sbb2/conv3/w/Adam/Initializer/zeros/Const*
T0*
_class
loc:@sbb2/conv3/w*&
_output_shapes
:@@
§
sbb2/conv3/w/AdamVarHandleOp*
_class
loc:@sbb2/conv3/w*
_output_shapes
: *
dtype0*
shape:@@*"
shared_namesbb2/conv3/w/Adam

2sbb2/conv3/w/Adam/IsInitialized/VarIsInitializedOpVarIsInitializedOpsbb2/conv3/w/Adam*
_class
loc:@sbb2/conv3/w*
_output_shapes
: 

sbb2/conv3/w/Adam/AssignAssignVariableOpsbb2/conv3/w/Adam#sbb2/conv3/w/Adam/Initializer/zeros*&
 _has_manual_control_dependencies(*
dtype0
 
%sbb2/conv3/w/Adam/Read/ReadVariableOpReadVariableOpsbb2/conv3/w/Adam*
_class
loc:@sbb2/conv3/w*&
_output_shapes
:@@*
dtype0
Ż
5sbb2/conv3/w/Adam_1/Initializer/zeros/shape_as_tensorConst*
_class
loc:@sbb2/conv3/w*
_output_shapes
:*
dtype0*%
valueB"      @   @   

+sbb2/conv3/w/Adam_1/Initializer/zeros/ConstConst*
_class
loc:@sbb2/conv3/w*
_output_shapes
: *
dtype0*
valueB
 *    
ă
%sbb2/conv3/w/Adam_1/Initializer/zerosFill5sbb2/conv3/w/Adam_1/Initializer/zeros/shape_as_tensor+sbb2/conv3/w/Adam_1/Initializer/zeros/Const*
T0*
_class
loc:@sbb2/conv3/w*&
_output_shapes
:@@
Ť
sbb2/conv3/w/Adam_1VarHandleOp*
_class
loc:@sbb2/conv3/w*
_output_shapes
: *
dtype0*
shape:@@*$
shared_namesbb2/conv3/w/Adam_1

4sbb2/conv3/w/Adam_1/IsInitialized/VarIsInitializedOpVarIsInitializedOpsbb2/conv3/w/Adam_1*
_class
loc:@sbb2/conv3/w*
_output_shapes
: 

sbb2/conv3/w/Adam_1/AssignAssignVariableOpsbb2/conv3/w/Adam_1%sbb2/conv3/w/Adam_1/Initializer/zeros*&
 _has_manual_control_dependencies(*
dtype0
¤
'sbb2/conv3/w/Adam_1/Read/ReadVariableOpReadVariableOpsbb2/conv3/w/Adam_1*
_class
loc:@sbb2/conv3/w*&
_output_shapes
:@@*
dtype0

#sbb2/conv3/b/Adam/Initializer/zerosConst*
_class
loc:@sbb2/conv3/b*
_output_shapes
:@*
dtype0*
valueB@*    

sbb2/conv3/b/AdamVarHandleOp*
_class
loc:@sbb2/conv3/b*
_output_shapes
: *
dtype0*
shape:@*"
shared_namesbb2/conv3/b/Adam

2sbb2/conv3/b/Adam/IsInitialized/VarIsInitializedOpVarIsInitializedOpsbb2/conv3/b/Adam*
_class
loc:@sbb2/conv3/b*
_output_shapes
: 

sbb2/conv3/b/Adam/AssignAssignVariableOpsbb2/conv3/b/Adam#sbb2/conv3/b/Adam/Initializer/zeros*&
 _has_manual_control_dependencies(*
dtype0

%sbb2/conv3/b/Adam/Read/ReadVariableOpReadVariableOpsbb2/conv3/b/Adam*
_class
loc:@sbb2/conv3/b*
_output_shapes
:@*
dtype0

%sbb2/conv3/b/Adam_1/Initializer/zerosConst*
_class
loc:@sbb2/conv3/b*
_output_shapes
:@*
dtype0*
valueB@*    

sbb2/conv3/b/Adam_1VarHandleOp*
_class
loc:@sbb2/conv3/b*
_output_shapes
: *
dtype0*
shape:@*$
shared_namesbb2/conv3/b/Adam_1

4sbb2/conv3/b/Adam_1/IsInitialized/VarIsInitializedOpVarIsInitializedOpsbb2/conv3/b/Adam_1*
_class
loc:@sbb2/conv3/b*
_output_shapes
: 

sbb2/conv3/b/Adam_1/AssignAssignVariableOpsbb2/conv3/b/Adam_1%sbb2/conv3/b/Adam_1/Initializer/zeros*&
 _has_manual_control_dependencies(*
dtype0

'sbb2/conv3/b/Adam_1/Read/ReadVariableOpReadVariableOpsbb2/conv3/b/Adam_1*
_class
loc:@sbb2/conv3/b*
_output_shapes
:@*
dtype0
­
3sbb2/conv4/w/Adam/Initializer/zeros/shape_as_tensorConst*
_class
loc:@sbb2/conv4/w*
_output_shapes
:*
dtype0*%
valueB"      @      

)sbb2/conv4/w/Adam/Initializer/zeros/ConstConst*
_class
loc:@sbb2/conv4/w*
_output_shapes
: *
dtype0*
valueB
 *    
Ţ
#sbb2/conv4/w/Adam/Initializer/zerosFill3sbb2/conv4/w/Adam/Initializer/zeros/shape_as_tensor)sbb2/conv4/w/Adam/Initializer/zeros/Const*
T0*
_class
loc:@sbb2/conv4/w*'
_output_shapes
:@
¨
sbb2/conv4/w/AdamVarHandleOp*
_class
loc:@sbb2/conv4/w*
_output_shapes
: *
dtype0*
shape:@*"
shared_namesbb2/conv4/w/Adam

2sbb2/conv4/w/Adam/IsInitialized/VarIsInitializedOpVarIsInitializedOpsbb2/conv4/w/Adam*
_class
loc:@sbb2/conv4/w*
_output_shapes
: 

sbb2/conv4/w/Adam/AssignAssignVariableOpsbb2/conv4/w/Adam#sbb2/conv4/w/Adam/Initializer/zeros*&
 _has_manual_control_dependencies(*
dtype0
Ą
%sbb2/conv4/w/Adam/Read/ReadVariableOpReadVariableOpsbb2/conv4/w/Adam*
_class
loc:@sbb2/conv4/w*'
_output_shapes
:@*
dtype0
Ż
5sbb2/conv4/w/Adam_1/Initializer/zeros/shape_as_tensorConst*
_class
loc:@sbb2/conv4/w*
_output_shapes
:*
dtype0*%
valueB"      @      

+sbb2/conv4/w/Adam_1/Initializer/zeros/ConstConst*
_class
loc:@sbb2/conv4/w*
_output_shapes
: *
dtype0*
valueB
 *    
ä
%sbb2/conv4/w/Adam_1/Initializer/zerosFill5sbb2/conv4/w/Adam_1/Initializer/zeros/shape_as_tensor+sbb2/conv4/w/Adam_1/Initializer/zeros/Const*
T0*
_class
loc:@sbb2/conv4/w*'
_output_shapes
:@
Ź
sbb2/conv4/w/Adam_1VarHandleOp*
_class
loc:@sbb2/conv4/w*
_output_shapes
: *
dtype0*
shape:@*$
shared_namesbb2/conv4/w/Adam_1

4sbb2/conv4/w/Adam_1/IsInitialized/VarIsInitializedOpVarIsInitializedOpsbb2/conv4/w/Adam_1*
_class
loc:@sbb2/conv4/w*
_output_shapes
: 

sbb2/conv4/w/Adam_1/AssignAssignVariableOpsbb2/conv4/w/Adam_1%sbb2/conv4/w/Adam_1/Initializer/zeros*&
 _has_manual_control_dependencies(*
dtype0
Ľ
'sbb2/conv4/w/Adam_1/Read/ReadVariableOpReadVariableOpsbb2/conv4/w/Adam_1*
_class
loc:@sbb2/conv4/w*'
_output_shapes
:@*
dtype0

#sbb2/conv4/b/Adam/Initializer/zerosConst*
_class
loc:@sbb2/conv4/b*
_output_shapes	
:*
dtype0*
valueB*    

sbb2/conv4/b/AdamVarHandleOp*
_class
loc:@sbb2/conv4/b*
_output_shapes
: *
dtype0*
shape:*"
shared_namesbb2/conv4/b/Adam

2sbb2/conv4/b/Adam/IsInitialized/VarIsInitializedOpVarIsInitializedOpsbb2/conv4/b/Adam*
_class
loc:@sbb2/conv4/b*
_output_shapes
: 

sbb2/conv4/b/Adam/AssignAssignVariableOpsbb2/conv4/b/Adam#sbb2/conv4/b/Adam/Initializer/zeros*&
 _has_manual_control_dependencies(*
dtype0

%sbb2/conv4/b/Adam/Read/ReadVariableOpReadVariableOpsbb2/conv4/b/Adam*
_class
loc:@sbb2/conv4/b*
_output_shapes	
:*
dtype0

%sbb2/conv4/b/Adam_1/Initializer/zerosConst*
_class
loc:@sbb2/conv4/b*
_output_shapes	
:*
dtype0*
valueB*    
 
sbb2/conv4/b/Adam_1VarHandleOp*
_class
loc:@sbb2/conv4/b*
_output_shapes
: *
dtype0*
shape:*$
shared_namesbb2/conv4/b/Adam_1

4sbb2/conv4/b/Adam_1/IsInitialized/VarIsInitializedOpVarIsInitializedOpsbb2/conv4/b/Adam_1*
_class
loc:@sbb2/conv4/b*
_output_shapes
: 

sbb2/conv4/b/Adam_1/AssignAssignVariableOpsbb2/conv4/b/Adam_1%sbb2/conv4/b/Adam_1/Initializer/zeros*&
 _has_manual_control_dependencies(*
dtype0

'sbb2/conv4/b/Adam_1/Read/ReadVariableOpReadVariableOpsbb2/conv4/b/Adam_1*
_class
loc:@sbb2/conv4/b*
_output_shapes	
:*
dtype0
ˇ
5sbb2/batch_normalization/gamma/Adam/Initializer/zerosConst*1
_class'
%#loc:@sbb2/batch_normalization/gamma*
_output_shapes	
:*
dtype0*
valueB*    
Ň
#sbb2/batch_normalization/gamma/AdamVarHandleOp*1
_class'
%#loc:@sbb2/batch_normalization/gamma*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#sbb2/batch_normalization/gamma/Adam
Ę
Dsbb2/batch_normalization/gamma/Adam/IsInitialized/VarIsInitializedOpVarIsInitializedOp#sbb2/batch_normalization/gamma/Adam*1
_class'
%#loc:@sbb2/batch_normalization/gamma*
_output_shapes
: 
Ď
*sbb2/batch_normalization/gamma/Adam/AssignAssignVariableOp#sbb2/batch_normalization/gamma/Adam5sbb2/batch_normalization/gamma/Adam/Initializer/zeros*&
 _has_manual_control_dependencies(*
dtype0
Ë
7sbb2/batch_normalization/gamma/Adam/Read/ReadVariableOpReadVariableOp#sbb2/batch_normalization/gamma/Adam*1
_class'
%#loc:@sbb2/batch_normalization/gamma*
_output_shapes	
:*
dtype0
š
7sbb2/batch_normalization/gamma/Adam_1/Initializer/zerosConst*1
_class'
%#loc:@sbb2/batch_normalization/gamma*
_output_shapes	
:*
dtype0*
valueB*    
Ö
%sbb2/batch_normalization/gamma/Adam_1VarHandleOp*1
_class'
%#loc:@sbb2/batch_normalization/gamma*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%sbb2/batch_normalization/gamma/Adam_1
Î
Fsbb2/batch_normalization/gamma/Adam_1/IsInitialized/VarIsInitializedOpVarIsInitializedOp%sbb2/batch_normalization/gamma/Adam_1*1
_class'
%#loc:@sbb2/batch_normalization/gamma*
_output_shapes
: 
Ő
,sbb2/batch_normalization/gamma/Adam_1/AssignAssignVariableOp%sbb2/batch_normalization/gamma/Adam_17sbb2/batch_normalization/gamma/Adam_1/Initializer/zeros*&
 _has_manual_control_dependencies(*
dtype0
Ď
9sbb2/batch_normalization/gamma/Adam_1/Read/ReadVariableOpReadVariableOp%sbb2/batch_normalization/gamma/Adam_1*1
_class'
%#loc:@sbb2/batch_normalization/gamma*
_output_shapes	
:*
dtype0
ľ
4sbb2/batch_normalization/beta/Adam/Initializer/zerosConst*0
_class&
$"loc:@sbb2/batch_normalization/beta*
_output_shapes	
:*
dtype0*
valueB*    
Ď
"sbb2/batch_normalization/beta/AdamVarHandleOp*0
_class&
$"loc:@sbb2/batch_normalization/beta*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"sbb2/batch_normalization/beta/Adam
Ç
Csbb2/batch_normalization/beta/Adam/IsInitialized/VarIsInitializedOpVarIsInitializedOp"sbb2/batch_normalization/beta/Adam*0
_class&
$"loc:@sbb2/batch_normalization/beta*
_output_shapes
: 
Ě
)sbb2/batch_normalization/beta/Adam/AssignAssignVariableOp"sbb2/batch_normalization/beta/Adam4sbb2/batch_normalization/beta/Adam/Initializer/zeros*&
 _has_manual_control_dependencies(*
dtype0
Č
6sbb2/batch_normalization/beta/Adam/Read/ReadVariableOpReadVariableOp"sbb2/batch_normalization/beta/Adam*0
_class&
$"loc:@sbb2/batch_normalization/beta*
_output_shapes	
:*
dtype0
ˇ
6sbb2/batch_normalization/beta/Adam_1/Initializer/zerosConst*0
_class&
$"loc:@sbb2/batch_normalization/beta*
_output_shapes	
:*
dtype0*
valueB*    
Ó
$sbb2/batch_normalization/beta/Adam_1VarHandleOp*0
_class&
$"loc:@sbb2/batch_normalization/beta*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$sbb2/batch_normalization/beta/Adam_1
Ë
Esbb2/batch_normalization/beta/Adam_1/IsInitialized/VarIsInitializedOpVarIsInitializedOp$sbb2/batch_normalization/beta/Adam_1*0
_class&
$"loc:@sbb2/batch_normalization/beta*
_output_shapes
: 
Ň
+sbb2/batch_normalization/beta/Adam_1/AssignAssignVariableOp$sbb2/batch_normalization/beta/Adam_16sbb2/batch_normalization/beta/Adam_1/Initializer/zeros*&
 _has_manual_control_dependencies(*
dtype0
Ě
8sbb2/batch_normalization/beta/Adam_1/Read/ReadVariableOpReadVariableOp$sbb2/batch_normalization/beta/Adam_1*0
_class&
$"loc:@sbb2/batch_normalization/beta*
_output_shapes	
:*
dtype0
­
3sbb3/conv1/w/Adam/Initializer/zeros/shape_as_tensorConst*
_class
loc:@sbb3/conv1/w*
_output_shapes
:*
dtype0*%
valueB"         @   

)sbb3/conv1/w/Adam/Initializer/zeros/ConstConst*
_class
loc:@sbb3/conv1/w*
_output_shapes
: *
dtype0*
valueB
 *    
Ţ
#sbb3/conv1/w/Adam/Initializer/zerosFill3sbb3/conv1/w/Adam/Initializer/zeros/shape_as_tensor)sbb3/conv1/w/Adam/Initializer/zeros/Const*
T0*
_class
loc:@sbb3/conv1/w*'
_output_shapes
:@
¨
sbb3/conv1/w/AdamVarHandleOp*
_class
loc:@sbb3/conv1/w*
_output_shapes
: *
dtype0*
shape:@*"
shared_namesbb3/conv1/w/Adam

2sbb3/conv1/w/Adam/IsInitialized/VarIsInitializedOpVarIsInitializedOpsbb3/conv1/w/Adam*
_class
loc:@sbb3/conv1/w*
_output_shapes
: 

sbb3/conv1/w/Adam/AssignAssignVariableOpsbb3/conv1/w/Adam#sbb3/conv1/w/Adam/Initializer/zeros*&
 _has_manual_control_dependencies(*
dtype0
Ą
%sbb3/conv1/w/Adam/Read/ReadVariableOpReadVariableOpsbb3/conv1/w/Adam*
_class
loc:@sbb3/conv1/w*'
_output_shapes
:@*
dtype0
Ż
5sbb3/conv1/w/Adam_1/Initializer/zeros/shape_as_tensorConst*
_class
loc:@sbb3/conv1/w*
_output_shapes
:*
dtype0*%
valueB"         @   

+sbb3/conv1/w/Adam_1/Initializer/zeros/ConstConst*
_class
loc:@sbb3/conv1/w*
_output_shapes
: *
dtype0*
valueB
 *    
ä
%sbb3/conv1/w/Adam_1/Initializer/zerosFill5sbb3/conv1/w/Adam_1/Initializer/zeros/shape_as_tensor+sbb3/conv1/w/Adam_1/Initializer/zeros/Const*
T0*
_class
loc:@sbb3/conv1/w*'
_output_shapes
:@
Ź
sbb3/conv1/w/Adam_1VarHandleOp*
_class
loc:@sbb3/conv1/w*
_output_shapes
: *
dtype0*
shape:@*$
shared_namesbb3/conv1/w/Adam_1

4sbb3/conv1/w/Adam_1/IsInitialized/VarIsInitializedOpVarIsInitializedOpsbb3/conv1/w/Adam_1*
_class
loc:@sbb3/conv1/w*
_output_shapes
: 

sbb3/conv1/w/Adam_1/AssignAssignVariableOpsbb3/conv1/w/Adam_1%sbb3/conv1/w/Adam_1/Initializer/zeros*&
 _has_manual_control_dependencies(*
dtype0
Ľ
'sbb3/conv1/w/Adam_1/Read/ReadVariableOpReadVariableOpsbb3/conv1/w/Adam_1*
_class
loc:@sbb3/conv1/w*'
_output_shapes
:@*
dtype0

#sbb3/conv1/b/Adam/Initializer/zerosConst*
_class
loc:@sbb3/conv1/b*
_output_shapes
:@*
dtype0*
valueB@*    

sbb3/conv1/b/AdamVarHandleOp*
_class
loc:@sbb3/conv1/b*
_output_shapes
: *
dtype0*
shape:@*"
shared_namesbb3/conv1/b/Adam

2sbb3/conv1/b/Adam/IsInitialized/VarIsInitializedOpVarIsInitializedOpsbb3/conv1/b/Adam*
_class
loc:@sbb3/conv1/b*
_output_shapes
: 

sbb3/conv1/b/Adam/AssignAssignVariableOpsbb3/conv1/b/Adam#sbb3/conv1/b/Adam/Initializer/zeros*&
 _has_manual_control_dependencies(*
dtype0

%sbb3/conv1/b/Adam/Read/ReadVariableOpReadVariableOpsbb3/conv1/b/Adam*
_class
loc:@sbb3/conv1/b*
_output_shapes
:@*
dtype0

%sbb3/conv1/b/Adam_1/Initializer/zerosConst*
_class
loc:@sbb3/conv1/b*
_output_shapes
:@*
dtype0*
valueB@*    

sbb3/conv1/b/Adam_1VarHandleOp*
_class
loc:@sbb3/conv1/b*
_output_shapes
: *
dtype0*
shape:@*$
shared_namesbb3/conv1/b/Adam_1

4sbb3/conv1/b/Adam_1/IsInitialized/VarIsInitializedOpVarIsInitializedOpsbb3/conv1/b/Adam_1*
_class
loc:@sbb3/conv1/b*
_output_shapes
: 

sbb3/conv1/b/Adam_1/AssignAssignVariableOpsbb3/conv1/b/Adam_1%sbb3/conv1/b/Adam_1/Initializer/zeros*&
 _has_manual_control_dependencies(*
dtype0

'sbb3/conv1/b/Adam_1/Read/ReadVariableOpReadVariableOpsbb3/conv1/b/Adam_1*
_class
loc:@sbb3/conv1/b*
_output_shapes
:@*
dtype0
­
3sbb3/conv2/w/Adam/Initializer/zeros/shape_as_tensorConst*
_class
loc:@sbb3/conv2/w*
_output_shapes
:*
dtype0*%
valueB"      @   @   

)sbb3/conv2/w/Adam/Initializer/zeros/ConstConst*
_class
loc:@sbb3/conv2/w*
_output_shapes
: *
dtype0*
valueB
 *    
Ý
#sbb3/conv2/w/Adam/Initializer/zerosFill3sbb3/conv2/w/Adam/Initializer/zeros/shape_as_tensor)sbb3/conv2/w/Adam/Initializer/zeros/Const*
T0*
_class
loc:@sbb3/conv2/w*&
_output_shapes
:@@
§
sbb3/conv2/w/AdamVarHandleOp*
_class
loc:@sbb3/conv2/w*
_output_shapes
: *
dtype0*
shape:@@*"
shared_namesbb3/conv2/w/Adam

2sbb3/conv2/w/Adam/IsInitialized/VarIsInitializedOpVarIsInitializedOpsbb3/conv2/w/Adam*
_class
loc:@sbb3/conv2/w*
_output_shapes
: 

sbb3/conv2/w/Adam/AssignAssignVariableOpsbb3/conv2/w/Adam#sbb3/conv2/w/Adam/Initializer/zeros*&
 _has_manual_control_dependencies(*
dtype0
 
%sbb3/conv2/w/Adam/Read/ReadVariableOpReadVariableOpsbb3/conv2/w/Adam*
_class
loc:@sbb3/conv2/w*&
_output_shapes
:@@*
dtype0
Ż
5sbb3/conv2/w/Adam_1/Initializer/zeros/shape_as_tensorConst*
_class
loc:@sbb3/conv2/w*
_output_shapes
:*
dtype0*%
valueB"      @   @   

+sbb3/conv2/w/Adam_1/Initializer/zeros/ConstConst*
_class
loc:@sbb3/conv2/w*
_output_shapes
: *
dtype0*
valueB
 *    
ă
%sbb3/conv2/w/Adam_1/Initializer/zerosFill5sbb3/conv2/w/Adam_1/Initializer/zeros/shape_as_tensor+sbb3/conv2/w/Adam_1/Initializer/zeros/Const*
T0*
_class
loc:@sbb3/conv2/w*&
_output_shapes
:@@
Ť
sbb3/conv2/w/Adam_1VarHandleOp*
_class
loc:@sbb3/conv2/w*
_output_shapes
: *
dtype0*
shape:@@*$
shared_namesbb3/conv2/w/Adam_1

4sbb3/conv2/w/Adam_1/IsInitialized/VarIsInitializedOpVarIsInitializedOpsbb3/conv2/w/Adam_1*
_class
loc:@sbb3/conv2/w*
_output_shapes
: 

sbb3/conv2/w/Adam_1/AssignAssignVariableOpsbb3/conv2/w/Adam_1%sbb3/conv2/w/Adam_1/Initializer/zeros*&
 _has_manual_control_dependencies(*
dtype0
¤
'sbb3/conv2/w/Adam_1/Read/ReadVariableOpReadVariableOpsbb3/conv2/w/Adam_1*
_class
loc:@sbb3/conv2/w*&
_output_shapes
:@@*
dtype0

#sbb3/conv2/b/Adam/Initializer/zerosConst*
_class
loc:@sbb3/conv2/b*
_output_shapes
:@*
dtype0*
valueB@*    

sbb3/conv2/b/AdamVarHandleOp*
_class
loc:@sbb3/conv2/b*
_output_shapes
: *
dtype0*
shape:@*"
shared_namesbb3/conv2/b/Adam

2sbb3/conv2/b/Adam/IsInitialized/VarIsInitializedOpVarIsInitializedOpsbb3/conv2/b/Adam*
_class
loc:@sbb3/conv2/b*
_output_shapes
: 

sbb3/conv2/b/Adam/AssignAssignVariableOpsbb3/conv2/b/Adam#sbb3/conv2/b/Adam/Initializer/zeros*&
 _has_manual_control_dependencies(*
dtype0

%sbb3/conv2/b/Adam/Read/ReadVariableOpReadVariableOpsbb3/conv2/b/Adam*
_class
loc:@sbb3/conv2/b*
_output_shapes
:@*
dtype0

%sbb3/conv2/b/Adam_1/Initializer/zerosConst*
_class
loc:@sbb3/conv2/b*
_output_shapes
:@*
dtype0*
valueB@*    

sbb3/conv2/b/Adam_1VarHandleOp*
_class
loc:@sbb3/conv2/b*
_output_shapes
: *
dtype0*
shape:@*$
shared_namesbb3/conv2/b/Adam_1

4sbb3/conv2/b/Adam_1/IsInitialized/VarIsInitializedOpVarIsInitializedOpsbb3/conv2/b/Adam_1*
_class
loc:@sbb3/conv2/b*
_output_shapes
: 

sbb3/conv2/b/Adam_1/AssignAssignVariableOpsbb3/conv2/b/Adam_1%sbb3/conv2/b/Adam_1/Initializer/zeros*&
 _has_manual_control_dependencies(*
dtype0

'sbb3/conv2/b/Adam_1/Read/ReadVariableOpReadVariableOpsbb3/conv2/b/Adam_1*
_class
loc:@sbb3/conv2/b*
_output_shapes
:@*
dtype0
­
3sbb3/conv3/w/Adam/Initializer/zeros/shape_as_tensorConst*
_class
loc:@sbb3/conv3/w*
_output_shapes
:*
dtype0*%
valueB"      @   @   

)sbb3/conv3/w/Adam/Initializer/zeros/ConstConst*
_class
loc:@sbb3/conv3/w*
_output_shapes
: *
dtype0*
valueB
 *    
Ý
#sbb3/conv3/w/Adam/Initializer/zerosFill3sbb3/conv3/w/Adam/Initializer/zeros/shape_as_tensor)sbb3/conv3/w/Adam/Initializer/zeros/Const*
T0*
_class
loc:@sbb3/conv3/w*&
_output_shapes
:@@
§
sbb3/conv3/w/AdamVarHandleOp*
_class
loc:@sbb3/conv3/w*
_output_shapes
: *
dtype0*
shape:@@*"
shared_namesbb3/conv3/w/Adam

2sbb3/conv3/w/Adam/IsInitialized/VarIsInitializedOpVarIsInitializedOpsbb3/conv3/w/Adam*
_class
loc:@sbb3/conv3/w*
_output_shapes
: 

sbb3/conv3/w/Adam/AssignAssignVariableOpsbb3/conv3/w/Adam#sbb3/conv3/w/Adam/Initializer/zeros*&
 _has_manual_control_dependencies(*
dtype0
 
%sbb3/conv3/w/Adam/Read/ReadVariableOpReadVariableOpsbb3/conv3/w/Adam*
_class
loc:@sbb3/conv3/w*&
_output_shapes
:@@*
dtype0
Ż
5sbb3/conv3/w/Adam_1/Initializer/zeros/shape_as_tensorConst*
_class
loc:@sbb3/conv3/w*
_output_shapes
:*
dtype0*%
valueB"      @   @   

+sbb3/conv3/w/Adam_1/Initializer/zeros/ConstConst*
_class
loc:@sbb3/conv3/w*
_output_shapes
: *
dtype0*
valueB
 *    
ă
%sbb3/conv3/w/Adam_1/Initializer/zerosFill5sbb3/conv3/w/Adam_1/Initializer/zeros/shape_as_tensor+sbb3/conv3/w/Adam_1/Initializer/zeros/Const*
T0*
_class
loc:@sbb3/conv3/w*&
_output_shapes
:@@
Ť
sbb3/conv3/w/Adam_1VarHandleOp*
_class
loc:@sbb3/conv3/w*
_output_shapes
: *
dtype0*
shape:@@*$
shared_namesbb3/conv3/w/Adam_1

4sbb3/conv3/w/Adam_1/IsInitialized/VarIsInitializedOpVarIsInitializedOpsbb3/conv3/w/Adam_1*
_class
loc:@sbb3/conv3/w*
_output_shapes
: 

sbb3/conv3/w/Adam_1/AssignAssignVariableOpsbb3/conv3/w/Adam_1%sbb3/conv3/w/Adam_1/Initializer/zeros*&
 _has_manual_control_dependencies(*
dtype0
¤
'sbb3/conv3/w/Adam_1/Read/ReadVariableOpReadVariableOpsbb3/conv3/w/Adam_1*
_class
loc:@sbb3/conv3/w*&
_output_shapes
:@@*
dtype0

#sbb3/conv3/b/Adam/Initializer/zerosConst*
_class
loc:@sbb3/conv3/b*
_output_shapes
:@*
dtype0*
valueB@*    

sbb3/conv3/b/AdamVarHandleOp*
_class
loc:@sbb3/conv3/b*
_output_shapes
: *
dtype0*
shape:@*"
shared_namesbb3/conv3/b/Adam

2sbb3/conv3/b/Adam/IsInitialized/VarIsInitializedOpVarIsInitializedOpsbb3/conv3/b/Adam*
_class
loc:@sbb3/conv3/b*
_output_shapes
: 

sbb3/conv3/b/Adam/AssignAssignVariableOpsbb3/conv3/b/Adam#sbb3/conv3/b/Adam/Initializer/zeros*&
 _has_manual_control_dependencies(*
dtype0

%sbb3/conv3/b/Adam/Read/ReadVariableOpReadVariableOpsbb3/conv3/b/Adam*
_class
loc:@sbb3/conv3/b*
_output_shapes
:@*
dtype0

%sbb3/conv3/b/Adam_1/Initializer/zerosConst*
_class
loc:@sbb3/conv3/b*
_output_shapes
:@*
dtype0*
valueB@*    

sbb3/conv3/b/Adam_1VarHandleOp*
_class
loc:@sbb3/conv3/b*
_output_shapes
: *
dtype0*
shape:@*$
shared_namesbb3/conv3/b/Adam_1

4sbb3/conv3/b/Adam_1/IsInitialized/VarIsInitializedOpVarIsInitializedOpsbb3/conv3/b/Adam_1*
_class
loc:@sbb3/conv3/b*
_output_shapes
: 

sbb3/conv3/b/Adam_1/AssignAssignVariableOpsbb3/conv3/b/Adam_1%sbb3/conv3/b/Adam_1/Initializer/zeros*&
 _has_manual_control_dependencies(*
dtype0

'sbb3/conv3/b/Adam_1/Read/ReadVariableOpReadVariableOpsbb3/conv3/b/Adam_1*
_class
loc:@sbb3/conv3/b*
_output_shapes
:@*
dtype0
­
3sbb3/conv4/w/Adam/Initializer/zeros/shape_as_tensorConst*
_class
loc:@sbb3/conv4/w*
_output_shapes
:*
dtype0*%
valueB"      @      

)sbb3/conv4/w/Adam/Initializer/zeros/ConstConst*
_class
loc:@sbb3/conv4/w*
_output_shapes
: *
dtype0*
valueB
 *    
Ţ
#sbb3/conv4/w/Adam/Initializer/zerosFill3sbb3/conv4/w/Adam/Initializer/zeros/shape_as_tensor)sbb3/conv4/w/Adam/Initializer/zeros/Const*
T0*
_class
loc:@sbb3/conv4/w*'
_output_shapes
:@
¨
sbb3/conv4/w/AdamVarHandleOp*
_class
loc:@sbb3/conv4/w*
_output_shapes
: *
dtype0*
shape:@*"
shared_namesbb3/conv4/w/Adam

2sbb3/conv4/w/Adam/IsInitialized/VarIsInitializedOpVarIsInitializedOpsbb3/conv4/w/Adam*
_class
loc:@sbb3/conv4/w*
_output_shapes
: 

sbb3/conv4/w/Adam/AssignAssignVariableOpsbb3/conv4/w/Adam#sbb3/conv4/w/Adam/Initializer/zeros*&
 _has_manual_control_dependencies(*
dtype0
Ą
%sbb3/conv4/w/Adam/Read/ReadVariableOpReadVariableOpsbb3/conv4/w/Adam*
_class
loc:@sbb3/conv4/w*'
_output_shapes
:@*
dtype0
Ż
5sbb3/conv4/w/Adam_1/Initializer/zeros/shape_as_tensorConst*
_class
loc:@sbb3/conv4/w*
_output_shapes
:*
dtype0*%
valueB"      @      

+sbb3/conv4/w/Adam_1/Initializer/zeros/ConstConst*
_class
loc:@sbb3/conv4/w*
_output_shapes
: *
dtype0*
valueB
 *    
ä
%sbb3/conv4/w/Adam_1/Initializer/zerosFill5sbb3/conv4/w/Adam_1/Initializer/zeros/shape_as_tensor+sbb3/conv4/w/Adam_1/Initializer/zeros/Const*
T0*
_class
loc:@sbb3/conv4/w*'
_output_shapes
:@
Ź
sbb3/conv4/w/Adam_1VarHandleOp*
_class
loc:@sbb3/conv4/w*
_output_shapes
: *
dtype0*
shape:@*$
shared_namesbb3/conv4/w/Adam_1

4sbb3/conv4/w/Adam_1/IsInitialized/VarIsInitializedOpVarIsInitializedOpsbb3/conv4/w/Adam_1*
_class
loc:@sbb3/conv4/w*
_output_shapes
: 

sbb3/conv4/w/Adam_1/AssignAssignVariableOpsbb3/conv4/w/Adam_1%sbb3/conv4/w/Adam_1/Initializer/zeros*&
 _has_manual_control_dependencies(*
dtype0
Ľ
'sbb3/conv4/w/Adam_1/Read/ReadVariableOpReadVariableOpsbb3/conv4/w/Adam_1*
_class
loc:@sbb3/conv4/w*'
_output_shapes
:@*
dtype0

#sbb3/conv4/b/Adam/Initializer/zerosConst*
_class
loc:@sbb3/conv4/b*
_output_shapes	
:*
dtype0*
valueB*    

sbb3/conv4/b/AdamVarHandleOp*
_class
loc:@sbb3/conv4/b*
_output_shapes
: *
dtype0*
shape:*"
shared_namesbb3/conv4/b/Adam

2sbb3/conv4/b/Adam/IsInitialized/VarIsInitializedOpVarIsInitializedOpsbb3/conv4/b/Adam*
_class
loc:@sbb3/conv4/b*
_output_shapes
: 

sbb3/conv4/b/Adam/AssignAssignVariableOpsbb3/conv4/b/Adam#sbb3/conv4/b/Adam/Initializer/zeros*&
 _has_manual_control_dependencies(*
dtype0

%sbb3/conv4/b/Adam/Read/ReadVariableOpReadVariableOpsbb3/conv4/b/Adam*
_class
loc:@sbb3/conv4/b*
_output_shapes	
:*
dtype0

%sbb3/conv4/b/Adam_1/Initializer/zerosConst*
_class
loc:@sbb3/conv4/b*
_output_shapes	
:*
dtype0*
valueB*    
 
sbb3/conv4/b/Adam_1VarHandleOp*
_class
loc:@sbb3/conv4/b*
_output_shapes
: *
dtype0*
shape:*$
shared_namesbb3/conv4/b/Adam_1

4sbb3/conv4/b/Adam_1/IsInitialized/VarIsInitializedOpVarIsInitializedOpsbb3/conv4/b/Adam_1*
_class
loc:@sbb3/conv4/b*
_output_shapes
: 

sbb3/conv4/b/Adam_1/AssignAssignVariableOpsbb3/conv4/b/Adam_1%sbb3/conv4/b/Adam_1/Initializer/zeros*&
 _has_manual_control_dependencies(*
dtype0

'sbb3/conv4/b/Adam_1/Read/ReadVariableOpReadVariableOpsbb3/conv4/b/Adam_1*
_class
loc:@sbb3/conv4/b*
_output_shapes	
:*
dtype0
ˇ
5sbb3/batch_normalization/gamma/Adam/Initializer/zerosConst*1
_class'
%#loc:@sbb3/batch_normalization/gamma*
_output_shapes	
:*
dtype0*
valueB*    
Ň
#sbb3/batch_normalization/gamma/AdamVarHandleOp*1
_class'
%#loc:@sbb3/batch_normalization/gamma*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#sbb3/batch_normalization/gamma/Adam
Ę
Dsbb3/batch_normalization/gamma/Adam/IsInitialized/VarIsInitializedOpVarIsInitializedOp#sbb3/batch_normalization/gamma/Adam*1
_class'
%#loc:@sbb3/batch_normalization/gamma*
_output_shapes
: 
Ď
*sbb3/batch_normalization/gamma/Adam/AssignAssignVariableOp#sbb3/batch_normalization/gamma/Adam5sbb3/batch_normalization/gamma/Adam/Initializer/zeros*&
 _has_manual_control_dependencies(*
dtype0
Ë
7sbb3/batch_normalization/gamma/Adam/Read/ReadVariableOpReadVariableOp#sbb3/batch_normalization/gamma/Adam*1
_class'
%#loc:@sbb3/batch_normalization/gamma*
_output_shapes	
:*
dtype0
š
7sbb3/batch_normalization/gamma/Adam_1/Initializer/zerosConst*1
_class'
%#loc:@sbb3/batch_normalization/gamma*
_output_shapes	
:*
dtype0*
valueB*    
Ö
%sbb3/batch_normalization/gamma/Adam_1VarHandleOp*1
_class'
%#loc:@sbb3/batch_normalization/gamma*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%sbb3/batch_normalization/gamma/Adam_1
Î
Fsbb3/batch_normalization/gamma/Adam_1/IsInitialized/VarIsInitializedOpVarIsInitializedOp%sbb3/batch_normalization/gamma/Adam_1*1
_class'
%#loc:@sbb3/batch_normalization/gamma*
_output_shapes
: 
Ő
,sbb3/batch_normalization/gamma/Adam_1/AssignAssignVariableOp%sbb3/batch_normalization/gamma/Adam_17sbb3/batch_normalization/gamma/Adam_1/Initializer/zeros*&
 _has_manual_control_dependencies(*
dtype0
Ď
9sbb3/batch_normalization/gamma/Adam_1/Read/ReadVariableOpReadVariableOp%sbb3/batch_normalization/gamma/Adam_1*1
_class'
%#loc:@sbb3/batch_normalization/gamma*
_output_shapes	
:*
dtype0
ľ
4sbb3/batch_normalization/beta/Adam/Initializer/zerosConst*0
_class&
$"loc:@sbb3/batch_normalization/beta*
_output_shapes	
:*
dtype0*
valueB*    
Ď
"sbb3/batch_normalization/beta/AdamVarHandleOp*0
_class&
$"loc:@sbb3/batch_normalization/beta*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"sbb3/batch_normalization/beta/Adam
Ç
Csbb3/batch_normalization/beta/Adam/IsInitialized/VarIsInitializedOpVarIsInitializedOp"sbb3/batch_normalization/beta/Adam*0
_class&
$"loc:@sbb3/batch_normalization/beta*
_output_shapes
: 
Ě
)sbb3/batch_normalization/beta/Adam/AssignAssignVariableOp"sbb3/batch_normalization/beta/Adam4sbb3/batch_normalization/beta/Adam/Initializer/zeros*&
 _has_manual_control_dependencies(*
dtype0
Č
6sbb3/batch_normalization/beta/Adam/Read/ReadVariableOpReadVariableOp"sbb3/batch_normalization/beta/Adam*0
_class&
$"loc:@sbb3/batch_normalization/beta*
_output_shapes	
:*
dtype0
ˇ
6sbb3/batch_normalization/beta/Adam_1/Initializer/zerosConst*0
_class&
$"loc:@sbb3/batch_normalization/beta*
_output_shapes	
:*
dtype0*
valueB*    
Ó
$sbb3/batch_normalization/beta/Adam_1VarHandleOp*0
_class&
$"loc:@sbb3/batch_normalization/beta*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$sbb3/batch_normalization/beta/Adam_1
Ë
Esbb3/batch_normalization/beta/Adam_1/IsInitialized/VarIsInitializedOpVarIsInitializedOp$sbb3/batch_normalization/beta/Adam_1*0
_class&
$"loc:@sbb3/batch_normalization/beta*
_output_shapes
: 
Ň
+sbb3/batch_normalization/beta/Adam_1/AssignAssignVariableOp$sbb3/batch_normalization/beta/Adam_16sbb3/batch_normalization/beta/Adam_1/Initializer/zeros*&
 _has_manual_control_dependencies(*
dtype0
Ě
8sbb3/batch_normalization/beta/Adam_1/Read/ReadVariableOpReadVariableOp$sbb3/batch_normalization/beta/Adam_1*0
_class&
$"loc:@sbb3/batch_normalization/beta*
_output_shapes	
:*
dtype0
§
0conv_d1/w/Adam/Initializer/zeros/shape_as_tensorConst*
_class
loc:@conv_d1/w*
_output_shapes
:*
dtype0*%
valueB"            

&conv_d1/w/Adam/Initializer/zeros/ConstConst*
_class
loc:@conv_d1/w*
_output_shapes
: *
dtype0*
valueB
 *    
Ó
 conv_d1/w/Adam/Initializer/zerosFill0conv_d1/w/Adam/Initializer/zeros/shape_as_tensor&conv_d1/w/Adam/Initializer/zeros/Const*
T0*
_class
loc:@conv_d1/w*(
_output_shapes
:
 
conv_d1/w/AdamVarHandleOp*
_class
loc:@conv_d1/w*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv_d1/w/Adam

/conv_d1/w/Adam/IsInitialized/VarIsInitializedOpVarIsInitializedOpconv_d1/w/Adam*
_class
loc:@conv_d1/w*
_output_shapes
: 

conv_d1/w/Adam/AssignAssignVariableOpconv_d1/w/Adam conv_d1/w/Adam/Initializer/zeros*&
 _has_manual_control_dependencies(*
dtype0

"conv_d1/w/Adam/Read/ReadVariableOpReadVariableOpconv_d1/w/Adam*
_class
loc:@conv_d1/w*(
_output_shapes
:*
dtype0
Š
2conv_d1/w/Adam_1/Initializer/zeros/shape_as_tensorConst*
_class
loc:@conv_d1/w*
_output_shapes
:*
dtype0*%
valueB"            

(conv_d1/w/Adam_1/Initializer/zeros/ConstConst*
_class
loc:@conv_d1/w*
_output_shapes
: *
dtype0*
valueB
 *    
Ů
"conv_d1/w/Adam_1/Initializer/zerosFill2conv_d1/w/Adam_1/Initializer/zeros/shape_as_tensor(conv_d1/w/Adam_1/Initializer/zeros/Const*
T0*
_class
loc:@conv_d1/w*(
_output_shapes
:
¤
conv_d1/w/Adam_1VarHandleOp*
_class
loc:@conv_d1/w*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv_d1/w/Adam_1

1conv_d1/w/Adam_1/IsInitialized/VarIsInitializedOpVarIsInitializedOpconv_d1/w/Adam_1*
_class
loc:@conv_d1/w*
_output_shapes
: 

conv_d1/w/Adam_1/AssignAssignVariableOpconv_d1/w/Adam_1"conv_d1/w/Adam_1/Initializer/zeros*&
 _has_manual_control_dependencies(*
dtype0

$conv_d1/w/Adam_1/Read/ReadVariableOpReadVariableOpconv_d1/w/Adam_1*
_class
loc:@conv_d1/w*(
_output_shapes
:*
dtype0

 conv_d1/b/Adam/Initializer/zerosConst*
_class
loc:@conv_d1/b*
_output_shapes	
:*
dtype0*
valueB*    

conv_d1/b/AdamVarHandleOp*
_class
loc:@conv_d1/b*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv_d1/b/Adam

/conv_d1/b/Adam/IsInitialized/VarIsInitializedOpVarIsInitializedOpconv_d1/b/Adam*
_class
loc:@conv_d1/b*
_output_shapes
: 

conv_d1/b/Adam/AssignAssignVariableOpconv_d1/b/Adam conv_d1/b/Adam/Initializer/zeros*&
 _has_manual_control_dependencies(*
dtype0

"conv_d1/b/Adam/Read/ReadVariableOpReadVariableOpconv_d1/b/Adam*
_class
loc:@conv_d1/b*
_output_shapes	
:*
dtype0

"conv_d1/b/Adam_1/Initializer/zerosConst*
_class
loc:@conv_d1/b*
_output_shapes	
:*
dtype0*
valueB*    

conv_d1/b/Adam_1VarHandleOp*
_class
loc:@conv_d1/b*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv_d1/b/Adam_1

1conv_d1/b/Adam_1/IsInitialized/VarIsInitializedOpVarIsInitializedOpconv_d1/b/Adam_1*
_class
loc:@conv_d1/b*
_output_shapes
: 

conv_d1/b/Adam_1/AssignAssignVariableOpconv_d1/b/Adam_1"conv_d1/b/Adam_1/Initializer/zeros*&
 _has_manual_control_dependencies(*
dtype0

$conv_d1/b/Adam_1/Read/ReadVariableOpReadVariableOpconv_d1/b/Adam_1*
_class
loc:@conv_d1/b*
_output_shapes	
:*
dtype0
ą
2batch_normalization_1/gamma/Adam/Initializer/zerosConst*.
_class$
" loc:@batch_normalization_1/gamma*
_output_shapes	
:*
dtype0*
valueB*    
É
 batch_normalization_1/gamma/AdamVarHandleOp*.
_class$
" loc:@batch_normalization_1/gamma*
_output_shapes
: *
dtype0*
shape:*1
shared_name" batch_normalization_1/gamma/Adam
Á
Abatch_normalization_1/gamma/Adam/IsInitialized/VarIsInitializedOpVarIsInitializedOp batch_normalization_1/gamma/Adam*.
_class$
" loc:@batch_normalization_1/gamma*
_output_shapes
: 
Ć
'batch_normalization_1/gamma/Adam/AssignAssignVariableOp batch_normalization_1/gamma/Adam2batch_normalization_1/gamma/Adam/Initializer/zeros*&
 _has_manual_control_dependencies(*
dtype0
Â
4batch_normalization_1/gamma/Adam/Read/ReadVariableOpReadVariableOp batch_normalization_1/gamma/Adam*.
_class$
" loc:@batch_normalization_1/gamma*
_output_shapes	
:*
dtype0
ł
4batch_normalization_1/gamma/Adam_1/Initializer/zerosConst*.
_class$
" loc:@batch_normalization_1/gamma*
_output_shapes	
:*
dtype0*
valueB*    
Í
"batch_normalization_1/gamma/Adam_1VarHandleOp*.
_class$
" loc:@batch_normalization_1/gamma*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"batch_normalization_1/gamma/Adam_1
Ĺ
Cbatch_normalization_1/gamma/Adam_1/IsInitialized/VarIsInitializedOpVarIsInitializedOp"batch_normalization_1/gamma/Adam_1*.
_class$
" loc:@batch_normalization_1/gamma*
_output_shapes
: 
Ě
)batch_normalization_1/gamma/Adam_1/AssignAssignVariableOp"batch_normalization_1/gamma/Adam_14batch_normalization_1/gamma/Adam_1/Initializer/zeros*&
 _has_manual_control_dependencies(*
dtype0
Ć
6batch_normalization_1/gamma/Adam_1/Read/ReadVariableOpReadVariableOp"batch_normalization_1/gamma/Adam_1*.
_class$
" loc:@batch_normalization_1/gamma*
_output_shapes	
:*
dtype0
Ż
1batch_normalization_1/beta/Adam/Initializer/zerosConst*-
_class#
!loc:@batch_normalization_1/beta*
_output_shapes	
:*
dtype0*
valueB*    
Ć
batch_normalization_1/beta/AdamVarHandleOp*-
_class#
!loc:@batch_normalization_1/beta*
_output_shapes
: *
dtype0*
shape:*0
shared_name!batch_normalization_1/beta/Adam
ž
@batch_normalization_1/beta/Adam/IsInitialized/VarIsInitializedOpVarIsInitializedOpbatch_normalization_1/beta/Adam*-
_class#
!loc:@batch_normalization_1/beta*
_output_shapes
: 
Ă
&batch_normalization_1/beta/Adam/AssignAssignVariableOpbatch_normalization_1/beta/Adam1batch_normalization_1/beta/Adam/Initializer/zeros*&
 _has_manual_control_dependencies(*
dtype0
ż
3batch_normalization_1/beta/Adam/Read/ReadVariableOpReadVariableOpbatch_normalization_1/beta/Adam*-
_class#
!loc:@batch_normalization_1/beta*
_output_shapes	
:*
dtype0
ą
3batch_normalization_1/beta/Adam_1/Initializer/zerosConst*-
_class#
!loc:@batch_normalization_1/beta*
_output_shapes	
:*
dtype0*
valueB*    
Ę
!batch_normalization_1/beta/Adam_1VarHandleOp*-
_class#
!loc:@batch_normalization_1/beta*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!batch_normalization_1/beta/Adam_1
Â
Bbatch_normalization_1/beta/Adam_1/IsInitialized/VarIsInitializedOpVarIsInitializedOp!batch_normalization_1/beta/Adam_1*-
_class#
!loc:@batch_normalization_1/beta*
_output_shapes
: 
É
(batch_normalization_1/beta/Adam_1/AssignAssignVariableOp!batch_normalization_1/beta/Adam_13batch_normalization_1/beta/Adam_1/Initializer/zeros*&
 _has_manual_control_dependencies(*
dtype0
Ă
5batch_normalization_1/beta/Adam_1/Read/ReadVariableOpReadVariableOp!batch_normalization_1/beta/Adam_1*-
_class#
!loc:@batch_normalization_1/beta*
_output_shapes	
:*
dtype0
§
0conv_d2/w/Adam/Initializer/zeros/shape_as_tensorConst*
_class
loc:@conv_d2/w*
_output_shapes
:*
dtype0*%
valueB"         #   

&conv_d2/w/Adam/Initializer/zeros/ConstConst*
_class
loc:@conv_d2/w*
_output_shapes
: *
dtype0*
valueB
 *    
Ň
 conv_d2/w/Adam/Initializer/zerosFill0conv_d2/w/Adam/Initializer/zeros/shape_as_tensor&conv_d2/w/Adam/Initializer/zeros/Const*
T0*
_class
loc:@conv_d2/w*'
_output_shapes
:#

conv_d2/w/AdamVarHandleOp*
_class
loc:@conv_d2/w*
_output_shapes
: *
dtype0*
shape:#*
shared_nameconv_d2/w/Adam

/conv_d2/w/Adam/IsInitialized/VarIsInitializedOpVarIsInitializedOpconv_d2/w/Adam*
_class
loc:@conv_d2/w*
_output_shapes
: 

conv_d2/w/Adam/AssignAssignVariableOpconv_d2/w/Adam conv_d2/w/Adam/Initializer/zeros*&
 _has_manual_control_dependencies(*
dtype0

"conv_d2/w/Adam/Read/ReadVariableOpReadVariableOpconv_d2/w/Adam*
_class
loc:@conv_d2/w*'
_output_shapes
:#*
dtype0
Š
2conv_d2/w/Adam_1/Initializer/zeros/shape_as_tensorConst*
_class
loc:@conv_d2/w*
_output_shapes
:*
dtype0*%
valueB"         #   

(conv_d2/w/Adam_1/Initializer/zeros/ConstConst*
_class
loc:@conv_d2/w*
_output_shapes
: *
dtype0*
valueB
 *    
Ř
"conv_d2/w/Adam_1/Initializer/zerosFill2conv_d2/w/Adam_1/Initializer/zeros/shape_as_tensor(conv_d2/w/Adam_1/Initializer/zeros/Const*
T0*
_class
loc:@conv_d2/w*'
_output_shapes
:#
Ł
conv_d2/w/Adam_1VarHandleOp*
_class
loc:@conv_d2/w*
_output_shapes
: *
dtype0*
shape:#*!
shared_nameconv_d2/w/Adam_1

1conv_d2/w/Adam_1/IsInitialized/VarIsInitializedOpVarIsInitializedOpconv_d2/w/Adam_1*
_class
loc:@conv_d2/w*
_output_shapes
: 

conv_d2/w/Adam_1/AssignAssignVariableOpconv_d2/w/Adam_1"conv_d2/w/Adam_1/Initializer/zeros*&
 _has_manual_control_dependencies(*
dtype0

$conv_d2/w/Adam_1/Read/ReadVariableOpReadVariableOpconv_d2/w/Adam_1*
_class
loc:@conv_d2/w*'
_output_shapes
:#*
dtype0

 conv_d2/b/Adam/Initializer/zerosConst*
_class
loc:@conv_d2/b*
_output_shapes
:#*
dtype0*
valueB#*    

conv_d2/b/AdamVarHandleOp*
_class
loc:@conv_d2/b*
_output_shapes
: *
dtype0*
shape:#*
shared_nameconv_d2/b/Adam

/conv_d2/b/Adam/IsInitialized/VarIsInitializedOpVarIsInitializedOpconv_d2/b/Adam*
_class
loc:@conv_d2/b*
_output_shapes
: 

conv_d2/b/Adam/AssignAssignVariableOpconv_d2/b/Adam conv_d2/b/Adam/Initializer/zeros*&
 _has_manual_control_dependencies(*
dtype0

"conv_d2/b/Adam/Read/ReadVariableOpReadVariableOpconv_d2/b/Adam*
_class
loc:@conv_d2/b*
_output_shapes
:#*
dtype0

"conv_d2/b/Adam_1/Initializer/zerosConst*
_class
loc:@conv_d2/b*
_output_shapes
:#*
dtype0*
valueB#*    

conv_d2/b/Adam_1VarHandleOp*
_class
loc:@conv_d2/b*
_output_shapes
: *
dtype0*
shape:#*!
shared_nameconv_d2/b/Adam_1

1conv_d2/b/Adam_1/IsInitialized/VarIsInitializedOpVarIsInitializedOpconv_d2/b/Adam_1*
_class
loc:@conv_d2/b*
_output_shapes
: 

conv_d2/b/Adam_1/AssignAssignVariableOpconv_d2/b/Adam_1"conv_d2/b/Adam_1/Initializer/zeros*&
 _has_manual_control_dependencies(*
dtype0

$conv_d2/b/Adam_1/Read/ReadVariableOpReadVariableOpconv_d2/b/Adam_1*
_class
loc:@conv_d2/b*
_output_shapes
:#*
dtype0
Ż
2batch_normalization_2/gamma/Adam/Initializer/zerosConst*.
_class$
" loc:@batch_normalization_2/gamma*
_output_shapes
:#*
dtype0*
valueB#*    
Č
 batch_normalization_2/gamma/AdamVarHandleOp*.
_class$
" loc:@batch_normalization_2/gamma*
_output_shapes
: *
dtype0*
shape:#*1
shared_name" batch_normalization_2/gamma/Adam
Á
Abatch_normalization_2/gamma/Adam/IsInitialized/VarIsInitializedOpVarIsInitializedOp batch_normalization_2/gamma/Adam*.
_class$
" loc:@batch_normalization_2/gamma*
_output_shapes
: 
Ć
'batch_normalization_2/gamma/Adam/AssignAssignVariableOp batch_normalization_2/gamma/Adam2batch_normalization_2/gamma/Adam/Initializer/zeros*&
 _has_manual_control_dependencies(*
dtype0
Á
4batch_normalization_2/gamma/Adam/Read/ReadVariableOpReadVariableOp batch_normalization_2/gamma/Adam*.
_class$
" loc:@batch_normalization_2/gamma*
_output_shapes
:#*
dtype0
ą
4batch_normalization_2/gamma/Adam_1/Initializer/zerosConst*.
_class$
" loc:@batch_normalization_2/gamma*
_output_shapes
:#*
dtype0*
valueB#*    
Ě
"batch_normalization_2/gamma/Adam_1VarHandleOp*.
_class$
" loc:@batch_normalization_2/gamma*
_output_shapes
: *
dtype0*
shape:#*3
shared_name$"batch_normalization_2/gamma/Adam_1
Ĺ
Cbatch_normalization_2/gamma/Adam_1/IsInitialized/VarIsInitializedOpVarIsInitializedOp"batch_normalization_2/gamma/Adam_1*.
_class$
" loc:@batch_normalization_2/gamma*
_output_shapes
: 
Ě
)batch_normalization_2/gamma/Adam_1/AssignAssignVariableOp"batch_normalization_2/gamma/Adam_14batch_normalization_2/gamma/Adam_1/Initializer/zeros*&
 _has_manual_control_dependencies(*
dtype0
Ĺ
6batch_normalization_2/gamma/Adam_1/Read/ReadVariableOpReadVariableOp"batch_normalization_2/gamma/Adam_1*.
_class$
" loc:@batch_normalization_2/gamma*
_output_shapes
:#*
dtype0
­
1batch_normalization_2/beta/Adam/Initializer/zerosConst*-
_class#
!loc:@batch_normalization_2/beta*
_output_shapes
:#*
dtype0*
valueB#*    
Ĺ
batch_normalization_2/beta/AdamVarHandleOp*-
_class#
!loc:@batch_normalization_2/beta*
_output_shapes
: *
dtype0*
shape:#*0
shared_name!batch_normalization_2/beta/Adam
ž
@batch_normalization_2/beta/Adam/IsInitialized/VarIsInitializedOpVarIsInitializedOpbatch_normalization_2/beta/Adam*-
_class#
!loc:@batch_normalization_2/beta*
_output_shapes
: 
Ă
&batch_normalization_2/beta/Adam/AssignAssignVariableOpbatch_normalization_2/beta/Adam1batch_normalization_2/beta/Adam/Initializer/zeros*&
 _has_manual_control_dependencies(*
dtype0
ž
3batch_normalization_2/beta/Adam/Read/ReadVariableOpReadVariableOpbatch_normalization_2/beta/Adam*-
_class#
!loc:@batch_normalization_2/beta*
_output_shapes
:#*
dtype0
Ż
3batch_normalization_2/beta/Adam_1/Initializer/zerosConst*-
_class#
!loc:@batch_normalization_2/beta*
_output_shapes
:#*
dtype0*
valueB#*    
É
!batch_normalization_2/beta/Adam_1VarHandleOp*-
_class#
!loc:@batch_normalization_2/beta*
_output_shapes
: *
dtype0*
shape:#*2
shared_name#!batch_normalization_2/beta/Adam_1
Â
Bbatch_normalization_2/beta/Adam_1/IsInitialized/VarIsInitializedOpVarIsInitializedOp!batch_normalization_2/beta/Adam_1*-
_class#
!loc:@batch_normalization_2/beta*
_output_shapes
: 
É
(batch_normalization_2/beta/Adam_1/AssignAssignVariableOp!batch_normalization_2/beta/Adam_13batch_normalization_2/beta/Adam_1/Initializer/zeros*&
 _has_manual_control_dependencies(*
dtype0
Â
5batch_normalization_2/beta/Adam_1/Read/ReadVariableOpReadVariableOp!batch_normalization_2/beta/Adam_1*-
_class#
!loc:@batch_normalization_2/beta*
_output_shapes
:#*
dtype0
Š
1conv_out/w/Adam/Initializer/zeros/shape_as_tensorConst*
_class
loc:@conv_out/w*
_output_shapes
:*
dtype0*%
valueB"      ă  #   

'conv_out/w/Adam/Initializer/zeros/ConstConst*
_class
loc:@conv_out/w*
_output_shapes
: *
dtype0*
valueB
 *    
Ö
!conv_out/w/Adam/Initializer/zerosFill1conv_out/w/Adam/Initializer/zeros/shape_as_tensor'conv_out/w/Adam/Initializer/zeros/Const*
T0*
_class
loc:@conv_out/w*'
_output_shapes
:ă#
˘
conv_out/w/AdamVarHandleOp*
_class
loc:@conv_out/w*
_output_shapes
: *
dtype0*
shape:ă#* 
shared_nameconv_out/w/Adam

0conv_out/w/Adam/IsInitialized/VarIsInitializedOpVarIsInitializedOpconv_out/w/Adam*
_class
loc:@conv_out/w*
_output_shapes
: 

conv_out/w/Adam/AssignAssignVariableOpconv_out/w/Adam!conv_out/w/Adam/Initializer/zeros*&
 _has_manual_control_dependencies(*
dtype0

#conv_out/w/Adam/Read/ReadVariableOpReadVariableOpconv_out/w/Adam*
_class
loc:@conv_out/w*'
_output_shapes
:ă#*
dtype0
Ť
3conv_out/w/Adam_1/Initializer/zeros/shape_as_tensorConst*
_class
loc:@conv_out/w*
_output_shapes
:*
dtype0*%
valueB"      ă  #   

)conv_out/w/Adam_1/Initializer/zeros/ConstConst*
_class
loc:@conv_out/w*
_output_shapes
: *
dtype0*
valueB
 *    
Ü
#conv_out/w/Adam_1/Initializer/zerosFill3conv_out/w/Adam_1/Initializer/zeros/shape_as_tensor)conv_out/w/Adam_1/Initializer/zeros/Const*
T0*
_class
loc:@conv_out/w*'
_output_shapes
:ă#
Ś
conv_out/w/Adam_1VarHandleOp*
_class
loc:@conv_out/w*
_output_shapes
: *
dtype0*
shape:ă#*"
shared_nameconv_out/w/Adam_1

2conv_out/w/Adam_1/IsInitialized/VarIsInitializedOpVarIsInitializedOpconv_out/w/Adam_1*
_class
loc:@conv_out/w*
_output_shapes
: 

conv_out/w/Adam_1/AssignAssignVariableOpconv_out/w/Adam_1#conv_out/w/Adam_1/Initializer/zeros*&
 _has_manual_control_dependencies(*
dtype0

%conv_out/w/Adam_1/Read/ReadVariableOpReadVariableOpconv_out/w/Adam_1*
_class
loc:@conv_out/w*'
_output_shapes
:ă#*
dtype0

!conv_out/b/Adam/Initializer/zerosConst*
_class
loc:@conv_out/b*
_output_shapes
:#*
dtype0*
valueB#*    

conv_out/b/AdamVarHandleOp*
_class
loc:@conv_out/b*
_output_shapes
: *
dtype0*
shape:#* 
shared_nameconv_out/b/Adam

0conv_out/b/Adam/IsInitialized/VarIsInitializedOpVarIsInitializedOpconv_out/b/Adam*
_class
loc:@conv_out/b*
_output_shapes
: 

conv_out/b/Adam/AssignAssignVariableOpconv_out/b/Adam!conv_out/b/Adam/Initializer/zeros*&
 _has_manual_control_dependencies(*
dtype0

#conv_out/b/Adam/Read/ReadVariableOpReadVariableOpconv_out/b/Adam*
_class
loc:@conv_out/b*
_output_shapes
:#*
dtype0

#conv_out/b/Adam_1/Initializer/zerosConst*
_class
loc:@conv_out/b*
_output_shapes
:#*
dtype0*
valueB#*    

conv_out/b/Adam_1VarHandleOp*
_class
loc:@conv_out/b*
_output_shapes
: *
dtype0*
shape:#*"
shared_nameconv_out/b/Adam_1

2conv_out/b/Adam_1/IsInitialized/VarIsInitializedOpVarIsInitializedOpconv_out/b/Adam_1*
_class
loc:@conv_out/b*
_output_shapes
: 

conv_out/b/Adam_1/AssignAssignVariableOpconv_out/b/Adam_1#conv_out/b/Adam_1/Initializer/zeros*&
 _has_manual_control_dependencies(*
dtype0

%conv_out/b/Adam_1/Read/ReadVariableOpReadVariableOpconv_out/b/Adam_1*
_class
loc:@conv_out/b*
_output_shapes
:#*
dtype0
O

Adam/beta1Const*
_output_shapes
: *
dtype0*
valueB
 *fff?
O

Adam/beta2Const*
_output_shapes
: *
dtype0*
valueB
 *wž?
Q
Adam/epsilonConst*
_output_shapes
: *
dtype0*
valueB
 *wĚ+2
x
4Adam/update_conv1/w/ResourceApplyAdam/ReadVariableOpReadVariableOpbeta1_power*
_output_shapes
: *
dtype0
z
6Adam/update_conv1/w/ResourceApplyAdam/ReadVariableOp_1ReadVariableOpbeta2_power*
_output_shapes
: *
dtype0

%Adam/update_conv1/w/ResourceApplyAdamResourceApplyAdamconv1/wconv1/w/Adamconv1/w/Adam_14Adam/update_conv1/w/ResourceApplyAdam/ReadVariableOp6Adam/update_conv1/w/ResourceApplyAdam/ReadVariableOp_1ExponentialDecay
Adam/beta1
Adam/beta2Adam/epsilon6gradients/conv1/Conv2D_grad/tuple/control_dependency_1*
T0*
_class
loc:@conv1/w*&
 _has_manual_control_dependencies(
x
4Adam/update_conv1/b/ResourceApplyAdam/ReadVariableOpReadVariableOpbeta1_power*
_output_shapes
: *
dtype0
z
6Adam/update_conv1/b/ResourceApplyAdam/ReadVariableOp_1ReadVariableOpbeta2_power*
_output_shapes
: *
dtype0

%Adam/update_conv1/b/ResourceApplyAdamResourceApplyAdamconv1/bconv1/b/Adamconv1/b/Adam_14Adam/update_conv1/b/ResourceApplyAdam/ReadVariableOp6Adam/update_conv1/b/ResourceApplyAdam/ReadVariableOp_1ExponentialDecay
Adam/beta1
Adam/beta2Adam/epsilon7gradients/conv1/BiasAdd_grad/tuple/control_dependency_1*
T0*
_class
loc:@conv1/b*&
 _has_manual_control_dependencies(

FAdam/update_batch_normalization/gamma/ResourceApplyAdam/ReadVariableOpReadVariableOpbeta1_power*
_output_shapes
: *
dtype0

HAdam/update_batch_normalization/gamma/ResourceApplyAdam/ReadVariableOp_1ReadVariableOpbeta2_power*
_output_shapes
: *
dtype0
˘
7Adam/update_batch_normalization/gamma/ResourceApplyAdamResourceApplyAdambatch_normalization/gammabatch_normalization/gamma/Adam batch_normalization/gamma/Adam_1FAdam/update_batch_normalization/gamma/ResourceApplyAdam/ReadVariableOpHAdam/update_batch_normalization/gamma/ResourceApplyAdam/ReadVariableOp_1ExponentialDecay
Adam/beta1
Adam/beta2Adam/epsilonNgradients/batch_normalization/FusedBatchNormV3_grad/tuple/control_dependency_1*
T0*,
_class"
 loc:@batch_normalization/gamma*&
 _has_manual_control_dependencies(

EAdam/update_batch_normalization/beta/ResourceApplyAdam/ReadVariableOpReadVariableOpbeta1_power*
_output_shapes
: *
dtype0

GAdam/update_batch_normalization/beta/ResourceApplyAdam/ReadVariableOp_1ReadVariableOpbeta2_power*
_output_shapes
: *
dtype0

6Adam/update_batch_normalization/beta/ResourceApplyAdamResourceApplyAdambatch_normalization/betabatch_normalization/beta/Adambatch_normalization/beta/Adam_1EAdam/update_batch_normalization/beta/ResourceApplyAdam/ReadVariableOpGAdam/update_batch_normalization/beta/ResourceApplyAdam/ReadVariableOp_1ExponentialDecay
Adam/beta1
Adam/beta2Adam/epsilonNgradients/batch_normalization/FusedBatchNormV3_grad/tuple/control_dependency_2*
T0*+
_class!
loc:@batch_normalization/beta*&
 _has_manual_control_dependencies(
}
9Adam/update_sbb1/conv1/w/ResourceApplyAdam/ReadVariableOpReadVariableOpbeta1_power*
_output_shapes
: *
dtype0

;Adam/update_sbb1/conv1/w/ResourceApplyAdam/ReadVariableOp_1ReadVariableOpbeta2_power*
_output_shapes
: *
dtype0
´
*Adam/update_sbb1/conv1/w/ResourceApplyAdamResourceApplyAdamsbb1/conv1/wsbb1/conv1/w/Adamsbb1/conv1/w/Adam_19Adam/update_sbb1/conv1/w/ResourceApplyAdam/ReadVariableOp;Adam/update_sbb1/conv1/w/ResourceApplyAdam/ReadVariableOp_1ExponentialDecay
Adam/beta1
Adam/beta2Adam/epsilon;gradients/sbb1/conv1/Conv2D_grad/tuple/control_dependency_1*
T0*
_class
loc:@sbb1/conv1/w*&
 _has_manual_control_dependencies(
}
9Adam/update_sbb1/conv1/b/ResourceApplyAdam/ReadVariableOpReadVariableOpbeta1_power*
_output_shapes
: *
dtype0

;Adam/update_sbb1/conv1/b/ResourceApplyAdam/ReadVariableOp_1ReadVariableOpbeta2_power*
_output_shapes
: *
dtype0
ľ
*Adam/update_sbb1/conv1/b/ResourceApplyAdamResourceApplyAdamsbb1/conv1/bsbb1/conv1/b/Adamsbb1/conv1/b/Adam_19Adam/update_sbb1/conv1/b/ResourceApplyAdam/ReadVariableOp;Adam/update_sbb1/conv1/b/ResourceApplyAdam/ReadVariableOp_1ExponentialDecay
Adam/beta1
Adam/beta2Adam/epsilon<gradients/sbb1/conv1/BiasAdd_grad/tuple/control_dependency_1*
T0*
_class
loc:@sbb1/conv1/b*&
 _has_manual_control_dependencies(
}
9Adam/update_sbb1/conv2/w/ResourceApplyAdam/ReadVariableOpReadVariableOpbeta1_power*
_output_shapes
: *
dtype0

;Adam/update_sbb1/conv2/w/ResourceApplyAdam/ReadVariableOp_1ReadVariableOpbeta2_power*
_output_shapes
: *
dtype0
´
*Adam/update_sbb1/conv2/w/ResourceApplyAdamResourceApplyAdamsbb1/conv2/wsbb1/conv2/w/Adamsbb1/conv2/w/Adam_19Adam/update_sbb1/conv2/w/ResourceApplyAdam/ReadVariableOp;Adam/update_sbb1/conv2/w/ResourceApplyAdam/ReadVariableOp_1ExponentialDecay
Adam/beta1
Adam/beta2Adam/epsilon;gradients/sbb1/conv2/Conv2D_grad/tuple/control_dependency_1*
T0*
_class
loc:@sbb1/conv2/w*&
 _has_manual_control_dependencies(
}
9Adam/update_sbb1/conv2/b/ResourceApplyAdam/ReadVariableOpReadVariableOpbeta1_power*
_output_shapes
: *
dtype0

;Adam/update_sbb1/conv2/b/ResourceApplyAdam/ReadVariableOp_1ReadVariableOpbeta2_power*
_output_shapes
: *
dtype0
ľ
*Adam/update_sbb1/conv2/b/ResourceApplyAdamResourceApplyAdamsbb1/conv2/bsbb1/conv2/b/Adamsbb1/conv2/b/Adam_19Adam/update_sbb1/conv2/b/ResourceApplyAdam/ReadVariableOp;Adam/update_sbb1/conv2/b/ResourceApplyAdam/ReadVariableOp_1ExponentialDecay
Adam/beta1
Adam/beta2Adam/epsilon<gradients/sbb1/conv2/BiasAdd_grad/tuple/control_dependency_1*
T0*
_class
loc:@sbb1/conv2/b*&
 _has_manual_control_dependencies(
}
9Adam/update_sbb1/conv3/w/ResourceApplyAdam/ReadVariableOpReadVariableOpbeta1_power*
_output_shapes
: *
dtype0

;Adam/update_sbb1/conv3/w/ResourceApplyAdam/ReadVariableOp_1ReadVariableOpbeta2_power*
_output_shapes
: *
dtype0
´
*Adam/update_sbb1/conv3/w/ResourceApplyAdamResourceApplyAdamsbb1/conv3/wsbb1/conv3/w/Adamsbb1/conv3/w/Adam_19Adam/update_sbb1/conv3/w/ResourceApplyAdam/ReadVariableOp;Adam/update_sbb1/conv3/w/ResourceApplyAdam/ReadVariableOp_1ExponentialDecay
Adam/beta1
Adam/beta2Adam/epsilon;gradients/sbb1/conv3/Conv2D_grad/tuple/control_dependency_1*
T0*
_class
loc:@sbb1/conv3/w*&
 _has_manual_control_dependencies(
}
9Adam/update_sbb1/conv3/b/ResourceApplyAdam/ReadVariableOpReadVariableOpbeta1_power*
_output_shapes
: *
dtype0

;Adam/update_sbb1/conv3/b/ResourceApplyAdam/ReadVariableOp_1ReadVariableOpbeta2_power*
_output_shapes
: *
dtype0
ľ
*Adam/update_sbb1/conv3/b/ResourceApplyAdamResourceApplyAdamsbb1/conv3/bsbb1/conv3/b/Adamsbb1/conv3/b/Adam_19Adam/update_sbb1/conv3/b/ResourceApplyAdam/ReadVariableOp;Adam/update_sbb1/conv3/b/ResourceApplyAdam/ReadVariableOp_1ExponentialDecay
Adam/beta1
Adam/beta2Adam/epsilon<gradients/sbb1/conv3/BiasAdd_grad/tuple/control_dependency_1*
T0*
_class
loc:@sbb1/conv3/b*&
 _has_manual_control_dependencies(
}
9Adam/update_sbb1/conv4/w/ResourceApplyAdam/ReadVariableOpReadVariableOpbeta1_power*
_output_shapes
: *
dtype0

;Adam/update_sbb1/conv4/w/ResourceApplyAdam/ReadVariableOp_1ReadVariableOpbeta2_power*
_output_shapes
: *
dtype0
´
*Adam/update_sbb1/conv4/w/ResourceApplyAdamResourceApplyAdamsbb1/conv4/wsbb1/conv4/w/Adamsbb1/conv4/w/Adam_19Adam/update_sbb1/conv4/w/ResourceApplyAdam/ReadVariableOp;Adam/update_sbb1/conv4/w/ResourceApplyAdam/ReadVariableOp_1ExponentialDecay
Adam/beta1
Adam/beta2Adam/epsilon;gradients/sbb1/conv4/Conv2D_grad/tuple/control_dependency_1*
T0*
_class
loc:@sbb1/conv4/w*&
 _has_manual_control_dependencies(
}
9Adam/update_sbb1/conv4/b/ResourceApplyAdam/ReadVariableOpReadVariableOpbeta1_power*
_output_shapes
: *
dtype0

;Adam/update_sbb1/conv4/b/ResourceApplyAdam/ReadVariableOp_1ReadVariableOpbeta2_power*
_output_shapes
: *
dtype0
ľ
*Adam/update_sbb1/conv4/b/ResourceApplyAdamResourceApplyAdamsbb1/conv4/bsbb1/conv4/b/Adamsbb1/conv4/b/Adam_19Adam/update_sbb1/conv4/b/ResourceApplyAdam/ReadVariableOp;Adam/update_sbb1/conv4/b/ResourceApplyAdam/ReadVariableOp_1ExponentialDecay
Adam/beta1
Adam/beta2Adam/epsilon<gradients/sbb1/conv4/BiasAdd_grad/tuple/control_dependency_1*
T0*
_class
loc:@sbb1/conv4/b*&
 _has_manual_control_dependencies(

KAdam/update_sbb1/batch_normalization/gamma/ResourceApplyAdam/ReadVariableOpReadVariableOpbeta1_power*
_output_shapes
: *
dtype0

MAdam/update_sbb1/batch_normalization/gamma/ResourceApplyAdam/ReadVariableOp_1ReadVariableOpbeta2_power*
_output_shapes
: *
dtype0
Ę
<Adam/update_sbb1/batch_normalization/gamma/ResourceApplyAdamResourceApplyAdamsbb1/batch_normalization/gamma#sbb1/batch_normalization/gamma/Adam%sbb1/batch_normalization/gamma/Adam_1KAdam/update_sbb1/batch_normalization/gamma/ResourceApplyAdam/ReadVariableOpMAdam/update_sbb1/batch_normalization/gamma/ResourceApplyAdam/ReadVariableOp_1ExponentialDecay
Adam/beta1
Adam/beta2Adam/epsilonSgradients/sbb1/batch_normalization/FusedBatchNormV3_grad/tuple/control_dependency_1*
T0*1
_class'
%#loc:@sbb1/batch_normalization/gamma*&
 _has_manual_control_dependencies(

JAdam/update_sbb1/batch_normalization/beta/ResourceApplyAdam/ReadVariableOpReadVariableOpbeta1_power*
_output_shapes
: *
dtype0

LAdam/update_sbb1/batch_normalization/beta/ResourceApplyAdam/ReadVariableOp_1ReadVariableOpbeta2_power*
_output_shapes
: *
dtype0
Ă
;Adam/update_sbb1/batch_normalization/beta/ResourceApplyAdamResourceApplyAdamsbb1/batch_normalization/beta"sbb1/batch_normalization/beta/Adam$sbb1/batch_normalization/beta/Adam_1JAdam/update_sbb1/batch_normalization/beta/ResourceApplyAdam/ReadVariableOpLAdam/update_sbb1/batch_normalization/beta/ResourceApplyAdam/ReadVariableOp_1ExponentialDecay
Adam/beta1
Adam/beta2Adam/epsilonSgradients/sbb1/batch_normalization/FusedBatchNormV3_grad/tuple/control_dependency_2*
T0*0
_class&
$"loc:@sbb1/batch_normalization/beta*&
 _has_manual_control_dependencies(
}
9Adam/update_sbb2/conv1/w/ResourceApplyAdam/ReadVariableOpReadVariableOpbeta1_power*
_output_shapes
: *
dtype0

;Adam/update_sbb2/conv1/w/ResourceApplyAdam/ReadVariableOp_1ReadVariableOpbeta2_power*
_output_shapes
: *
dtype0
´
*Adam/update_sbb2/conv1/w/ResourceApplyAdamResourceApplyAdamsbb2/conv1/wsbb2/conv1/w/Adamsbb2/conv1/w/Adam_19Adam/update_sbb2/conv1/w/ResourceApplyAdam/ReadVariableOp;Adam/update_sbb2/conv1/w/ResourceApplyAdam/ReadVariableOp_1ExponentialDecay
Adam/beta1
Adam/beta2Adam/epsilon;gradients/sbb2/conv1/Conv2D_grad/tuple/control_dependency_1*
T0*
_class
loc:@sbb2/conv1/w*&
 _has_manual_control_dependencies(
}
9Adam/update_sbb2/conv1/b/ResourceApplyAdam/ReadVariableOpReadVariableOpbeta1_power*
_output_shapes
: *
dtype0

;Adam/update_sbb2/conv1/b/ResourceApplyAdam/ReadVariableOp_1ReadVariableOpbeta2_power*
_output_shapes
: *
dtype0
ľ
*Adam/update_sbb2/conv1/b/ResourceApplyAdamResourceApplyAdamsbb2/conv1/bsbb2/conv1/b/Adamsbb2/conv1/b/Adam_19Adam/update_sbb2/conv1/b/ResourceApplyAdam/ReadVariableOp;Adam/update_sbb2/conv1/b/ResourceApplyAdam/ReadVariableOp_1ExponentialDecay
Adam/beta1
Adam/beta2Adam/epsilon<gradients/sbb2/conv1/BiasAdd_grad/tuple/control_dependency_1*
T0*
_class
loc:@sbb2/conv1/b*&
 _has_manual_control_dependencies(
}
9Adam/update_sbb2/conv2/w/ResourceApplyAdam/ReadVariableOpReadVariableOpbeta1_power*
_output_shapes
: *
dtype0

;Adam/update_sbb2/conv2/w/ResourceApplyAdam/ReadVariableOp_1ReadVariableOpbeta2_power*
_output_shapes
: *
dtype0
´
*Adam/update_sbb2/conv2/w/ResourceApplyAdamResourceApplyAdamsbb2/conv2/wsbb2/conv2/w/Adamsbb2/conv2/w/Adam_19Adam/update_sbb2/conv2/w/ResourceApplyAdam/ReadVariableOp;Adam/update_sbb2/conv2/w/ResourceApplyAdam/ReadVariableOp_1ExponentialDecay
Adam/beta1
Adam/beta2Adam/epsilon;gradients/sbb2/conv2/Conv2D_grad/tuple/control_dependency_1*
T0*
_class
loc:@sbb2/conv2/w*&
 _has_manual_control_dependencies(
}
9Adam/update_sbb2/conv2/b/ResourceApplyAdam/ReadVariableOpReadVariableOpbeta1_power*
_output_shapes
: *
dtype0

;Adam/update_sbb2/conv2/b/ResourceApplyAdam/ReadVariableOp_1ReadVariableOpbeta2_power*
_output_shapes
: *
dtype0
ľ
*Adam/update_sbb2/conv2/b/ResourceApplyAdamResourceApplyAdamsbb2/conv2/bsbb2/conv2/b/Adamsbb2/conv2/b/Adam_19Adam/update_sbb2/conv2/b/ResourceApplyAdam/ReadVariableOp;Adam/update_sbb2/conv2/b/ResourceApplyAdam/ReadVariableOp_1ExponentialDecay
Adam/beta1
Adam/beta2Adam/epsilon<gradients/sbb2/conv2/BiasAdd_grad/tuple/control_dependency_1*
T0*
_class
loc:@sbb2/conv2/b*&
 _has_manual_control_dependencies(
}
9Adam/update_sbb2/conv3/w/ResourceApplyAdam/ReadVariableOpReadVariableOpbeta1_power*
_output_shapes
: *
dtype0

;Adam/update_sbb2/conv3/w/ResourceApplyAdam/ReadVariableOp_1ReadVariableOpbeta2_power*
_output_shapes
: *
dtype0
´
*Adam/update_sbb2/conv3/w/ResourceApplyAdamResourceApplyAdamsbb2/conv3/wsbb2/conv3/w/Adamsbb2/conv3/w/Adam_19Adam/update_sbb2/conv3/w/ResourceApplyAdam/ReadVariableOp;Adam/update_sbb2/conv3/w/ResourceApplyAdam/ReadVariableOp_1ExponentialDecay
Adam/beta1
Adam/beta2Adam/epsilon;gradients/sbb2/conv3/Conv2D_grad/tuple/control_dependency_1*
T0*
_class
loc:@sbb2/conv3/w*&
 _has_manual_control_dependencies(
}
9Adam/update_sbb2/conv3/b/ResourceApplyAdam/ReadVariableOpReadVariableOpbeta1_power*
_output_shapes
: *
dtype0

;Adam/update_sbb2/conv3/b/ResourceApplyAdam/ReadVariableOp_1ReadVariableOpbeta2_power*
_output_shapes
: *
dtype0
ľ
*Adam/update_sbb2/conv3/b/ResourceApplyAdamResourceApplyAdamsbb2/conv3/bsbb2/conv3/b/Adamsbb2/conv3/b/Adam_19Adam/update_sbb2/conv3/b/ResourceApplyAdam/ReadVariableOp;Adam/update_sbb2/conv3/b/ResourceApplyAdam/ReadVariableOp_1ExponentialDecay
Adam/beta1
Adam/beta2Adam/epsilon<gradients/sbb2/conv3/BiasAdd_grad/tuple/control_dependency_1*
T0*
_class
loc:@sbb2/conv3/b*&
 _has_manual_control_dependencies(
}
9Adam/update_sbb2/conv4/w/ResourceApplyAdam/ReadVariableOpReadVariableOpbeta1_power*
_output_shapes
: *
dtype0

;Adam/update_sbb2/conv4/w/ResourceApplyAdam/ReadVariableOp_1ReadVariableOpbeta2_power*
_output_shapes
: *
dtype0
´
*Adam/update_sbb2/conv4/w/ResourceApplyAdamResourceApplyAdamsbb2/conv4/wsbb2/conv4/w/Adamsbb2/conv4/w/Adam_19Adam/update_sbb2/conv4/w/ResourceApplyAdam/ReadVariableOp;Adam/update_sbb2/conv4/w/ResourceApplyAdam/ReadVariableOp_1ExponentialDecay
Adam/beta1
Adam/beta2Adam/epsilon;gradients/sbb2/conv4/Conv2D_grad/tuple/control_dependency_1*
T0*
_class
loc:@sbb2/conv4/w*&
 _has_manual_control_dependencies(
}
9Adam/update_sbb2/conv4/b/ResourceApplyAdam/ReadVariableOpReadVariableOpbeta1_power*
_output_shapes
: *
dtype0

;Adam/update_sbb2/conv4/b/ResourceApplyAdam/ReadVariableOp_1ReadVariableOpbeta2_power*
_output_shapes
: *
dtype0
ľ
*Adam/update_sbb2/conv4/b/ResourceApplyAdamResourceApplyAdamsbb2/conv4/bsbb2/conv4/b/Adamsbb2/conv4/b/Adam_19Adam/update_sbb2/conv4/b/ResourceApplyAdam/ReadVariableOp;Adam/update_sbb2/conv4/b/ResourceApplyAdam/ReadVariableOp_1ExponentialDecay
Adam/beta1
Adam/beta2Adam/epsilon<gradients/sbb2/conv4/BiasAdd_grad/tuple/control_dependency_1*
T0*
_class
loc:@sbb2/conv4/b*&
 _has_manual_control_dependencies(

KAdam/update_sbb2/batch_normalization/gamma/ResourceApplyAdam/ReadVariableOpReadVariableOpbeta1_power*
_output_shapes
: *
dtype0

MAdam/update_sbb2/batch_normalization/gamma/ResourceApplyAdam/ReadVariableOp_1ReadVariableOpbeta2_power*
_output_shapes
: *
dtype0
Ę
<Adam/update_sbb2/batch_normalization/gamma/ResourceApplyAdamResourceApplyAdamsbb2/batch_normalization/gamma#sbb2/batch_normalization/gamma/Adam%sbb2/batch_normalization/gamma/Adam_1KAdam/update_sbb2/batch_normalization/gamma/ResourceApplyAdam/ReadVariableOpMAdam/update_sbb2/batch_normalization/gamma/ResourceApplyAdam/ReadVariableOp_1ExponentialDecay
Adam/beta1
Adam/beta2Adam/epsilonSgradients/sbb2/batch_normalization/FusedBatchNormV3_grad/tuple/control_dependency_1*
T0*1
_class'
%#loc:@sbb2/batch_normalization/gamma*&
 _has_manual_control_dependencies(

JAdam/update_sbb2/batch_normalization/beta/ResourceApplyAdam/ReadVariableOpReadVariableOpbeta1_power*
_output_shapes
: *
dtype0

LAdam/update_sbb2/batch_normalization/beta/ResourceApplyAdam/ReadVariableOp_1ReadVariableOpbeta2_power*
_output_shapes
: *
dtype0
Ă
;Adam/update_sbb2/batch_normalization/beta/ResourceApplyAdamResourceApplyAdamsbb2/batch_normalization/beta"sbb2/batch_normalization/beta/Adam$sbb2/batch_normalization/beta/Adam_1JAdam/update_sbb2/batch_normalization/beta/ResourceApplyAdam/ReadVariableOpLAdam/update_sbb2/batch_normalization/beta/ResourceApplyAdam/ReadVariableOp_1ExponentialDecay
Adam/beta1
Adam/beta2Adam/epsilonSgradients/sbb2/batch_normalization/FusedBatchNormV3_grad/tuple/control_dependency_2*
T0*0
_class&
$"loc:@sbb2/batch_normalization/beta*&
 _has_manual_control_dependencies(
}
9Adam/update_sbb3/conv1/w/ResourceApplyAdam/ReadVariableOpReadVariableOpbeta1_power*
_output_shapes
: *
dtype0

;Adam/update_sbb3/conv1/w/ResourceApplyAdam/ReadVariableOp_1ReadVariableOpbeta2_power*
_output_shapes
: *
dtype0
´
*Adam/update_sbb3/conv1/w/ResourceApplyAdamResourceApplyAdamsbb3/conv1/wsbb3/conv1/w/Adamsbb3/conv1/w/Adam_19Adam/update_sbb3/conv1/w/ResourceApplyAdam/ReadVariableOp;Adam/update_sbb3/conv1/w/ResourceApplyAdam/ReadVariableOp_1ExponentialDecay
Adam/beta1
Adam/beta2Adam/epsilon;gradients/sbb3/conv1/Conv2D_grad/tuple/control_dependency_1*
T0*
_class
loc:@sbb3/conv1/w*&
 _has_manual_control_dependencies(
}
9Adam/update_sbb3/conv1/b/ResourceApplyAdam/ReadVariableOpReadVariableOpbeta1_power*
_output_shapes
: *
dtype0

;Adam/update_sbb3/conv1/b/ResourceApplyAdam/ReadVariableOp_1ReadVariableOpbeta2_power*
_output_shapes
: *
dtype0
ľ
*Adam/update_sbb3/conv1/b/ResourceApplyAdamResourceApplyAdamsbb3/conv1/bsbb3/conv1/b/Adamsbb3/conv1/b/Adam_19Adam/update_sbb3/conv1/b/ResourceApplyAdam/ReadVariableOp;Adam/update_sbb3/conv1/b/ResourceApplyAdam/ReadVariableOp_1ExponentialDecay
Adam/beta1
Adam/beta2Adam/epsilon<gradients/sbb3/conv1/BiasAdd_grad/tuple/control_dependency_1*
T0*
_class
loc:@sbb3/conv1/b*&
 _has_manual_control_dependencies(
}
9Adam/update_sbb3/conv2/w/ResourceApplyAdam/ReadVariableOpReadVariableOpbeta1_power*
_output_shapes
: *
dtype0

;Adam/update_sbb3/conv2/w/ResourceApplyAdam/ReadVariableOp_1ReadVariableOpbeta2_power*
_output_shapes
: *
dtype0
´
*Adam/update_sbb3/conv2/w/ResourceApplyAdamResourceApplyAdamsbb3/conv2/wsbb3/conv2/w/Adamsbb3/conv2/w/Adam_19Adam/update_sbb3/conv2/w/ResourceApplyAdam/ReadVariableOp;Adam/update_sbb3/conv2/w/ResourceApplyAdam/ReadVariableOp_1ExponentialDecay
Adam/beta1
Adam/beta2Adam/epsilon;gradients/sbb3/conv2/Conv2D_grad/tuple/control_dependency_1*
T0*
_class
loc:@sbb3/conv2/w*&
 _has_manual_control_dependencies(
}
9Adam/update_sbb3/conv2/b/ResourceApplyAdam/ReadVariableOpReadVariableOpbeta1_power*
_output_shapes
: *
dtype0

;Adam/update_sbb3/conv2/b/ResourceApplyAdam/ReadVariableOp_1ReadVariableOpbeta2_power*
_output_shapes
: *
dtype0
ľ
*Adam/update_sbb3/conv2/b/ResourceApplyAdamResourceApplyAdamsbb3/conv2/bsbb3/conv2/b/Adamsbb3/conv2/b/Adam_19Adam/update_sbb3/conv2/b/ResourceApplyAdam/ReadVariableOp;Adam/update_sbb3/conv2/b/ResourceApplyAdam/ReadVariableOp_1ExponentialDecay
Adam/beta1
Adam/beta2Adam/epsilon<gradients/sbb3/conv2/BiasAdd_grad/tuple/control_dependency_1*
T0*
_class
loc:@sbb3/conv2/b*&
 _has_manual_control_dependencies(
}
9Adam/update_sbb3/conv3/w/ResourceApplyAdam/ReadVariableOpReadVariableOpbeta1_power*
_output_shapes
: *
dtype0

;Adam/update_sbb3/conv3/w/ResourceApplyAdam/ReadVariableOp_1ReadVariableOpbeta2_power*
_output_shapes
: *
dtype0
´
*Adam/update_sbb3/conv3/w/ResourceApplyAdamResourceApplyAdamsbb3/conv3/wsbb3/conv3/w/Adamsbb3/conv3/w/Adam_19Adam/update_sbb3/conv3/w/ResourceApplyAdam/ReadVariableOp;Adam/update_sbb3/conv3/w/ResourceApplyAdam/ReadVariableOp_1ExponentialDecay
Adam/beta1
Adam/beta2Adam/epsilon;gradients/sbb3/conv3/Conv2D_grad/tuple/control_dependency_1*
T0*
_class
loc:@sbb3/conv3/w*&
 _has_manual_control_dependencies(
}
9Adam/update_sbb3/conv3/b/ResourceApplyAdam/ReadVariableOpReadVariableOpbeta1_power*
_output_shapes
: *
dtype0

;Adam/update_sbb3/conv3/b/ResourceApplyAdam/ReadVariableOp_1ReadVariableOpbeta2_power*
_output_shapes
: *
dtype0
ľ
*Adam/update_sbb3/conv3/b/ResourceApplyAdamResourceApplyAdamsbb3/conv3/bsbb3/conv3/b/Adamsbb3/conv3/b/Adam_19Adam/update_sbb3/conv3/b/ResourceApplyAdam/ReadVariableOp;Adam/update_sbb3/conv3/b/ResourceApplyAdam/ReadVariableOp_1ExponentialDecay
Adam/beta1
Adam/beta2Adam/epsilon<gradients/sbb3/conv3/BiasAdd_grad/tuple/control_dependency_1*
T0*
_class
loc:@sbb3/conv3/b*&
 _has_manual_control_dependencies(
}
9Adam/update_sbb3/conv4/w/ResourceApplyAdam/ReadVariableOpReadVariableOpbeta1_power*
_output_shapes
: *
dtype0

;Adam/update_sbb3/conv4/w/ResourceApplyAdam/ReadVariableOp_1ReadVariableOpbeta2_power*
_output_shapes
: *
dtype0
´
*Adam/update_sbb3/conv4/w/ResourceApplyAdamResourceApplyAdamsbb3/conv4/wsbb3/conv4/w/Adamsbb3/conv4/w/Adam_19Adam/update_sbb3/conv4/w/ResourceApplyAdam/ReadVariableOp;Adam/update_sbb3/conv4/w/ResourceApplyAdam/ReadVariableOp_1ExponentialDecay
Adam/beta1
Adam/beta2Adam/epsilon;gradients/sbb3/conv4/Conv2D_grad/tuple/control_dependency_1*
T0*
_class
loc:@sbb3/conv4/w*&
 _has_manual_control_dependencies(
}
9Adam/update_sbb3/conv4/b/ResourceApplyAdam/ReadVariableOpReadVariableOpbeta1_power*
_output_shapes
: *
dtype0

;Adam/update_sbb3/conv4/b/ResourceApplyAdam/ReadVariableOp_1ReadVariableOpbeta2_power*
_output_shapes
: *
dtype0
ľ
*Adam/update_sbb3/conv4/b/ResourceApplyAdamResourceApplyAdamsbb3/conv4/bsbb3/conv4/b/Adamsbb3/conv4/b/Adam_19Adam/update_sbb3/conv4/b/ResourceApplyAdam/ReadVariableOp;Adam/update_sbb3/conv4/b/ResourceApplyAdam/ReadVariableOp_1ExponentialDecay
Adam/beta1
Adam/beta2Adam/epsilon<gradients/sbb3/conv4/BiasAdd_grad/tuple/control_dependency_1*
T0*
_class
loc:@sbb3/conv4/b*&
 _has_manual_control_dependencies(

KAdam/update_sbb3/batch_normalization/gamma/ResourceApplyAdam/ReadVariableOpReadVariableOpbeta1_power*
_output_shapes
: *
dtype0

MAdam/update_sbb3/batch_normalization/gamma/ResourceApplyAdam/ReadVariableOp_1ReadVariableOpbeta2_power*
_output_shapes
: *
dtype0
Ę
<Adam/update_sbb3/batch_normalization/gamma/ResourceApplyAdamResourceApplyAdamsbb3/batch_normalization/gamma#sbb3/batch_normalization/gamma/Adam%sbb3/batch_normalization/gamma/Adam_1KAdam/update_sbb3/batch_normalization/gamma/ResourceApplyAdam/ReadVariableOpMAdam/update_sbb3/batch_normalization/gamma/ResourceApplyAdam/ReadVariableOp_1ExponentialDecay
Adam/beta1
Adam/beta2Adam/epsilonSgradients/sbb3/batch_normalization/FusedBatchNormV3_grad/tuple/control_dependency_1*
T0*1
_class'
%#loc:@sbb3/batch_normalization/gamma*&
 _has_manual_control_dependencies(

JAdam/update_sbb3/batch_normalization/beta/ResourceApplyAdam/ReadVariableOpReadVariableOpbeta1_power*
_output_shapes
: *
dtype0

LAdam/update_sbb3/batch_normalization/beta/ResourceApplyAdam/ReadVariableOp_1ReadVariableOpbeta2_power*
_output_shapes
: *
dtype0
Ă
;Adam/update_sbb3/batch_normalization/beta/ResourceApplyAdamResourceApplyAdamsbb3/batch_normalization/beta"sbb3/batch_normalization/beta/Adam$sbb3/batch_normalization/beta/Adam_1JAdam/update_sbb3/batch_normalization/beta/ResourceApplyAdam/ReadVariableOpLAdam/update_sbb3/batch_normalization/beta/ResourceApplyAdam/ReadVariableOp_1ExponentialDecay
Adam/beta1
Adam/beta2Adam/epsilonSgradients/sbb3/batch_normalization/FusedBatchNormV3_grad/tuple/control_dependency_2*
T0*0
_class&
$"loc:@sbb3/batch_normalization/beta*&
 _has_manual_control_dependencies(
z
6Adam/update_conv_d1/w/ResourceApplyAdam/ReadVariableOpReadVariableOpbeta1_power*
_output_shapes
: *
dtype0
|
8Adam/update_conv_d1/w/ResourceApplyAdam/ReadVariableOp_1ReadVariableOpbeta2_power*
_output_shapes
: *
dtype0

'Adam/update_conv_d1/w/ResourceApplyAdamResourceApplyAdam	conv_d1/wconv_d1/w/Adamconv_d1/w/Adam_16Adam/update_conv_d1/w/ResourceApplyAdam/ReadVariableOp8Adam/update_conv_d1/w/ResourceApplyAdam/ReadVariableOp_1ExponentialDecay
Adam/beta1
Adam/beta2Adam/epsilon8gradients/conv_d1/Conv2D_grad/tuple/control_dependency_1*
T0*
_class
loc:@conv_d1/w*&
 _has_manual_control_dependencies(
z
6Adam/update_conv_d1/b/ResourceApplyAdam/ReadVariableOpReadVariableOpbeta1_power*
_output_shapes
: *
dtype0
|
8Adam/update_conv_d1/b/ResourceApplyAdam/ReadVariableOp_1ReadVariableOpbeta2_power*
_output_shapes
: *
dtype0

'Adam/update_conv_d1/b/ResourceApplyAdamResourceApplyAdam	conv_d1/bconv_d1/b/Adamconv_d1/b/Adam_16Adam/update_conv_d1/b/ResourceApplyAdam/ReadVariableOp8Adam/update_conv_d1/b/ResourceApplyAdam/ReadVariableOp_1ExponentialDecay
Adam/beta1
Adam/beta2Adam/epsilon9gradients/conv_d1/BiasAdd_grad/tuple/control_dependency_1*
T0*
_class
loc:@conv_d1/b*&
 _has_manual_control_dependencies(

HAdam/update_batch_normalization_1/gamma/ResourceApplyAdam/ReadVariableOpReadVariableOpbeta1_power*
_output_shapes
: *
dtype0

JAdam/update_batch_normalization_1/gamma/ResourceApplyAdam/ReadVariableOp_1ReadVariableOpbeta2_power*
_output_shapes
: *
dtype0
˛
9Adam/update_batch_normalization_1/gamma/ResourceApplyAdamResourceApplyAdambatch_normalization_1/gamma batch_normalization_1/gamma/Adam"batch_normalization_1/gamma/Adam_1HAdam/update_batch_normalization_1/gamma/ResourceApplyAdam/ReadVariableOpJAdam/update_batch_normalization_1/gamma/ResourceApplyAdam/ReadVariableOp_1ExponentialDecay
Adam/beta1
Adam/beta2Adam/epsilonPgradients/batch_normalization_1/FusedBatchNormV3_grad/tuple/control_dependency_1*
T0*.
_class$
" loc:@batch_normalization_1/gamma*&
 _has_manual_control_dependencies(

GAdam/update_batch_normalization_1/beta/ResourceApplyAdam/ReadVariableOpReadVariableOpbeta1_power*
_output_shapes
: *
dtype0

IAdam/update_batch_normalization_1/beta/ResourceApplyAdam/ReadVariableOp_1ReadVariableOpbeta2_power*
_output_shapes
: *
dtype0
Ť
8Adam/update_batch_normalization_1/beta/ResourceApplyAdamResourceApplyAdambatch_normalization_1/betabatch_normalization_1/beta/Adam!batch_normalization_1/beta/Adam_1GAdam/update_batch_normalization_1/beta/ResourceApplyAdam/ReadVariableOpIAdam/update_batch_normalization_1/beta/ResourceApplyAdam/ReadVariableOp_1ExponentialDecay
Adam/beta1
Adam/beta2Adam/epsilonPgradients/batch_normalization_1/FusedBatchNormV3_grad/tuple/control_dependency_2*
T0*-
_class#
!loc:@batch_normalization_1/beta*&
 _has_manual_control_dependencies(
z
6Adam/update_conv_d2/w/ResourceApplyAdam/ReadVariableOpReadVariableOpbeta1_power*
_output_shapes
: *
dtype0
|
8Adam/update_conv_d2/w/ResourceApplyAdam/ReadVariableOp_1ReadVariableOpbeta2_power*
_output_shapes
: *
dtype0

'Adam/update_conv_d2/w/ResourceApplyAdamResourceApplyAdam	conv_d2/wconv_d2/w/Adamconv_d2/w/Adam_16Adam/update_conv_d2/w/ResourceApplyAdam/ReadVariableOp8Adam/update_conv_d2/w/ResourceApplyAdam/ReadVariableOp_1ExponentialDecay
Adam/beta1
Adam/beta2Adam/epsilon8gradients/conv_d2/Conv2D_grad/tuple/control_dependency_1*
T0*
_class
loc:@conv_d2/w*&
 _has_manual_control_dependencies(
z
6Adam/update_conv_d2/b/ResourceApplyAdam/ReadVariableOpReadVariableOpbeta1_power*
_output_shapes
: *
dtype0
|
8Adam/update_conv_d2/b/ResourceApplyAdam/ReadVariableOp_1ReadVariableOpbeta2_power*
_output_shapes
: *
dtype0

'Adam/update_conv_d2/b/ResourceApplyAdamResourceApplyAdam	conv_d2/bconv_d2/b/Adamconv_d2/b/Adam_16Adam/update_conv_d2/b/ResourceApplyAdam/ReadVariableOp8Adam/update_conv_d2/b/ResourceApplyAdam/ReadVariableOp_1ExponentialDecay
Adam/beta1
Adam/beta2Adam/epsilon9gradients/conv_d2/BiasAdd_grad/tuple/control_dependency_1*
T0*
_class
loc:@conv_d2/b*&
 _has_manual_control_dependencies(

HAdam/update_batch_normalization_2/gamma/ResourceApplyAdam/ReadVariableOpReadVariableOpbeta1_power*
_output_shapes
: *
dtype0

JAdam/update_batch_normalization_2/gamma/ResourceApplyAdam/ReadVariableOp_1ReadVariableOpbeta2_power*
_output_shapes
: *
dtype0
˛
9Adam/update_batch_normalization_2/gamma/ResourceApplyAdamResourceApplyAdambatch_normalization_2/gamma batch_normalization_2/gamma/Adam"batch_normalization_2/gamma/Adam_1HAdam/update_batch_normalization_2/gamma/ResourceApplyAdam/ReadVariableOpJAdam/update_batch_normalization_2/gamma/ResourceApplyAdam/ReadVariableOp_1ExponentialDecay
Adam/beta1
Adam/beta2Adam/epsilonPgradients/batch_normalization_2/FusedBatchNormV3_grad/tuple/control_dependency_1*
T0*.
_class$
" loc:@batch_normalization_2/gamma*&
 _has_manual_control_dependencies(

GAdam/update_batch_normalization_2/beta/ResourceApplyAdam/ReadVariableOpReadVariableOpbeta1_power*
_output_shapes
: *
dtype0

IAdam/update_batch_normalization_2/beta/ResourceApplyAdam/ReadVariableOp_1ReadVariableOpbeta2_power*
_output_shapes
: *
dtype0
Ť
8Adam/update_batch_normalization_2/beta/ResourceApplyAdamResourceApplyAdambatch_normalization_2/betabatch_normalization_2/beta/Adam!batch_normalization_2/beta/Adam_1GAdam/update_batch_normalization_2/beta/ResourceApplyAdam/ReadVariableOpIAdam/update_batch_normalization_2/beta/ResourceApplyAdam/ReadVariableOp_1ExponentialDecay
Adam/beta1
Adam/beta2Adam/epsilonPgradients/batch_normalization_2/FusedBatchNormV3_grad/tuple/control_dependency_2*
T0*-
_class#
!loc:@batch_normalization_2/beta*&
 _has_manual_control_dependencies(
{
7Adam/update_conv_out/w/ResourceApplyAdam/ReadVariableOpReadVariableOpbeta1_power*
_output_shapes
: *
dtype0
}
9Adam/update_conv_out/w/ResourceApplyAdam/ReadVariableOp_1ReadVariableOpbeta2_power*
_output_shapes
: *
dtype0
¤
(Adam/update_conv_out/w/ResourceApplyAdamResourceApplyAdam
conv_out/wconv_out/w/Adamconv_out/w/Adam_17Adam/update_conv_out/w/ResourceApplyAdam/ReadVariableOp9Adam/update_conv_out/w/ResourceApplyAdam/ReadVariableOp_1ExponentialDecay
Adam/beta1
Adam/beta2Adam/epsilon9gradients/conv_out/Conv2D_grad/tuple/control_dependency_1*
T0*
_class
loc:@conv_out/w*&
 _has_manual_control_dependencies(
{
7Adam/update_conv_out/b/ResourceApplyAdam/ReadVariableOpReadVariableOpbeta1_power*
_output_shapes
: *
dtype0
}
9Adam/update_conv_out/b/ResourceApplyAdam/ReadVariableOp_1ReadVariableOpbeta2_power*
_output_shapes
: *
dtype0
Ľ
(Adam/update_conv_out/b/ResourceApplyAdamResourceApplyAdam
conv_out/bconv_out/b/Adamconv_out/b/Adam_17Adam/update_conv_out/b/ResourceApplyAdam/ReadVariableOp9Adam/update_conv_out/b/ResourceApplyAdam/ReadVariableOp_1ExponentialDecay
Adam/beta1
Adam/beta2Adam/epsilon:gradients/conv_out/BiasAdd_grad/tuple/control_dependency_1*
T0*
_class
loc:@conv_out/b*&
 _has_manual_control_dependencies(
ľ
Adam/ReadVariableOpReadVariableOpbeta1_power7^Adam/update_batch_normalization/beta/ResourceApplyAdam8^Adam/update_batch_normalization/gamma/ResourceApplyAdam9^Adam/update_batch_normalization_1/beta/ResourceApplyAdam:^Adam/update_batch_normalization_1/gamma/ResourceApplyAdam9^Adam/update_batch_normalization_2/beta/ResourceApplyAdam:^Adam/update_batch_normalization_2/gamma/ResourceApplyAdam&^Adam/update_conv1/b/ResourceApplyAdam&^Adam/update_conv1/w/ResourceApplyAdam(^Adam/update_conv_d1/b/ResourceApplyAdam(^Adam/update_conv_d1/w/ResourceApplyAdam(^Adam/update_conv_d2/b/ResourceApplyAdam(^Adam/update_conv_d2/w/ResourceApplyAdam)^Adam/update_conv_out/b/ResourceApplyAdam)^Adam/update_conv_out/w/ResourceApplyAdam<^Adam/update_sbb1/batch_normalization/beta/ResourceApplyAdam=^Adam/update_sbb1/batch_normalization/gamma/ResourceApplyAdam+^Adam/update_sbb1/conv1/b/ResourceApplyAdam+^Adam/update_sbb1/conv1/w/ResourceApplyAdam+^Adam/update_sbb1/conv2/b/ResourceApplyAdam+^Adam/update_sbb1/conv2/w/ResourceApplyAdam+^Adam/update_sbb1/conv3/b/ResourceApplyAdam+^Adam/update_sbb1/conv3/w/ResourceApplyAdam+^Adam/update_sbb1/conv4/b/ResourceApplyAdam+^Adam/update_sbb1/conv4/w/ResourceApplyAdam<^Adam/update_sbb2/batch_normalization/beta/ResourceApplyAdam=^Adam/update_sbb2/batch_normalization/gamma/ResourceApplyAdam+^Adam/update_sbb2/conv1/b/ResourceApplyAdam+^Adam/update_sbb2/conv1/w/ResourceApplyAdam+^Adam/update_sbb2/conv2/b/ResourceApplyAdam+^Adam/update_sbb2/conv2/w/ResourceApplyAdam+^Adam/update_sbb2/conv3/b/ResourceApplyAdam+^Adam/update_sbb2/conv3/w/ResourceApplyAdam+^Adam/update_sbb2/conv4/b/ResourceApplyAdam+^Adam/update_sbb2/conv4/w/ResourceApplyAdam<^Adam/update_sbb3/batch_normalization/beta/ResourceApplyAdam=^Adam/update_sbb3/batch_normalization/gamma/ResourceApplyAdam+^Adam/update_sbb3/conv1/b/ResourceApplyAdam+^Adam/update_sbb3/conv1/w/ResourceApplyAdam+^Adam/update_sbb3/conv2/b/ResourceApplyAdam+^Adam/update_sbb3/conv2/w/ResourceApplyAdam+^Adam/update_sbb3/conv3/b/ResourceApplyAdam+^Adam/update_sbb3/conv3/w/ResourceApplyAdam+^Adam/update_sbb3/conv4/b/ResourceApplyAdam+^Adam/update_sbb3/conv4/w/ResourceApplyAdam*
_output_shapes
: *
dtype0
~
Adam/mulMulAdam/ReadVariableOp
Adam/beta1*
T0*+
_class!
loc:@batch_normalization/beta*
_output_shapes
: 
¸
Adam/AssignVariableOpAssignVariableOpbeta1_powerAdam/mul*+
_class!
loc:@batch_normalization/beta*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape(
ü
Adam/ReadVariableOp_1ReadVariableOpbeta1_power^Adam/AssignVariableOp7^Adam/update_batch_normalization/beta/ResourceApplyAdam8^Adam/update_batch_normalization/gamma/ResourceApplyAdam9^Adam/update_batch_normalization_1/beta/ResourceApplyAdam:^Adam/update_batch_normalization_1/gamma/ResourceApplyAdam9^Adam/update_batch_normalization_2/beta/ResourceApplyAdam:^Adam/update_batch_normalization_2/gamma/ResourceApplyAdam&^Adam/update_conv1/b/ResourceApplyAdam&^Adam/update_conv1/w/ResourceApplyAdam(^Adam/update_conv_d1/b/ResourceApplyAdam(^Adam/update_conv_d1/w/ResourceApplyAdam(^Adam/update_conv_d2/b/ResourceApplyAdam(^Adam/update_conv_d2/w/ResourceApplyAdam)^Adam/update_conv_out/b/ResourceApplyAdam)^Adam/update_conv_out/w/ResourceApplyAdam<^Adam/update_sbb1/batch_normalization/beta/ResourceApplyAdam=^Adam/update_sbb1/batch_normalization/gamma/ResourceApplyAdam+^Adam/update_sbb1/conv1/b/ResourceApplyAdam+^Adam/update_sbb1/conv1/w/ResourceApplyAdam+^Adam/update_sbb1/conv2/b/ResourceApplyAdam+^Adam/update_sbb1/conv2/w/ResourceApplyAdam+^Adam/update_sbb1/conv3/b/ResourceApplyAdam+^Adam/update_sbb1/conv3/w/ResourceApplyAdam+^Adam/update_sbb1/conv4/b/ResourceApplyAdam+^Adam/update_sbb1/conv4/w/ResourceApplyAdam<^Adam/update_sbb2/batch_normalization/beta/ResourceApplyAdam=^Adam/update_sbb2/batch_normalization/gamma/ResourceApplyAdam+^Adam/update_sbb2/conv1/b/ResourceApplyAdam+^Adam/update_sbb2/conv1/w/ResourceApplyAdam+^Adam/update_sbb2/conv2/b/ResourceApplyAdam+^Adam/update_sbb2/conv2/w/ResourceApplyAdam+^Adam/update_sbb2/conv3/b/ResourceApplyAdam+^Adam/update_sbb2/conv3/w/ResourceApplyAdam+^Adam/update_sbb2/conv4/b/ResourceApplyAdam+^Adam/update_sbb2/conv4/w/ResourceApplyAdam<^Adam/update_sbb3/batch_normalization/beta/ResourceApplyAdam=^Adam/update_sbb3/batch_normalization/gamma/ResourceApplyAdam+^Adam/update_sbb3/conv1/b/ResourceApplyAdam+^Adam/update_sbb3/conv1/w/ResourceApplyAdam+^Adam/update_sbb3/conv2/b/ResourceApplyAdam+^Adam/update_sbb3/conv2/w/ResourceApplyAdam+^Adam/update_sbb3/conv3/b/ResourceApplyAdam+^Adam/update_sbb3/conv3/w/ResourceApplyAdam+^Adam/update_sbb3/conv4/b/ResourceApplyAdam+^Adam/update_sbb3/conv4/w/ResourceApplyAdam*+
_class!
loc:@batch_normalization/beta*
_output_shapes
: *
dtype0
ˇ
Adam/ReadVariableOp_2ReadVariableOpbeta2_power7^Adam/update_batch_normalization/beta/ResourceApplyAdam8^Adam/update_batch_normalization/gamma/ResourceApplyAdam9^Adam/update_batch_normalization_1/beta/ResourceApplyAdam:^Adam/update_batch_normalization_1/gamma/ResourceApplyAdam9^Adam/update_batch_normalization_2/beta/ResourceApplyAdam:^Adam/update_batch_normalization_2/gamma/ResourceApplyAdam&^Adam/update_conv1/b/ResourceApplyAdam&^Adam/update_conv1/w/ResourceApplyAdam(^Adam/update_conv_d1/b/ResourceApplyAdam(^Adam/update_conv_d1/w/ResourceApplyAdam(^Adam/update_conv_d2/b/ResourceApplyAdam(^Adam/update_conv_d2/w/ResourceApplyAdam)^Adam/update_conv_out/b/ResourceApplyAdam)^Adam/update_conv_out/w/ResourceApplyAdam<^Adam/update_sbb1/batch_normalization/beta/ResourceApplyAdam=^Adam/update_sbb1/batch_normalization/gamma/ResourceApplyAdam+^Adam/update_sbb1/conv1/b/ResourceApplyAdam+^Adam/update_sbb1/conv1/w/ResourceApplyAdam+^Adam/update_sbb1/conv2/b/ResourceApplyAdam+^Adam/update_sbb1/conv2/w/ResourceApplyAdam+^Adam/update_sbb1/conv3/b/ResourceApplyAdam+^Adam/update_sbb1/conv3/w/ResourceApplyAdam+^Adam/update_sbb1/conv4/b/ResourceApplyAdam+^Adam/update_sbb1/conv4/w/ResourceApplyAdam<^Adam/update_sbb2/batch_normalization/beta/ResourceApplyAdam=^Adam/update_sbb2/batch_normalization/gamma/ResourceApplyAdam+^Adam/update_sbb2/conv1/b/ResourceApplyAdam+^Adam/update_sbb2/conv1/w/ResourceApplyAdam+^Adam/update_sbb2/conv2/b/ResourceApplyAdam+^Adam/update_sbb2/conv2/w/ResourceApplyAdam+^Adam/update_sbb2/conv3/b/ResourceApplyAdam+^Adam/update_sbb2/conv3/w/ResourceApplyAdam+^Adam/update_sbb2/conv4/b/ResourceApplyAdam+^Adam/update_sbb2/conv4/w/ResourceApplyAdam<^Adam/update_sbb3/batch_normalization/beta/ResourceApplyAdam=^Adam/update_sbb3/batch_normalization/gamma/ResourceApplyAdam+^Adam/update_sbb3/conv1/b/ResourceApplyAdam+^Adam/update_sbb3/conv1/w/ResourceApplyAdam+^Adam/update_sbb3/conv2/b/ResourceApplyAdam+^Adam/update_sbb3/conv2/w/ResourceApplyAdam+^Adam/update_sbb3/conv3/b/ResourceApplyAdam+^Adam/update_sbb3/conv3/w/ResourceApplyAdam+^Adam/update_sbb3/conv4/b/ResourceApplyAdam+^Adam/update_sbb3/conv4/w/ResourceApplyAdam*
_output_shapes
: *
dtype0


Adam/mul_1MulAdam/ReadVariableOp_2
Adam/beta2*
T0*+
_class!
loc:@batch_normalization/beta*
_output_shapes
: 
ź
Adam/AssignVariableOp_1AssignVariableOpbeta2_power
Adam/mul_1*+
_class!
loc:@batch_normalization/beta*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape(
ţ
Adam/ReadVariableOp_3ReadVariableOpbeta2_power^Adam/AssignVariableOp_17^Adam/update_batch_normalization/beta/ResourceApplyAdam8^Adam/update_batch_normalization/gamma/ResourceApplyAdam9^Adam/update_batch_normalization_1/beta/ResourceApplyAdam:^Adam/update_batch_normalization_1/gamma/ResourceApplyAdam9^Adam/update_batch_normalization_2/beta/ResourceApplyAdam:^Adam/update_batch_normalization_2/gamma/ResourceApplyAdam&^Adam/update_conv1/b/ResourceApplyAdam&^Adam/update_conv1/w/ResourceApplyAdam(^Adam/update_conv_d1/b/ResourceApplyAdam(^Adam/update_conv_d1/w/ResourceApplyAdam(^Adam/update_conv_d2/b/ResourceApplyAdam(^Adam/update_conv_d2/w/ResourceApplyAdam)^Adam/update_conv_out/b/ResourceApplyAdam)^Adam/update_conv_out/w/ResourceApplyAdam<^Adam/update_sbb1/batch_normalization/beta/ResourceApplyAdam=^Adam/update_sbb1/batch_normalization/gamma/ResourceApplyAdam+^Adam/update_sbb1/conv1/b/ResourceApplyAdam+^Adam/update_sbb1/conv1/w/ResourceApplyAdam+^Adam/update_sbb1/conv2/b/ResourceApplyAdam+^Adam/update_sbb1/conv2/w/ResourceApplyAdam+^Adam/update_sbb1/conv3/b/ResourceApplyAdam+^Adam/update_sbb1/conv3/w/ResourceApplyAdam+^Adam/update_sbb1/conv4/b/ResourceApplyAdam+^Adam/update_sbb1/conv4/w/ResourceApplyAdam<^Adam/update_sbb2/batch_normalization/beta/ResourceApplyAdam=^Adam/update_sbb2/batch_normalization/gamma/ResourceApplyAdam+^Adam/update_sbb2/conv1/b/ResourceApplyAdam+^Adam/update_sbb2/conv1/w/ResourceApplyAdam+^Adam/update_sbb2/conv2/b/ResourceApplyAdam+^Adam/update_sbb2/conv2/w/ResourceApplyAdam+^Adam/update_sbb2/conv3/b/ResourceApplyAdam+^Adam/update_sbb2/conv3/w/ResourceApplyAdam+^Adam/update_sbb2/conv4/b/ResourceApplyAdam+^Adam/update_sbb2/conv4/w/ResourceApplyAdam<^Adam/update_sbb3/batch_normalization/beta/ResourceApplyAdam=^Adam/update_sbb3/batch_normalization/gamma/ResourceApplyAdam+^Adam/update_sbb3/conv1/b/ResourceApplyAdam+^Adam/update_sbb3/conv1/w/ResourceApplyAdam+^Adam/update_sbb3/conv2/b/ResourceApplyAdam+^Adam/update_sbb3/conv2/w/ResourceApplyAdam+^Adam/update_sbb3/conv3/b/ResourceApplyAdam+^Adam/update_sbb3/conv3/w/ResourceApplyAdam+^Adam/update_sbb3/conv4/b/ResourceApplyAdam+^Adam/update_sbb3/conv4/w/ResourceApplyAdam*+
_class!
loc:@batch_normalization/beta*
_output_shapes
: *
dtype0
Ë
Adam/updateNoOp^Adam/AssignVariableOp^Adam/AssignVariableOp_17^Adam/update_batch_normalization/beta/ResourceApplyAdam8^Adam/update_batch_normalization/gamma/ResourceApplyAdam9^Adam/update_batch_normalization_1/beta/ResourceApplyAdam:^Adam/update_batch_normalization_1/gamma/ResourceApplyAdam9^Adam/update_batch_normalization_2/beta/ResourceApplyAdam:^Adam/update_batch_normalization_2/gamma/ResourceApplyAdam&^Adam/update_conv1/b/ResourceApplyAdam&^Adam/update_conv1/w/ResourceApplyAdam(^Adam/update_conv_d1/b/ResourceApplyAdam(^Adam/update_conv_d1/w/ResourceApplyAdam(^Adam/update_conv_d2/b/ResourceApplyAdam(^Adam/update_conv_d2/w/ResourceApplyAdam)^Adam/update_conv_out/b/ResourceApplyAdam)^Adam/update_conv_out/w/ResourceApplyAdam<^Adam/update_sbb1/batch_normalization/beta/ResourceApplyAdam=^Adam/update_sbb1/batch_normalization/gamma/ResourceApplyAdam+^Adam/update_sbb1/conv1/b/ResourceApplyAdam+^Adam/update_sbb1/conv1/w/ResourceApplyAdam+^Adam/update_sbb1/conv2/b/ResourceApplyAdam+^Adam/update_sbb1/conv2/w/ResourceApplyAdam+^Adam/update_sbb1/conv3/b/ResourceApplyAdam+^Adam/update_sbb1/conv3/w/ResourceApplyAdam+^Adam/update_sbb1/conv4/b/ResourceApplyAdam+^Adam/update_sbb1/conv4/w/ResourceApplyAdam<^Adam/update_sbb2/batch_normalization/beta/ResourceApplyAdam=^Adam/update_sbb2/batch_normalization/gamma/ResourceApplyAdam+^Adam/update_sbb2/conv1/b/ResourceApplyAdam+^Adam/update_sbb2/conv1/w/ResourceApplyAdam+^Adam/update_sbb2/conv2/b/ResourceApplyAdam+^Adam/update_sbb2/conv2/w/ResourceApplyAdam+^Adam/update_sbb2/conv3/b/ResourceApplyAdam+^Adam/update_sbb2/conv3/w/ResourceApplyAdam+^Adam/update_sbb2/conv4/b/ResourceApplyAdam+^Adam/update_sbb2/conv4/w/ResourceApplyAdam<^Adam/update_sbb3/batch_normalization/beta/ResourceApplyAdam=^Adam/update_sbb3/batch_normalization/gamma/ResourceApplyAdam+^Adam/update_sbb3/conv1/b/ResourceApplyAdam+^Adam/update_sbb3/conv1/w/ResourceApplyAdam+^Adam/update_sbb3/conv2/b/ResourceApplyAdam+^Adam/update_sbb3/conv2/w/ResourceApplyAdam+^Adam/update_sbb3/conv3/b/ResourceApplyAdam+^Adam/update_sbb3/conv3/w/ResourceApplyAdam+^Adam/update_sbb3/conv4/b/ResourceApplyAdam+^Adam/update_sbb3/conv4/w/ResourceApplyAdam*&
 _has_manual_control_dependencies(
w

Adam/ConstConst^Adam/update*
_class
loc:@Variable*
_output_shapes
: *
dtype0*
value	B :
[
AdamAssignAddVariableOpVariable
Adam/Const*
_class
loc:@Variable*
dtype0
Ŕ#
initNoOp^Variable/Assign%^batch_normalization/beta/Adam/Assign'^batch_normalization/beta/Adam_1/Assign ^batch_normalization/beta/Assign&^batch_normalization/gamma/Adam/Assign(^batch_normalization/gamma/Adam_1/Assign!^batch_normalization/gamma/Assign'^batch_normalization/moving_mean/Assign+^batch_normalization/moving_variance/Assign'^batch_normalization_1/beta/Adam/Assign)^batch_normalization_1/beta/Adam_1/Assign"^batch_normalization_1/beta/Assign(^batch_normalization_1/gamma/Adam/Assign*^batch_normalization_1/gamma/Adam_1/Assign#^batch_normalization_1/gamma/Assign)^batch_normalization_1/moving_mean/Assign-^batch_normalization_1/moving_variance/Assign'^batch_normalization_2/beta/Adam/Assign)^batch_normalization_2/beta/Adam_1/Assign"^batch_normalization_2/beta/Assign(^batch_normalization_2/gamma/Adam/Assign*^batch_normalization_2/gamma/Adam_1/Assign#^batch_normalization_2/gamma/Assign)^batch_normalization_2/moving_mean/Assign-^batch_normalization_2/moving_variance/Assign^beta1_power/Assign^beta2_power/Assign^conv1/b/Adam/Assign^conv1/b/Adam_1/Assign^conv1/b/Assign^conv1/w/Adam/Assign^conv1/w/Adam_1/Assign^conv1/w/Assign^conv_d1/b/Adam/Assign^conv_d1/b/Adam_1/Assign^conv_d1/b/Assign^conv_d1/w/Adam/Assign^conv_d1/w/Adam_1/Assign^conv_d1/w/Assign^conv_d2/b/Adam/Assign^conv_d2/b/Adam_1/Assign^conv_d2/b/Assign^conv_d2/w/Adam/Assign^conv_d2/w/Adam_1/Assign^conv_d2/w/Assign^conv_out/b/Adam/Assign^conv_out/b/Adam_1/Assign^conv_out/b/Assign^conv_out/w/Adam/Assign^conv_out/w/Adam_1/Assign^conv_out/w/Assign*^sbb1/batch_normalization/beta/Adam/Assign,^sbb1/batch_normalization/beta/Adam_1/Assign%^sbb1/batch_normalization/beta/Assign+^sbb1/batch_normalization/gamma/Adam/Assign-^sbb1/batch_normalization/gamma/Adam_1/Assign&^sbb1/batch_normalization/gamma/Assign,^sbb1/batch_normalization/moving_mean/Assign0^sbb1/batch_normalization/moving_variance/Assign^sbb1/conv1/b/Adam/Assign^sbb1/conv1/b/Adam_1/Assign^sbb1/conv1/b/Assign^sbb1/conv1/w/Adam/Assign^sbb1/conv1/w/Adam_1/Assign^sbb1/conv1/w/Assign^sbb1/conv2/b/Adam/Assign^sbb1/conv2/b/Adam_1/Assign^sbb1/conv2/b/Assign^sbb1/conv2/w/Adam/Assign^sbb1/conv2/w/Adam_1/Assign^sbb1/conv2/w/Assign^sbb1/conv3/b/Adam/Assign^sbb1/conv3/b/Adam_1/Assign^sbb1/conv3/b/Assign^sbb1/conv3/w/Adam/Assign^sbb1/conv3/w/Adam_1/Assign^sbb1/conv3/w/Assign^sbb1/conv4/b/Adam/Assign^sbb1/conv4/b/Adam_1/Assign^sbb1/conv4/b/Assign^sbb1/conv4/w/Adam/Assign^sbb1/conv4/w/Adam_1/Assign^sbb1/conv4/w/Assign*^sbb2/batch_normalization/beta/Adam/Assign,^sbb2/batch_normalization/beta/Adam_1/Assign%^sbb2/batch_normalization/beta/Assign+^sbb2/batch_normalization/gamma/Adam/Assign-^sbb2/batch_normalization/gamma/Adam_1/Assign&^sbb2/batch_normalization/gamma/Assign,^sbb2/batch_normalization/moving_mean/Assign0^sbb2/batch_normalization/moving_variance/Assign^sbb2/conv1/b/Adam/Assign^sbb2/conv1/b/Adam_1/Assign^sbb2/conv1/b/Assign^sbb2/conv1/w/Adam/Assign^sbb2/conv1/w/Adam_1/Assign^sbb2/conv1/w/Assign^sbb2/conv2/b/Adam/Assign^sbb2/conv2/b/Adam_1/Assign^sbb2/conv2/b/Assign^sbb2/conv2/w/Adam/Assign^sbb2/conv2/w/Adam_1/Assign^sbb2/conv2/w/Assign^sbb2/conv3/b/Adam/Assign^sbb2/conv3/b/Adam_1/Assign^sbb2/conv3/b/Assign^sbb2/conv3/w/Adam/Assign^sbb2/conv3/w/Adam_1/Assign^sbb2/conv3/w/Assign^sbb2/conv4/b/Adam/Assign^sbb2/conv4/b/Adam_1/Assign^sbb2/conv4/b/Assign^sbb2/conv4/w/Adam/Assign^sbb2/conv4/w/Adam_1/Assign^sbb2/conv4/w/Assign*^sbb3/batch_normalization/beta/Adam/Assign,^sbb3/batch_normalization/beta/Adam_1/Assign%^sbb3/batch_normalization/beta/Assign+^sbb3/batch_normalization/gamma/Adam/Assign-^sbb3/batch_normalization/gamma/Adam_1/Assign&^sbb3/batch_normalization/gamma/Assign,^sbb3/batch_normalization/moving_mean/Assign0^sbb3/batch_normalization/moving_variance/Assign^sbb3/conv1/b/Adam/Assign^sbb3/conv1/b/Adam_1/Assign^sbb3/conv1/b/Assign^sbb3/conv1/w/Adam/Assign^sbb3/conv1/w/Adam_1/Assign^sbb3/conv1/w/Assign^sbb3/conv2/b/Adam/Assign^sbb3/conv2/b/Adam_1/Assign^sbb3/conv2/b/Assign^sbb3/conv2/w/Adam/Assign^sbb3/conv2/w/Adam_1/Assign^sbb3/conv2/w/Assign^sbb3/conv3/b/Adam/Assign^sbb3/conv3/b/Adam_1/Assign^sbb3/conv3/b/Assign^sbb3/conv3/w/Adam/Assign^sbb3/conv3/w/Adam_1/Assign^sbb3/conv3/w/Assign^sbb3/conv4/b/Adam/Assign^sbb3/conv4/b/Adam_1/Assign^sbb3/conv4/b/Assign^sbb3/conv4/w/Adam/Assign^sbb3/conv4/w/Adam_1/Assign^sbb3/conv4/w/Assign
Â#
init_1NoOp^Variable/Assign%^batch_normalization/beta/Adam/Assign'^batch_normalization/beta/Adam_1/Assign ^batch_normalization/beta/Assign&^batch_normalization/gamma/Adam/Assign(^batch_normalization/gamma/Adam_1/Assign!^batch_normalization/gamma/Assign'^batch_normalization/moving_mean/Assign+^batch_normalization/moving_variance/Assign'^batch_normalization_1/beta/Adam/Assign)^batch_normalization_1/beta/Adam_1/Assign"^batch_normalization_1/beta/Assign(^batch_normalization_1/gamma/Adam/Assign*^batch_normalization_1/gamma/Adam_1/Assign#^batch_normalization_1/gamma/Assign)^batch_normalization_1/moving_mean/Assign-^batch_normalization_1/moving_variance/Assign'^batch_normalization_2/beta/Adam/Assign)^batch_normalization_2/beta/Adam_1/Assign"^batch_normalization_2/beta/Assign(^batch_normalization_2/gamma/Adam/Assign*^batch_normalization_2/gamma/Adam_1/Assign#^batch_normalization_2/gamma/Assign)^batch_normalization_2/moving_mean/Assign-^batch_normalization_2/moving_variance/Assign^beta1_power/Assign^beta2_power/Assign^conv1/b/Adam/Assign^conv1/b/Adam_1/Assign^conv1/b/Assign^conv1/w/Adam/Assign^conv1/w/Adam_1/Assign^conv1/w/Assign^conv_d1/b/Adam/Assign^conv_d1/b/Adam_1/Assign^conv_d1/b/Assign^conv_d1/w/Adam/Assign^conv_d1/w/Adam_1/Assign^conv_d1/w/Assign^conv_d2/b/Adam/Assign^conv_d2/b/Adam_1/Assign^conv_d2/b/Assign^conv_d2/w/Adam/Assign^conv_d2/w/Adam_1/Assign^conv_d2/w/Assign^conv_out/b/Adam/Assign^conv_out/b/Adam_1/Assign^conv_out/b/Assign^conv_out/w/Adam/Assign^conv_out/w/Adam_1/Assign^conv_out/w/Assign*^sbb1/batch_normalization/beta/Adam/Assign,^sbb1/batch_normalization/beta/Adam_1/Assign%^sbb1/batch_normalization/beta/Assign+^sbb1/batch_normalization/gamma/Adam/Assign-^sbb1/batch_normalization/gamma/Adam_1/Assign&^sbb1/batch_normalization/gamma/Assign,^sbb1/batch_normalization/moving_mean/Assign0^sbb1/batch_normalization/moving_variance/Assign^sbb1/conv1/b/Adam/Assign^sbb1/conv1/b/Adam_1/Assign^sbb1/conv1/b/Assign^sbb1/conv1/w/Adam/Assign^sbb1/conv1/w/Adam_1/Assign^sbb1/conv1/w/Assign^sbb1/conv2/b/Adam/Assign^sbb1/conv2/b/Adam_1/Assign^sbb1/conv2/b/Assign^sbb1/conv2/w/Adam/Assign^sbb1/conv2/w/Adam_1/Assign^sbb1/conv2/w/Assign^sbb1/conv3/b/Adam/Assign^sbb1/conv3/b/Adam_1/Assign^sbb1/conv3/b/Assign^sbb1/conv3/w/Adam/Assign^sbb1/conv3/w/Adam_1/Assign^sbb1/conv3/w/Assign^sbb1/conv4/b/Adam/Assign^sbb1/conv4/b/Adam_1/Assign^sbb1/conv4/b/Assign^sbb1/conv4/w/Adam/Assign^sbb1/conv4/w/Adam_1/Assign^sbb1/conv4/w/Assign*^sbb2/batch_normalization/beta/Adam/Assign,^sbb2/batch_normalization/beta/Adam_1/Assign%^sbb2/batch_normalization/beta/Assign+^sbb2/batch_normalization/gamma/Adam/Assign-^sbb2/batch_normalization/gamma/Adam_1/Assign&^sbb2/batch_normalization/gamma/Assign,^sbb2/batch_normalization/moving_mean/Assign0^sbb2/batch_normalization/moving_variance/Assign^sbb2/conv1/b/Adam/Assign^sbb2/conv1/b/Adam_1/Assign^sbb2/conv1/b/Assign^sbb2/conv1/w/Adam/Assign^sbb2/conv1/w/Adam_1/Assign^sbb2/conv1/w/Assign^sbb2/conv2/b/Adam/Assign^sbb2/conv2/b/Adam_1/Assign^sbb2/conv2/b/Assign^sbb2/conv2/w/Adam/Assign^sbb2/conv2/w/Adam_1/Assign^sbb2/conv2/w/Assign^sbb2/conv3/b/Adam/Assign^sbb2/conv3/b/Adam_1/Assign^sbb2/conv3/b/Assign^sbb2/conv3/w/Adam/Assign^sbb2/conv3/w/Adam_1/Assign^sbb2/conv3/w/Assign^sbb2/conv4/b/Adam/Assign^sbb2/conv4/b/Adam_1/Assign^sbb2/conv4/b/Assign^sbb2/conv4/w/Adam/Assign^sbb2/conv4/w/Adam_1/Assign^sbb2/conv4/w/Assign*^sbb3/batch_normalization/beta/Adam/Assign,^sbb3/batch_normalization/beta/Adam_1/Assign%^sbb3/batch_normalization/beta/Assign+^sbb3/batch_normalization/gamma/Adam/Assign-^sbb3/batch_normalization/gamma/Adam_1/Assign&^sbb3/batch_normalization/gamma/Assign,^sbb3/batch_normalization/moving_mean/Assign0^sbb3/batch_normalization/moving_variance/Assign^sbb3/conv1/b/Adam/Assign^sbb3/conv1/b/Adam_1/Assign^sbb3/conv1/b/Assign^sbb3/conv1/w/Adam/Assign^sbb3/conv1/w/Adam_1/Assign^sbb3/conv1/w/Assign^sbb3/conv2/b/Adam/Assign^sbb3/conv2/b/Adam_1/Assign^sbb3/conv2/b/Assign^sbb3/conv2/w/Adam/Assign^sbb3/conv2/w/Adam_1/Assign^sbb3/conv2/w/Assign^sbb3/conv3/b/Adam/Assign^sbb3/conv3/b/Adam_1/Assign^sbb3/conv3/b/Assign^sbb3/conv3/w/Adam/Assign^sbb3/conv3/w/Adam_1/Assign^sbb3/conv3/w/Assign^sbb3/conv4/b/Adam/Assign^sbb3/conv4/b/Adam_1/Assign^sbb3/conv4/b/Assign^sbb3/conv4/w/Adam/Assign^sbb3/conv4/w/Adam_1/Assign^sbb3/conv4/w/Assign
Y
save/filename/inputConst*
_output_shapes
: *
dtype0*
valueB Bmodel
n
save/filenamePlaceholderWithDefaultsave/filename/input*
_output_shapes
: *
dtype0*
shape: 
e

save/ConstPlaceholderWithDefaultsave/filename*
_output_shapes
: *
dtype0*
shape: 

save/SaveV2/tensor_namesConst*
_output_shapes	
:*
dtype0*˛
value¨BĽBVariableBbatch_normalization/betaBbatch_normalization/beta/AdamBbatch_normalization/beta/Adam_1Bbatch_normalization/gammaBbatch_normalization/gamma/AdamB batch_normalization/gamma/Adam_1Bbatch_normalization/moving_meanB#batch_normalization/moving_varianceBbatch_normalization_1/betaBbatch_normalization_1/beta/AdamB!batch_normalization_1/beta/Adam_1Bbatch_normalization_1/gammaB batch_normalization_1/gamma/AdamB"batch_normalization_1/gamma/Adam_1B!batch_normalization_1/moving_meanB%batch_normalization_1/moving_varianceBbatch_normalization_2/betaBbatch_normalization_2/beta/AdamB!batch_normalization_2/beta/Adam_1Bbatch_normalization_2/gammaB batch_normalization_2/gamma/AdamB"batch_normalization_2/gamma/Adam_1B!batch_normalization_2/moving_meanB%batch_normalization_2/moving_varianceBbeta1_powerBbeta2_powerBconv1/bBconv1/b/AdamBconv1/b/Adam_1Bconv1/wBconv1/w/AdamBconv1/w/Adam_1B	conv_d1/bBconv_d1/b/AdamBconv_d1/b/Adam_1B	conv_d1/wBconv_d1/w/AdamBconv_d1/w/Adam_1B	conv_d2/bBconv_d2/b/AdamBconv_d2/b/Adam_1B	conv_d2/wBconv_d2/w/AdamBconv_d2/w/Adam_1B
conv_out/bBconv_out/b/AdamBconv_out/b/Adam_1B
conv_out/wBconv_out/w/AdamBconv_out/w/Adam_1Bsbb1/batch_normalization/betaB"sbb1/batch_normalization/beta/AdamB$sbb1/batch_normalization/beta/Adam_1Bsbb1/batch_normalization/gammaB#sbb1/batch_normalization/gamma/AdamB%sbb1/batch_normalization/gamma/Adam_1B$sbb1/batch_normalization/moving_meanB(sbb1/batch_normalization/moving_varianceBsbb1/conv1/bBsbb1/conv1/b/AdamBsbb1/conv1/b/Adam_1Bsbb1/conv1/wBsbb1/conv1/w/AdamBsbb1/conv1/w/Adam_1Bsbb1/conv2/bBsbb1/conv2/b/AdamBsbb1/conv2/b/Adam_1Bsbb1/conv2/wBsbb1/conv2/w/AdamBsbb1/conv2/w/Adam_1Bsbb1/conv3/bBsbb1/conv3/b/AdamBsbb1/conv3/b/Adam_1Bsbb1/conv3/wBsbb1/conv3/w/AdamBsbb1/conv3/w/Adam_1Bsbb1/conv4/bBsbb1/conv4/b/AdamBsbb1/conv4/b/Adam_1Bsbb1/conv4/wBsbb1/conv4/w/AdamBsbb1/conv4/w/Adam_1Bsbb2/batch_normalization/betaB"sbb2/batch_normalization/beta/AdamB$sbb2/batch_normalization/beta/Adam_1Bsbb2/batch_normalization/gammaB#sbb2/batch_normalization/gamma/AdamB%sbb2/batch_normalization/gamma/Adam_1B$sbb2/batch_normalization/moving_meanB(sbb2/batch_normalization/moving_varianceBsbb2/conv1/bBsbb2/conv1/b/AdamBsbb2/conv1/b/Adam_1Bsbb2/conv1/wBsbb2/conv1/w/AdamBsbb2/conv1/w/Adam_1Bsbb2/conv2/bBsbb2/conv2/b/AdamBsbb2/conv2/b/Adam_1Bsbb2/conv2/wBsbb2/conv2/w/AdamBsbb2/conv2/w/Adam_1Bsbb2/conv3/bBsbb2/conv3/b/AdamBsbb2/conv3/b/Adam_1Bsbb2/conv3/wBsbb2/conv3/w/AdamBsbb2/conv3/w/Adam_1Bsbb2/conv4/bBsbb2/conv4/b/AdamBsbb2/conv4/b/Adam_1Bsbb2/conv4/wBsbb2/conv4/w/AdamBsbb2/conv4/w/Adam_1Bsbb3/batch_normalization/betaB"sbb3/batch_normalization/beta/AdamB$sbb3/batch_normalization/beta/Adam_1Bsbb3/batch_normalization/gammaB#sbb3/batch_normalization/gamma/AdamB%sbb3/batch_normalization/gamma/Adam_1B$sbb3/batch_normalization/moving_meanB(sbb3/batch_normalization/moving_varianceBsbb3/conv1/bBsbb3/conv1/b/AdamBsbb3/conv1/b/Adam_1Bsbb3/conv1/wBsbb3/conv1/w/AdamBsbb3/conv1/w/Adam_1Bsbb3/conv2/bBsbb3/conv2/b/AdamBsbb3/conv2/b/Adam_1Bsbb3/conv2/wBsbb3/conv2/w/AdamBsbb3/conv2/w/Adam_1Bsbb3/conv3/bBsbb3/conv3/b/AdamBsbb3/conv3/b/Adam_1Bsbb3/conv3/wBsbb3/conv3/w/AdamBsbb3/conv3/w/Adam_1Bsbb3/conv4/bBsbb3/conv4/b/AdamBsbb3/conv4/b/Adam_1Bsbb3/conv4/wBsbb3/conv4/w/AdamBsbb3/conv4/w/Adam_1

save/SaveV2/shape_and_slicesConst*
_output_shapes	
:*
dtype0*ź
value˛BŻB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
Ŕ3
save/SaveV2SaveV2
save/Constsave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesVariable/Read/ReadVariableOp,batch_normalization/beta/Read/ReadVariableOp1batch_normalization/beta/Adam/Read/ReadVariableOp3batch_normalization/beta/Adam_1/Read/ReadVariableOp-batch_normalization/gamma/Read/ReadVariableOp2batch_normalization/gamma/Adam/Read/ReadVariableOp4batch_normalization/gamma/Adam_1/Read/ReadVariableOp3batch_normalization/moving_mean/Read/ReadVariableOp7batch_normalization/moving_variance/Read/ReadVariableOp.batch_normalization_1/beta/Read/ReadVariableOp3batch_normalization_1/beta/Adam/Read/ReadVariableOp5batch_normalization_1/beta/Adam_1/Read/ReadVariableOp/batch_normalization_1/gamma/Read/ReadVariableOp4batch_normalization_1/gamma/Adam/Read/ReadVariableOp6batch_normalization_1/gamma/Adam_1/Read/ReadVariableOp5batch_normalization_1/moving_mean/Read/ReadVariableOp9batch_normalization_1/moving_variance/Read/ReadVariableOp.batch_normalization_2/beta/Read/ReadVariableOp3batch_normalization_2/beta/Adam/Read/ReadVariableOp5batch_normalization_2/beta/Adam_1/Read/ReadVariableOp/batch_normalization_2/gamma/Read/ReadVariableOp4batch_normalization_2/gamma/Adam/Read/ReadVariableOp6batch_normalization_2/gamma/Adam_1/Read/ReadVariableOp5batch_normalization_2/moving_mean/Read/ReadVariableOp9batch_normalization_2/moving_variance/Read/ReadVariableOpbeta1_power/Read/ReadVariableOpbeta2_power/Read/ReadVariableOpconv1/b/Read/ReadVariableOp conv1/b/Adam/Read/ReadVariableOp"conv1/b/Adam_1/Read/ReadVariableOpconv1/w/Read/ReadVariableOp conv1/w/Adam/Read/ReadVariableOp"conv1/w/Adam_1/Read/ReadVariableOpconv_d1/b/Read/ReadVariableOp"conv_d1/b/Adam/Read/ReadVariableOp$conv_d1/b/Adam_1/Read/ReadVariableOpconv_d1/w/Read/ReadVariableOp"conv_d1/w/Adam/Read/ReadVariableOp$conv_d1/w/Adam_1/Read/ReadVariableOpconv_d2/b/Read/ReadVariableOp"conv_d2/b/Adam/Read/ReadVariableOp$conv_d2/b/Adam_1/Read/ReadVariableOpconv_d2/w/Read/ReadVariableOp"conv_d2/w/Adam/Read/ReadVariableOp$conv_d2/w/Adam_1/Read/ReadVariableOpconv_out/b/Read/ReadVariableOp#conv_out/b/Adam/Read/ReadVariableOp%conv_out/b/Adam_1/Read/ReadVariableOpconv_out/w/Read/ReadVariableOp#conv_out/w/Adam/Read/ReadVariableOp%conv_out/w/Adam_1/Read/ReadVariableOp1sbb1/batch_normalization/beta/Read/ReadVariableOp6sbb1/batch_normalization/beta/Adam/Read/ReadVariableOp8sbb1/batch_normalization/beta/Adam_1/Read/ReadVariableOp2sbb1/batch_normalization/gamma/Read/ReadVariableOp7sbb1/batch_normalization/gamma/Adam/Read/ReadVariableOp9sbb1/batch_normalization/gamma/Adam_1/Read/ReadVariableOp8sbb1/batch_normalization/moving_mean/Read/ReadVariableOp<sbb1/batch_normalization/moving_variance/Read/ReadVariableOp sbb1/conv1/b/Read/ReadVariableOp%sbb1/conv1/b/Adam/Read/ReadVariableOp'sbb1/conv1/b/Adam_1/Read/ReadVariableOp sbb1/conv1/w/Read/ReadVariableOp%sbb1/conv1/w/Adam/Read/ReadVariableOp'sbb1/conv1/w/Adam_1/Read/ReadVariableOp sbb1/conv2/b/Read/ReadVariableOp%sbb1/conv2/b/Adam/Read/ReadVariableOp'sbb1/conv2/b/Adam_1/Read/ReadVariableOp sbb1/conv2/w/Read/ReadVariableOp%sbb1/conv2/w/Adam/Read/ReadVariableOp'sbb1/conv2/w/Adam_1/Read/ReadVariableOp sbb1/conv3/b/Read/ReadVariableOp%sbb1/conv3/b/Adam/Read/ReadVariableOp'sbb1/conv3/b/Adam_1/Read/ReadVariableOp sbb1/conv3/w/Read/ReadVariableOp%sbb1/conv3/w/Adam/Read/ReadVariableOp'sbb1/conv3/w/Adam_1/Read/ReadVariableOp sbb1/conv4/b/Read/ReadVariableOp%sbb1/conv4/b/Adam/Read/ReadVariableOp'sbb1/conv4/b/Adam_1/Read/ReadVariableOp sbb1/conv4/w/Read/ReadVariableOp%sbb1/conv4/w/Adam/Read/ReadVariableOp'sbb1/conv4/w/Adam_1/Read/ReadVariableOp1sbb2/batch_normalization/beta/Read/ReadVariableOp6sbb2/batch_normalization/beta/Adam/Read/ReadVariableOp8sbb2/batch_normalization/beta/Adam_1/Read/ReadVariableOp2sbb2/batch_normalization/gamma/Read/ReadVariableOp7sbb2/batch_normalization/gamma/Adam/Read/ReadVariableOp9sbb2/batch_normalization/gamma/Adam_1/Read/ReadVariableOp8sbb2/batch_normalization/moving_mean/Read/ReadVariableOp<sbb2/batch_normalization/moving_variance/Read/ReadVariableOp sbb2/conv1/b/Read/ReadVariableOp%sbb2/conv1/b/Adam/Read/ReadVariableOp'sbb2/conv1/b/Adam_1/Read/ReadVariableOp sbb2/conv1/w/Read/ReadVariableOp%sbb2/conv1/w/Adam/Read/ReadVariableOp'sbb2/conv1/w/Adam_1/Read/ReadVariableOp sbb2/conv2/b/Read/ReadVariableOp%sbb2/conv2/b/Adam/Read/ReadVariableOp'sbb2/conv2/b/Adam_1/Read/ReadVariableOp sbb2/conv2/w/Read/ReadVariableOp%sbb2/conv2/w/Adam/Read/ReadVariableOp'sbb2/conv2/w/Adam_1/Read/ReadVariableOp sbb2/conv3/b/Read/ReadVariableOp%sbb2/conv3/b/Adam/Read/ReadVariableOp'sbb2/conv3/b/Adam_1/Read/ReadVariableOp sbb2/conv3/w/Read/ReadVariableOp%sbb2/conv3/w/Adam/Read/ReadVariableOp'sbb2/conv3/w/Adam_1/Read/ReadVariableOp sbb2/conv4/b/Read/ReadVariableOp%sbb2/conv4/b/Adam/Read/ReadVariableOp'sbb2/conv4/b/Adam_1/Read/ReadVariableOp sbb2/conv4/w/Read/ReadVariableOp%sbb2/conv4/w/Adam/Read/ReadVariableOp'sbb2/conv4/w/Adam_1/Read/ReadVariableOp1sbb3/batch_normalization/beta/Read/ReadVariableOp6sbb3/batch_normalization/beta/Adam/Read/ReadVariableOp8sbb3/batch_normalization/beta/Adam_1/Read/ReadVariableOp2sbb3/batch_normalization/gamma/Read/ReadVariableOp7sbb3/batch_normalization/gamma/Adam/Read/ReadVariableOp9sbb3/batch_normalization/gamma/Adam_1/Read/ReadVariableOp8sbb3/batch_normalization/moving_mean/Read/ReadVariableOp<sbb3/batch_normalization/moving_variance/Read/ReadVariableOp sbb3/conv1/b/Read/ReadVariableOp%sbb3/conv1/b/Adam/Read/ReadVariableOp'sbb3/conv1/b/Adam_1/Read/ReadVariableOp sbb3/conv1/w/Read/ReadVariableOp%sbb3/conv1/w/Adam/Read/ReadVariableOp'sbb3/conv1/w/Adam_1/Read/ReadVariableOp sbb3/conv2/b/Read/ReadVariableOp%sbb3/conv2/b/Adam/Read/ReadVariableOp'sbb3/conv2/b/Adam_1/Read/ReadVariableOp sbb3/conv2/w/Read/ReadVariableOp%sbb3/conv2/w/Adam/Read/ReadVariableOp'sbb3/conv2/w/Adam_1/Read/ReadVariableOp sbb3/conv3/b/Read/ReadVariableOp%sbb3/conv3/b/Adam/Read/ReadVariableOp'sbb3/conv3/b/Adam_1/Read/ReadVariableOp sbb3/conv3/w/Read/ReadVariableOp%sbb3/conv3/w/Adam/Read/ReadVariableOp'sbb3/conv3/w/Adam_1/Read/ReadVariableOp sbb3/conv4/b/Read/ReadVariableOp%sbb3/conv4/b/Adam/Read/ReadVariableOp'sbb3/conv4/b/Adam_1/Read/ReadVariableOp sbb3/conv4/w/Read/ReadVariableOp%sbb3/conv4/w/Adam/Read/ReadVariableOp'sbb3/conv4/w/Adam_1/Read/ReadVariableOp*&
 _has_manual_control_dependencies(*¤
dtypes
2
}
save/control_dependencyIdentity
save/Const^save/SaveV2*
T0*
_class
loc:@save/Const*
_output_shapes
: 

save/RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes	
:*
dtype0*˛
value¨BĽBVariableBbatch_normalization/betaBbatch_normalization/beta/AdamBbatch_normalization/beta/Adam_1Bbatch_normalization/gammaBbatch_normalization/gamma/AdamB batch_normalization/gamma/Adam_1Bbatch_normalization/moving_meanB#batch_normalization/moving_varianceBbatch_normalization_1/betaBbatch_normalization_1/beta/AdamB!batch_normalization_1/beta/Adam_1Bbatch_normalization_1/gammaB batch_normalization_1/gamma/AdamB"batch_normalization_1/gamma/Adam_1B!batch_normalization_1/moving_meanB%batch_normalization_1/moving_varianceBbatch_normalization_2/betaBbatch_normalization_2/beta/AdamB!batch_normalization_2/beta/Adam_1Bbatch_normalization_2/gammaB batch_normalization_2/gamma/AdamB"batch_normalization_2/gamma/Adam_1B!batch_normalization_2/moving_meanB%batch_normalization_2/moving_varianceBbeta1_powerBbeta2_powerBconv1/bBconv1/b/AdamBconv1/b/Adam_1Bconv1/wBconv1/w/AdamBconv1/w/Adam_1B	conv_d1/bBconv_d1/b/AdamBconv_d1/b/Adam_1B	conv_d1/wBconv_d1/w/AdamBconv_d1/w/Adam_1B	conv_d2/bBconv_d2/b/AdamBconv_d2/b/Adam_1B	conv_d2/wBconv_d2/w/AdamBconv_d2/w/Adam_1B
conv_out/bBconv_out/b/AdamBconv_out/b/Adam_1B
conv_out/wBconv_out/w/AdamBconv_out/w/Adam_1Bsbb1/batch_normalization/betaB"sbb1/batch_normalization/beta/AdamB$sbb1/batch_normalization/beta/Adam_1Bsbb1/batch_normalization/gammaB#sbb1/batch_normalization/gamma/AdamB%sbb1/batch_normalization/gamma/Adam_1B$sbb1/batch_normalization/moving_meanB(sbb1/batch_normalization/moving_varianceBsbb1/conv1/bBsbb1/conv1/b/AdamBsbb1/conv1/b/Adam_1Bsbb1/conv1/wBsbb1/conv1/w/AdamBsbb1/conv1/w/Adam_1Bsbb1/conv2/bBsbb1/conv2/b/AdamBsbb1/conv2/b/Adam_1Bsbb1/conv2/wBsbb1/conv2/w/AdamBsbb1/conv2/w/Adam_1Bsbb1/conv3/bBsbb1/conv3/b/AdamBsbb1/conv3/b/Adam_1Bsbb1/conv3/wBsbb1/conv3/w/AdamBsbb1/conv3/w/Adam_1Bsbb1/conv4/bBsbb1/conv4/b/AdamBsbb1/conv4/b/Adam_1Bsbb1/conv4/wBsbb1/conv4/w/AdamBsbb1/conv4/w/Adam_1Bsbb2/batch_normalization/betaB"sbb2/batch_normalization/beta/AdamB$sbb2/batch_normalization/beta/Adam_1Bsbb2/batch_normalization/gammaB#sbb2/batch_normalization/gamma/AdamB%sbb2/batch_normalization/gamma/Adam_1B$sbb2/batch_normalization/moving_meanB(sbb2/batch_normalization/moving_varianceBsbb2/conv1/bBsbb2/conv1/b/AdamBsbb2/conv1/b/Adam_1Bsbb2/conv1/wBsbb2/conv1/w/AdamBsbb2/conv1/w/Adam_1Bsbb2/conv2/bBsbb2/conv2/b/AdamBsbb2/conv2/b/Adam_1Bsbb2/conv2/wBsbb2/conv2/w/AdamBsbb2/conv2/w/Adam_1Bsbb2/conv3/bBsbb2/conv3/b/AdamBsbb2/conv3/b/Adam_1Bsbb2/conv3/wBsbb2/conv3/w/AdamBsbb2/conv3/w/Adam_1Bsbb2/conv4/bBsbb2/conv4/b/AdamBsbb2/conv4/b/Adam_1Bsbb2/conv4/wBsbb2/conv4/w/AdamBsbb2/conv4/w/Adam_1Bsbb3/batch_normalization/betaB"sbb3/batch_normalization/beta/AdamB$sbb3/batch_normalization/beta/Adam_1Bsbb3/batch_normalization/gammaB#sbb3/batch_normalization/gamma/AdamB%sbb3/batch_normalization/gamma/Adam_1B$sbb3/batch_normalization/moving_meanB(sbb3/batch_normalization/moving_varianceBsbb3/conv1/bBsbb3/conv1/b/AdamBsbb3/conv1/b/Adam_1Bsbb3/conv1/wBsbb3/conv1/w/AdamBsbb3/conv1/w/Adam_1Bsbb3/conv2/bBsbb3/conv2/b/AdamBsbb3/conv2/b/Adam_1Bsbb3/conv2/wBsbb3/conv2/w/AdamBsbb3/conv2/w/Adam_1Bsbb3/conv3/bBsbb3/conv3/b/AdamBsbb3/conv3/b/Adam_1Bsbb3/conv3/wBsbb3/conv3/w/AdamBsbb3/conv3/w/Adam_1Bsbb3/conv4/bBsbb3/conv4/b/AdamBsbb3/conv4/b/Adam_1Bsbb3/conv4/wBsbb3/conv4/w/AdamBsbb3/conv4/w/Adam_1
 
save/RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes	
:*
dtype0*ź
value˛BŻB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 

save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices"/device:CPU:0*â
_output_shapesĎ
Ě:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*¤
dtypes
2
L
save/IdentityIdentitysave/RestoreV2*
T0*
_output_shapes
:
w
save/AssignVariableOpAssignVariableOpVariablesave/Identity*&
 _has_manual_control_dependencies(*
dtype0
P
save/Identity_1Identitysave/RestoreV2:1*
T0*
_output_shapes
:

save/AssignVariableOp_1AssignVariableOpbatch_normalization/betasave/Identity_1*&
 _has_manual_control_dependencies(*
dtype0
P
save/Identity_2Identitysave/RestoreV2:2*
T0*
_output_shapes
:

save/AssignVariableOp_2AssignVariableOpbatch_normalization/beta/Adamsave/Identity_2*&
 _has_manual_control_dependencies(*
dtype0
P
save/Identity_3Identitysave/RestoreV2:3*
T0*
_output_shapes
:

save/AssignVariableOp_3AssignVariableOpbatch_normalization/beta/Adam_1save/Identity_3*&
 _has_manual_control_dependencies(*
dtype0
P
save/Identity_4Identitysave/RestoreV2:4*
T0*
_output_shapes
:

save/AssignVariableOp_4AssignVariableOpbatch_normalization/gammasave/Identity_4*&
 _has_manual_control_dependencies(*
dtype0
P
save/Identity_5Identitysave/RestoreV2:5*
T0*
_output_shapes
:

save/AssignVariableOp_5AssignVariableOpbatch_normalization/gamma/Adamsave/Identity_5*&
 _has_manual_control_dependencies(*
dtype0
P
save/Identity_6Identitysave/RestoreV2:6*
T0*
_output_shapes
:

save/AssignVariableOp_6AssignVariableOp batch_normalization/gamma/Adam_1save/Identity_6*&
 _has_manual_control_dependencies(*
dtype0
P
save/Identity_7Identitysave/RestoreV2:7*
T0*
_output_shapes
:

save/AssignVariableOp_7AssignVariableOpbatch_normalization/moving_meansave/Identity_7*&
 _has_manual_control_dependencies(*
dtype0
P
save/Identity_8Identitysave/RestoreV2:8*
T0*
_output_shapes
:

save/AssignVariableOp_8AssignVariableOp#batch_normalization/moving_variancesave/Identity_8*&
 _has_manual_control_dependencies(*
dtype0
P
save/Identity_9Identitysave/RestoreV2:9*
T0*
_output_shapes
:

save/AssignVariableOp_9AssignVariableOpbatch_normalization_1/betasave/Identity_9*&
 _has_manual_control_dependencies(*
dtype0
R
save/Identity_10Identitysave/RestoreV2:10*
T0*
_output_shapes
:

save/AssignVariableOp_10AssignVariableOpbatch_normalization_1/beta/Adamsave/Identity_10*&
 _has_manual_control_dependencies(*
dtype0
R
save/Identity_11Identitysave/RestoreV2:11*
T0*
_output_shapes
:

save/AssignVariableOp_11AssignVariableOp!batch_normalization_1/beta/Adam_1save/Identity_11*&
 _has_manual_control_dependencies(*
dtype0
R
save/Identity_12Identitysave/RestoreV2:12*
T0*
_output_shapes
:

save/AssignVariableOp_12AssignVariableOpbatch_normalization_1/gammasave/Identity_12*&
 _has_manual_control_dependencies(*
dtype0
R
save/Identity_13Identitysave/RestoreV2:13*
T0*
_output_shapes
:

save/AssignVariableOp_13AssignVariableOp batch_normalization_1/gamma/Adamsave/Identity_13*&
 _has_manual_control_dependencies(*
dtype0
R
save/Identity_14Identitysave/RestoreV2:14*
T0*
_output_shapes
:

save/AssignVariableOp_14AssignVariableOp"batch_normalization_1/gamma/Adam_1save/Identity_14*&
 _has_manual_control_dependencies(*
dtype0
R
save/Identity_15Identitysave/RestoreV2:15*
T0*
_output_shapes
:

save/AssignVariableOp_15AssignVariableOp!batch_normalization_1/moving_meansave/Identity_15*&
 _has_manual_control_dependencies(*
dtype0
R
save/Identity_16Identitysave/RestoreV2:16*
T0*
_output_shapes
:

save/AssignVariableOp_16AssignVariableOp%batch_normalization_1/moving_variancesave/Identity_16*&
 _has_manual_control_dependencies(*
dtype0
R
save/Identity_17Identitysave/RestoreV2:17*
T0*
_output_shapes
:

save/AssignVariableOp_17AssignVariableOpbatch_normalization_2/betasave/Identity_17*&
 _has_manual_control_dependencies(*
dtype0
R
save/Identity_18Identitysave/RestoreV2:18*
T0*
_output_shapes
:

save/AssignVariableOp_18AssignVariableOpbatch_normalization_2/beta/Adamsave/Identity_18*&
 _has_manual_control_dependencies(*
dtype0
R
save/Identity_19Identitysave/RestoreV2:19*
T0*
_output_shapes
:

save/AssignVariableOp_19AssignVariableOp!batch_normalization_2/beta/Adam_1save/Identity_19*&
 _has_manual_control_dependencies(*
dtype0
R
save/Identity_20Identitysave/RestoreV2:20*
T0*
_output_shapes
:

save/AssignVariableOp_20AssignVariableOpbatch_normalization_2/gammasave/Identity_20*&
 _has_manual_control_dependencies(*
dtype0
R
save/Identity_21Identitysave/RestoreV2:21*
T0*
_output_shapes
:

save/AssignVariableOp_21AssignVariableOp batch_normalization_2/gamma/Adamsave/Identity_21*&
 _has_manual_control_dependencies(*
dtype0
R
save/Identity_22Identitysave/RestoreV2:22*
T0*
_output_shapes
:

save/AssignVariableOp_22AssignVariableOp"batch_normalization_2/gamma/Adam_1save/Identity_22*&
 _has_manual_control_dependencies(*
dtype0
R
save/Identity_23Identitysave/RestoreV2:23*
T0*
_output_shapes
:

save/AssignVariableOp_23AssignVariableOp!batch_normalization_2/moving_meansave/Identity_23*&
 _has_manual_control_dependencies(*
dtype0
R
save/Identity_24Identitysave/RestoreV2:24*
T0*
_output_shapes
:

save/AssignVariableOp_24AssignVariableOp%batch_normalization_2/moving_variancesave/Identity_24*&
 _has_manual_control_dependencies(*
dtype0
R
save/Identity_25Identitysave/RestoreV2:25*
T0*
_output_shapes
:

save/AssignVariableOp_25AssignVariableOpbeta1_powersave/Identity_25*&
 _has_manual_control_dependencies(*
dtype0
R
save/Identity_26Identitysave/RestoreV2:26*
T0*
_output_shapes
:

save/AssignVariableOp_26AssignVariableOpbeta2_powersave/Identity_26*&
 _has_manual_control_dependencies(*
dtype0
R
save/Identity_27Identitysave/RestoreV2:27*
T0*
_output_shapes
:
|
save/AssignVariableOp_27AssignVariableOpconv1/bsave/Identity_27*&
 _has_manual_control_dependencies(*
dtype0
R
save/Identity_28Identitysave/RestoreV2:28*
T0*
_output_shapes
:

save/AssignVariableOp_28AssignVariableOpconv1/b/Adamsave/Identity_28*&
 _has_manual_control_dependencies(*
dtype0
R
save/Identity_29Identitysave/RestoreV2:29*
T0*
_output_shapes
:

save/AssignVariableOp_29AssignVariableOpconv1/b/Adam_1save/Identity_29*&
 _has_manual_control_dependencies(*
dtype0
R
save/Identity_30Identitysave/RestoreV2:30*
T0*
_output_shapes
:
|
save/AssignVariableOp_30AssignVariableOpconv1/wsave/Identity_30*&
 _has_manual_control_dependencies(*
dtype0
R
save/Identity_31Identitysave/RestoreV2:31*
T0*
_output_shapes
:

save/AssignVariableOp_31AssignVariableOpconv1/w/Adamsave/Identity_31*&
 _has_manual_control_dependencies(*
dtype0
R
save/Identity_32Identitysave/RestoreV2:32*
T0*
_output_shapes
:

save/AssignVariableOp_32AssignVariableOpconv1/w/Adam_1save/Identity_32*&
 _has_manual_control_dependencies(*
dtype0
R
save/Identity_33Identitysave/RestoreV2:33*
T0*
_output_shapes
:
~
save/AssignVariableOp_33AssignVariableOp	conv_d1/bsave/Identity_33*&
 _has_manual_control_dependencies(*
dtype0
R
save/Identity_34Identitysave/RestoreV2:34*
T0*
_output_shapes
:

save/AssignVariableOp_34AssignVariableOpconv_d1/b/Adamsave/Identity_34*&
 _has_manual_control_dependencies(*
dtype0
R
save/Identity_35Identitysave/RestoreV2:35*
T0*
_output_shapes
:

save/AssignVariableOp_35AssignVariableOpconv_d1/b/Adam_1save/Identity_35*&
 _has_manual_control_dependencies(*
dtype0
R
save/Identity_36Identitysave/RestoreV2:36*
T0*
_output_shapes
:
~
save/AssignVariableOp_36AssignVariableOp	conv_d1/wsave/Identity_36*&
 _has_manual_control_dependencies(*
dtype0
R
save/Identity_37Identitysave/RestoreV2:37*
T0*
_output_shapes
:

save/AssignVariableOp_37AssignVariableOpconv_d1/w/Adamsave/Identity_37*&
 _has_manual_control_dependencies(*
dtype0
R
save/Identity_38Identitysave/RestoreV2:38*
T0*
_output_shapes
:

save/AssignVariableOp_38AssignVariableOpconv_d1/w/Adam_1save/Identity_38*&
 _has_manual_control_dependencies(*
dtype0
R
save/Identity_39Identitysave/RestoreV2:39*
T0*
_output_shapes
:
~
save/AssignVariableOp_39AssignVariableOp	conv_d2/bsave/Identity_39*&
 _has_manual_control_dependencies(*
dtype0
R
save/Identity_40Identitysave/RestoreV2:40*
T0*
_output_shapes
:

save/AssignVariableOp_40AssignVariableOpconv_d2/b/Adamsave/Identity_40*&
 _has_manual_control_dependencies(*
dtype0
R
save/Identity_41Identitysave/RestoreV2:41*
T0*
_output_shapes
:

save/AssignVariableOp_41AssignVariableOpconv_d2/b/Adam_1save/Identity_41*&
 _has_manual_control_dependencies(*
dtype0
R
save/Identity_42Identitysave/RestoreV2:42*
T0*
_output_shapes
:
~
save/AssignVariableOp_42AssignVariableOp	conv_d2/wsave/Identity_42*&
 _has_manual_control_dependencies(*
dtype0
R
save/Identity_43Identitysave/RestoreV2:43*
T0*
_output_shapes
:

save/AssignVariableOp_43AssignVariableOpconv_d2/w/Adamsave/Identity_43*&
 _has_manual_control_dependencies(*
dtype0
R
save/Identity_44Identitysave/RestoreV2:44*
T0*
_output_shapes
:

save/AssignVariableOp_44AssignVariableOpconv_d2/w/Adam_1save/Identity_44*&
 _has_manual_control_dependencies(*
dtype0
R
save/Identity_45Identitysave/RestoreV2:45*
T0*
_output_shapes
:

save/AssignVariableOp_45AssignVariableOp
conv_out/bsave/Identity_45*&
 _has_manual_control_dependencies(*
dtype0
R
save/Identity_46Identitysave/RestoreV2:46*
T0*
_output_shapes
:

save/AssignVariableOp_46AssignVariableOpconv_out/b/Adamsave/Identity_46*&
 _has_manual_control_dependencies(*
dtype0
R
save/Identity_47Identitysave/RestoreV2:47*
T0*
_output_shapes
:

save/AssignVariableOp_47AssignVariableOpconv_out/b/Adam_1save/Identity_47*&
 _has_manual_control_dependencies(*
dtype0
R
save/Identity_48Identitysave/RestoreV2:48*
T0*
_output_shapes
:

save/AssignVariableOp_48AssignVariableOp
conv_out/wsave/Identity_48*&
 _has_manual_control_dependencies(*
dtype0
R
save/Identity_49Identitysave/RestoreV2:49*
T0*
_output_shapes
:

save/AssignVariableOp_49AssignVariableOpconv_out/w/Adamsave/Identity_49*&
 _has_manual_control_dependencies(*
dtype0
R
save/Identity_50Identitysave/RestoreV2:50*
T0*
_output_shapes
:

save/AssignVariableOp_50AssignVariableOpconv_out/w/Adam_1save/Identity_50*&
 _has_manual_control_dependencies(*
dtype0
R
save/Identity_51Identitysave/RestoreV2:51*
T0*
_output_shapes
:

save/AssignVariableOp_51AssignVariableOpsbb1/batch_normalization/betasave/Identity_51*&
 _has_manual_control_dependencies(*
dtype0
R
save/Identity_52Identitysave/RestoreV2:52*
T0*
_output_shapes
:

save/AssignVariableOp_52AssignVariableOp"sbb1/batch_normalization/beta/Adamsave/Identity_52*&
 _has_manual_control_dependencies(*
dtype0
R
save/Identity_53Identitysave/RestoreV2:53*
T0*
_output_shapes
:

save/AssignVariableOp_53AssignVariableOp$sbb1/batch_normalization/beta/Adam_1save/Identity_53*&
 _has_manual_control_dependencies(*
dtype0
R
save/Identity_54Identitysave/RestoreV2:54*
T0*
_output_shapes
:

save/AssignVariableOp_54AssignVariableOpsbb1/batch_normalization/gammasave/Identity_54*&
 _has_manual_control_dependencies(*
dtype0
R
save/Identity_55Identitysave/RestoreV2:55*
T0*
_output_shapes
:

save/AssignVariableOp_55AssignVariableOp#sbb1/batch_normalization/gamma/Adamsave/Identity_55*&
 _has_manual_control_dependencies(*
dtype0
R
save/Identity_56Identitysave/RestoreV2:56*
T0*
_output_shapes
:

save/AssignVariableOp_56AssignVariableOp%sbb1/batch_normalization/gamma/Adam_1save/Identity_56*&
 _has_manual_control_dependencies(*
dtype0
R
save/Identity_57Identitysave/RestoreV2:57*
T0*
_output_shapes
:

save/AssignVariableOp_57AssignVariableOp$sbb1/batch_normalization/moving_meansave/Identity_57*&
 _has_manual_control_dependencies(*
dtype0
R
save/Identity_58Identitysave/RestoreV2:58*
T0*
_output_shapes
:

save/AssignVariableOp_58AssignVariableOp(sbb1/batch_normalization/moving_variancesave/Identity_58*&
 _has_manual_control_dependencies(*
dtype0
R
save/Identity_59Identitysave/RestoreV2:59*
T0*
_output_shapes
:

save/AssignVariableOp_59AssignVariableOpsbb1/conv1/bsave/Identity_59*&
 _has_manual_control_dependencies(*
dtype0
R
save/Identity_60Identitysave/RestoreV2:60*
T0*
_output_shapes
:

save/AssignVariableOp_60AssignVariableOpsbb1/conv1/b/Adamsave/Identity_60*&
 _has_manual_control_dependencies(*
dtype0
R
save/Identity_61Identitysave/RestoreV2:61*
T0*
_output_shapes
:

save/AssignVariableOp_61AssignVariableOpsbb1/conv1/b/Adam_1save/Identity_61*&
 _has_manual_control_dependencies(*
dtype0
R
save/Identity_62Identitysave/RestoreV2:62*
T0*
_output_shapes
:

save/AssignVariableOp_62AssignVariableOpsbb1/conv1/wsave/Identity_62*&
 _has_manual_control_dependencies(*
dtype0
R
save/Identity_63Identitysave/RestoreV2:63*
T0*
_output_shapes
:

save/AssignVariableOp_63AssignVariableOpsbb1/conv1/w/Adamsave/Identity_63*&
 _has_manual_control_dependencies(*
dtype0
R
save/Identity_64Identitysave/RestoreV2:64*
T0*
_output_shapes
:

save/AssignVariableOp_64AssignVariableOpsbb1/conv1/w/Adam_1save/Identity_64*&
 _has_manual_control_dependencies(*
dtype0
R
save/Identity_65Identitysave/RestoreV2:65*
T0*
_output_shapes
:

save/AssignVariableOp_65AssignVariableOpsbb1/conv2/bsave/Identity_65*&
 _has_manual_control_dependencies(*
dtype0
R
save/Identity_66Identitysave/RestoreV2:66*
T0*
_output_shapes
:

save/AssignVariableOp_66AssignVariableOpsbb1/conv2/b/Adamsave/Identity_66*&
 _has_manual_control_dependencies(*
dtype0
R
save/Identity_67Identitysave/RestoreV2:67*
T0*
_output_shapes
:

save/AssignVariableOp_67AssignVariableOpsbb1/conv2/b/Adam_1save/Identity_67*&
 _has_manual_control_dependencies(*
dtype0
R
save/Identity_68Identitysave/RestoreV2:68*
T0*
_output_shapes
:

save/AssignVariableOp_68AssignVariableOpsbb1/conv2/wsave/Identity_68*&
 _has_manual_control_dependencies(*
dtype0
R
save/Identity_69Identitysave/RestoreV2:69*
T0*
_output_shapes
:

save/AssignVariableOp_69AssignVariableOpsbb1/conv2/w/Adamsave/Identity_69*&
 _has_manual_control_dependencies(*
dtype0
R
save/Identity_70Identitysave/RestoreV2:70*
T0*
_output_shapes
:

save/AssignVariableOp_70AssignVariableOpsbb1/conv2/w/Adam_1save/Identity_70*&
 _has_manual_control_dependencies(*
dtype0
R
save/Identity_71Identitysave/RestoreV2:71*
T0*
_output_shapes
:

save/AssignVariableOp_71AssignVariableOpsbb1/conv3/bsave/Identity_71*&
 _has_manual_control_dependencies(*
dtype0
R
save/Identity_72Identitysave/RestoreV2:72*
T0*
_output_shapes
:

save/AssignVariableOp_72AssignVariableOpsbb1/conv3/b/Adamsave/Identity_72*&
 _has_manual_control_dependencies(*
dtype0
R
save/Identity_73Identitysave/RestoreV2:73*
T0*
_output_shapes
:

save/AssignVariableOp_73AssignVariableOpsbb1/conv3/b/Adam_1save/Identity_73*&
 _has_manual_control_dependencies(*
dtype0
R
save/Identity_74Identitysave/RestoreV2:74*
T0*
_output_shapes
:

save/AssignVariableOp_74AssignVariableOpsbb1/conv3/wsave/Identity_74*&
 _has_manual_control_dependencies(*
dtype0
R
save/Identity_75Identitysave/RestoreV2:75*
T0*
_output_shapes
:

save/AssignVariableOp_75AssignVariableOpsbb1/conv3/w/Adamsave/Identity_75*&
 _has_manual_control_dependencies(*
dtype0
R
save/Identity_76Identitysave/RestoreV2:76*
T0*
_output_shapes
:

save/AssignVariableOp_76AssignVariableOpsbb1/conv3/w/Adam_1save/Identity_76*&
 _has_manual_control_dependencies(*
dtype0
R
save/Identity_77Identitysave/RestoreV2:77*
T0*
_output_shapes
:

save/AssignVariableOp_77AssignVariableOpsbb1/conv4/bsave/Identity_77*&
 _has_manual_control_dependencies(*
dtype0
R
save/Identity_78Identitysave/RestoreV2:78*
T0*
_output_shapes
:

save/AssignVariableOp_78AssignVariableOpsbb1/conv4/b/Adamsave/Identity_78*&
 _has_manual_control_dependencies(*
dtype0
R
save/Identity_79Identitysave/RestoreV2:79*
T0*
_output_shapes
:

save/AssignVariableOp_79AssignVariableOpsbb1/conv4/b/Adam_1save/Identity_79*&
 _has_manual_control_dependencies(*
dtype0
R
save/Identity_80Identitysave/RestoreV2:80*
T0*
_output_shapes
:

save/AssignVariableOp_80AssignVariableOpsbb1/conv4/wsave/Identity_80*&
 _has_manual_control_dependencies(*
dtype0
R
save/Identity_81Identitysave/RestoreV2:81*
T0*
_output_shapes
:

save/AssignVariableOp_81AssignVariableOpsbb1/conv4/w/Adamsave/Identity_81*&
 _has_manual_control_dependencies(*
dtype0
R
save/Identity_82Identitysave/RestoreV2:82*
T0*
_output_shapes
:

save/AssignVariableOp_82AssignVariableOpsbb1/conv4/w/Adam_1save/Identity_82*&
 _has_manual_control_dependencies(*
dtype0
R
save/Identity_83Identitysave/RestoreV2:83*
T0*
_output_shapes
:

save/AssignVariableOp_83AssignVariableOpsbb2/batch_normalization/betasave/Identity_83*&
 _has_manual_control_dependencies(*
dtype0
R
save/Identity_84Identitysave/RestoreV2:84*
T0*
_output_shapes
:

save/AssignVariableOp_84AssignVariableOp"sbb2/batch_normalization/beta/Adamsave/Identity_84*&
 _has_manual_control_dependencies(*
dtype0
R
save/Identity_85Identitysave/RestoreV2:85*
T0*
_output_shapes
:

save/AssignVariableOp_85AssignVariableOp$sbb2/batch_normalization/beta/Adam_1save/Identity_85*&
 _has_manual_control_dependencies(*
dtype0
R
save/Identity_86Identitysave/RestoreV2:86*
T0*
_output_shapes
:

save/AssignVariableOp_86AssignVariableOpsbb2/batch_normalization/gammasave/Identity_86*&
 _has_manual_control_dependencies(*
dtype0
R
save/Identity_87Identitysave/RestoreV2:87*
T0*
_output_shapes
:

save/AssignVariableOp_87AssignVariableOp#sbb2/batch_normalization/gamma/Adamsave/Identity_87*&
 _has_manual_control_dependencies(*
dtype0
R
save/Identity_88Identitysave/RestoreV2:88*
T0*
_output_shapes
:

save/AssignVariableOp_88AssignVariableOp%sbb2/batch_normalization/gamma/Adam_1save/Identity_88*&
 _has_manual_control_dependencies(*
dtype0
R
save/Identity_89Identitysave/RestoreV2:89*
T0*
_output_shapes
:

save/AssignVariableOp_89AssignVariableOp$sbb2/batch_normalization/moving_meansave/Identity_89*&
 _has_manual_control_dependencies(*
dtype0
R
save/Identity_90Identitysave/RestoreV2:90*
T0*
_output_shapes
:

save/AssignVariableOp_90AssignVariableOp(sbb2/batch_normalization/moving_variancesave/Identity_90*&
 _has_manual_control_dependencies(*
dtype0
R
save/Identity_91Identitysave/RestoreV2:91*
T0*
_output_shapes
:

save/AssignVariableOp_91AssignVariableOpsbb2/conv1/bsave/Identity_91*&
 _has_manual_control_dependencies(*
dtype0
R
save/Identity_92Identitysave/RestoreV2:92*
T0*
_output_shapes
:

save/AssignVariableOp_92AssignVariableOpsbb2/conv1/b/Adamsave/Identity_92*&
 _has_manual_control_dependencies(*
dtype0
R
save/Identity_93Identitysave/RestoreV2:93*
T0*
_output_shapes
:

save/AssignVariableOp_93AssignVariableOpsbb2/conv1/b/Adam_1save/Identity_93*&
 _has_manual_control_dependencies(*
dtype0
R
save/Identity_94Identitysave/RestoreV2:94*
T0*
_output_shapes
:

save/AssignVariableOp_94AssignVariableOpsbb2/conv1/wsave/Identity_94*&
 _has_manual_control_dependencies(*
dtype0
R
save/Identity_95Identitysave/RestoreV2:95*
T0*
_output_shapes
:

save/AssignVariableOp_95AssignVariableOpsbb2/conv1/w/Adamsave/Identity_95*&
 _has_manual_control_dependencies(*
dtype0
R
save/Identity_96Identitysave/RestoreV2:96*
T0*
_output_shapes
:

save/AssignVariableOp_96AssignVariableOpsbb2/conv1/w/Adam_1save/Identity_96*&
 _has_manual_control_dependencies(*
dtype0
R
save/Identity_97Identitysave/RestoreV2:97*
T0*
_output_shapes
:

save/AssignVariableOp_97AssignVariableOpsbb2/conv2/bsave/Identity_97*&
 _has_manual_control_dependencies(*
dtype0
R
save/Identity_98Identitysave/RestoreV2:98*
T0*
_output_shapes
:

save/AssignVariableOp_98AssignVariableOpsbb2/conv2/b/Adamsave/Identity_98*&
 _has_manual_control_dependencies(*
dtype0
R
save/Identity_99Identitysave/RestoreV2:99*
T0*
_output_shapes
:

save/AssignVariableOp_99AssignVariableOpsbb2/conv2/b/Adam_1save/Identity_99*&
 _has_manual_control_dependencies(*
dtype0
T
save/Identity_100Identitysave/RestoreV2:100*
T0*
_output_shapes
:

save/AssignVariableOp_100AssignVariableOpsbb2/conv2/wsave/Identity_100*&
 _has_manual_control_dependencies(*
dtype0
T
save/Identity_101Identitysave/RestoreV2:101*
T0*
_output_shapes
:

save/AssignVariableOp_101AssignVariableOpsbb2/conv2/w/Adamsave/Identity_101*&
 _has_manual_control_dependencies(*
dtype0
T
save/Identity_102Identitysave/RestoreV2:102*
T0*
_output_shapes
:

save/AssignVariableOp_102AssignVariableOpsbb2/conv2/w/Adam_1save/Identity_102*&
 _has_manual_control_dependencies(*
dtype0
T
save/Identity_103Identitysave/RestoreV2:103*
T0*
_output_shapes
:

save/AssignVariableOp_103AssignVariableOpsbb2/conv3/bsave/Identity_103*&
 _has_manual_control_dependencies(*
dtype0
T
save/Identity_104Identitysave/RestoreV2:104*
T0*
_output_shapes
:

save/AssignVariableOp_104AssignVariableOpsbb2/conv3/b/Adamsave/Identity_104*&
 _has_manual_control_dependencies(*
dtype0
T
save/Identity_105Identitysave/RestoreV2:105*
T0*
_output_shapes
:

save/AssignVariableOp_105AssignVariableOpsbb2/conv3/b/Adam_1save/Identity_105*&
 _has_manual_control_dependencies(*
dtype0
T
save/Identity_106Identitysave/RestoreV2:106*
T0*
_output_shapes
:

save/AssignVariableOp_106AssignVariableOpsbb2/conv3/wsave/Identity_106*&
 _has_manual_control_dependencies(*
dtype0
T
save/Identity_107Identitysave/RestoreV2:107*
T0*
_output_shapes
:

save/AssignVariableOp_107AssignVariableOpsbb2/conv3/w/Adamsave/Identity_107*&
 _has_manual_control_dependencies(*
dtype0
T
save/Identity_108Identitysave/RestoreV2:108*
T0*
_output_shapes
:

save/AssignVariableOp_108AssignVariableOpsbb2/conv3/w/Adam_1save/Identity_108*&
 _has_manual_control_dependencies(*
dtype0
T
save/Identity_109Identitysave/RestoreV2:109*
T0*
_output_shapes
:

save/AssignVariableOp_109AssignVariableOpsbb2/conv4/bsave/Identity_109*&
 _has_manual_control_dependencies(*
dtype0
T
save/Identity_110Identitysave/RestoreV2:110*
T0*
_output_shapes
:

save/AssignVariableOp_110AssignVariableOpsbb2/conv4/b/Adamsave/Identity_110*&
 _has_manual_control_dependencies(*
dtype0
T
save/Identity_111Identitysave/RestoreV2:111*
T0*
_output_shapes
:

save/AssignVariableOp_111AssignVariableOpsbb2/conv4/b/Adam_1save/Identity_111*&
 _has_manual_control_dependencies(*
dtype0
T
save/Identity_112Identitysave/RestoreV2:112*
T0*
_output_shapes
:

save/AssignVariableOp_112AssignVariableOpsbb2/conv4/wsave/Identity_112*&
 _has_manual_control_dependencies(*
dtype0
T
save/Identity_113Identitysave/RestoreV2:113*
T0*
_output_shapes
:

save/AssignVariableOp_113AssignVariableOpsbb2/conv4/w/Adamsave/Identity_113*&
 _has_manual_control_dependencies(*
dtype0
T
save/Identity_114Identitysave/RestoreV2:114*
T0*
_output_shapes
:

save/AssignVariableOp_114AssignVariableOpsbb2/conv4/w/Adam_1save/Identity_114*&
 _has_manual_control_dependencies(*
dtype0
T
save/Identity_115Identitysave/RestoreV2:115*
T0*
_output_shapes
:

save/AssignVariableOp_115AssignVariableOpsbb3/batch_normalization/betasave/Identity_115*&
 _has_manual_control_dependencies(*
dtype0
T
save/Identity_116Identitysave/RestoreV2:116*
T0*
_output_shapes
:

save/AssignVariableOp_116AssignVariableOp"sbb3/batch_normalization/beta/Adamsave/Identity_116*&
 _has_manual_control_dependencies(*
dtype0
T
save/Identity_117Identitysave/RestoreV2:117*
T0*
_output_shapes
:

save/AssignVariableOp_117AssignVariableOp$sbb3/batch_normalization/beta/Adam_1save/Identity_117*&
 _has_manual_control_dependencies(*
dtype0
T
save/Identity_118Identitysave/RestoreV2:118*
T0*
_output_shapes
:

save/AssignVariableOp_118AssignVariableOpsbb3/batch_normalization/gammasave/Identity_118*&
 _has_manual_control_dependencies(*
dtype0
T
save/Identity_119Identitysave/RestoreV2:119*
T0*
_output_shapes
:

save/AssignVariableOp_119AssignVariableOp#sbb3/batch_normalization/gamma/Adamsave/Identity_119*&
 _has_manual_control_dependencies(*
dtype0
T
save/Identity_120Identitysave/RestoreV2:120*
T0*
_output_shapes
:

save/AssignVariableOp_120AssignVariableOp%sbb3/batch_normalization/gamma/Adam_1save/Identity_120*&
 _has_manual_control_dependencies(*
dtype0
T
save/Identity_121Identitysave/RestoreV2:121*
T0*
_output_shapes
:

save/AssignVariableOp_121AssignVariableOp$sbb3/batch_normalization/moving_meansave/Identity_121*&
 _has_manual_control_dependencies(*
dtype0
T
save/Identity_122Identitysave/RestoreV2:122*
T0*
_output_shapes
:

save/AssignVariableOp_122AssignVariableOp(sbb3/batch_normalization/moving_variancesave/Identity_122*&
 _has_manual_control_dependencies(*
dtype0
T
save/Identity_123Identitysave/RestoreV2:123*
T0*
_output_shapes
:

save/AssignVariableOp_123AssignVariableOpsbb3/conv1/bsave/Identity_123*&
 _has_manual_control_dependencies(*
dtype0
T
save/Identity_124Identitysave/RestoreV2:124*
T0*
_output_shapes
:

save/AssignVariableOp_124AssignVariableOpsbb3/conv1/b/Adamsave/Identity_124*&
 _has_manual_control_dependencies(*
dtype0
T
save/Identity_125Identitysave/RestoreV2:125*
T0*
_output_shapes
:

save/AssignVariableOp_125AssignVariableOpsbb3/conv1/b/Adam_1save/Identity_125*&
 _has_manual_control_dependencies(*
dtype0
T
save/Identity_126Identitysave/RestoreV2:126*
T0*
_output_shapes
:

save/AssignVariableOp_126AssignVariableOpsbb3/conv1/wsave/Identity_126*&
 _has_manual_control_dependencies(*
dtype0
T
save/Identity_127Identitysave/RestoreV2:127*
T0*
_output_shapes
:

save/AssignVariableOp_127AssignVariableOpsbb3/conv1/w/Adamsave/Identity_127*&
 _has_manual_control_dependencies(*
dtype0
T
save/Identity_128Identitysave/RestoreV2:128*
T0*
_output_shapes
:

save/AssignVariableOp_128AssignVariableOpsbb3/conv1/w/Adam_1save/Identity_128*&
 _has_manual_control_dependencies(*
dtype0
T
save/Identity_129Identitysave/RestoreV2:129*
T0*
_output_shapes
:

save/AssignVariableOp_129AssignVariableOpsbb3/conv2/bsave/Identity_129*&
 _has_manual_control_dependencies(*
dtype0
T
save/Identity_130Identitysave/RestoreV2:130*
T0*
_output_shapes
:

save/AssignVariableOp_130AssignVariableOpsbb3/conv2/b/Adamsave/Identity_130*&
 _has_manual_control_dependencies(*
dtype0
T
save/Identity_131Identitysave/RestoreV2:131*
T0*
_output_shapes
:

save/AssignVariableOp_131AssignVariableOpsbb3/conv2/b/Adam_1save/Identity_131*&
 _has_manual_control_dependencies(*
dtype0
T
save/Identity_132Identitysave/RestoreV2:132*
T0*
_output_shapes
:

save/AssignVariableOp_132AssignVariableOpsbb3/conv2/wsave/Identity_132*&
 _has_manual_control_dependencies(*
dtype0
T
save/Identity_133Identitysave/RestoreV2:133*
T0*
_output_shapes
:

save/AssignVariableOp_133AssignVariableOpsbb3/conv2/w/Adamsave/Identity_133*&
 _has_manual_control_dependencies(*
dtype0
T
save/Identity_134Identitysave/RestoreV2:134*
T0*
_output_shapes
:

save/AssignVariableOp_134AssignVariableOpsbb3/conv2/w/Adam_1save/Identity_134*&
 _has_manual_control_dependencies(*
dtype0
T
save/Identity_135Identitysave/RestoreV2:135*
T0*
_output_shapes
:

save/AssignVariableOp_135AssignVariableOpsbb3/conv3/bsave/Identity_135*&
 _has_manual_control_dependencies(*
dtype0
T
save/Identity_136Identitysave/RestoreV2:136*
T0*
_output_shapes
:

save/AssignVariableOp_136AssignVariableOpsbb3/conv3/b/Adamsave/Identity_136*&
 _has_manual_control_dependencies(*
dtype0
T
save/Identity_137Identitysave/RestoreV2:137*
T0*
_output_shapes
:

save/AssignVariableOp_137AssignVariableOpsbb3/conv3/b/Adam_1save/Identity_137*&
 _has_manual_control_dependencies(*
dtype0
T
save/Identity_138Identitysave/RestoreV2:138*
T0*
_output_shapes
:

save/AssignVariableOp_138AssignVariableOpsbb3/conv3/wsave/Identity_138*&
 _has_manual_control_dependencies(*
dtype0
T
save/Identity_139Identitysave/RestoreV2:139*
T0*
_output_shapes
:

save/AssignVariableOp_139AssignVariableOpsbb3/conv3/w/Adamsave/Identity_139*&
 _has_manual_control_dependencies(*
dtype0
T
save/Identity_140Identitysave/RestoreV2:140*
T0*
_output_shapes
:

save/AssignVariableOp_140AssignVariableOpsbb3/conv3/w/Adam_1save/Identity_140*&
 _has_manual_control_dependencies(*
dtype0
T
save/Identity_141Identitysave/RestoreV2:141*
T0*
_output_shapes
:

save/AssignVariableOp_141AssignVariableOpsbb3/conv4/bsave/Identity_141*&
 _has_manual_control_dependencies(*
dtype0
T
save/Identity_142Identitysave/RestoreV2:142*
T0*
_output_shapes
:

save/AssignVariableOp_142AssignVariableOpsbb3/conv4/b/Adamsave/Identity_142*&
 _has_manual_control_dependencies(*
dtype0
T
save/Identity_143Identitysave/RestoreV2:143*
T0*
_output_shapes
:

save/AssignVariableOp_143AssignVariableOpsbb3/conv4/b/Adam_1save/Identity_143*&
 _has_manual_control_dependencies(*
dtype0
T
save/Identity_144Identitysave/RestoreV2:144*
T0*
_output_shapes
:

save/AssignVariableOp_144AssignVariableOpsbb3/conv4/wsave/Identity_144*&
 _has_manual_control_dependencies(*
dtype0
T
save/Identity_145Identitysave/RestoreV2:145*
T0*
_output_shapes
:

save/AssignVariableOp_145AssignVariableOpsbb3/conv4/w/Adamsave/Identity_145*&
 _has_manual_control_dependencies(*
dtype0
T
save/Identity_146Identitysave/RestoreV2:146*
T0*
_output_shapes
:

save/AssignVariableOp_146AssignVariableOpsbb3/conv4/w/Adam_1save/Identity_146*&
 _has_manual_control_dependencies(*
dtype0
ź
save/restore_allNoOp^save/AssignVariableOp^save/AssignVariableOp_1^save/AssignVariableOp_10^save/AssignVariableOp_100^save/AssignVariableOp_101^save/AssignVariableOp_102^save/AssignVariableOp_103^save/AssignVariableOp_104^save/AssignVariableOp_105^save/AssignVariableOp_106^save/AssignVariableOp_107^save/AssignVariableOp_108^save/AssignVariableOp_109^save/AssignVariableOp_11^save/AssignVariableOp_110^save/AssignVariableOp_111^save/AssignVariableOp_112^save/AssignVariableOp_113^save/AssignVariableOp_114^save/AssignVariableOp_115^save/AssignVariableOp_116^save/AssignVariableOp_117^save/AssignVariableOp_118^save/AssignVariableOp_119^save/AssignVariableOp_12^save/AssignVariableOp_120^save/AssignVariableOp_121^save/AssignVariableOp_122^save/AssignVariableOp_123^save/AssignVariableOp_124^save/AssignVariableOp_125^save/AssignVariableOp_126^save/AssignVariableOp_127^save/AssignVariableOp_128^save/AssignVariableOp_129^save/AssignVariableOp_13^save/AssignVariableOp_130^save/AssignVariableOp_131^save/AssignVariableOp_132^save/AssignVariableOp_133^save/AssignVariableOp_134^save/AssignVariableOp_135^save/AssignVariableOp_136^save/AssignVariableOp_137^save/AssignVariableOp_138^save/AssignVariableOp_139^save/AssignVariableOp_14^save/AssignVariableOp_140^save/AssignVariableOp_141^save/AssignVariableOp_142^save/AssignVariableOp_143^save/AssignVariableOp_144^save/AssignVariableOp_145^save/AssignVariableOp_146^save/AssignVariableOp_15^save/AssignVariableOp_16^save/AssignVariableOp_17^save/AssignVariableOp_18^save/AssignVariableOp_19^save/AssignVariableOp_2^save/AssignVariableOp_20^save/AssignVariableOp_21^save/AssignVariableOp_22^save/AssignVariableOp_23^save/AssignVariableOp_24^save/AssignVariableOp_25^save/AssignVariableOp_26^save/AssignVariableOp_27^save/AssignVariableOp_28^save/AssignVariableOp_29^save/AssignVariableOp_3^save/AssignVariableOp_30^save/AssignVariableOp_31^save/AssignVariableOp_32^save/AssignVariableOp_33^save/AssignVariableOp_34^save/AssignVariableOp_35^save/AssignVariableOp_36^save/AssignVariableOp_37^save/AssignVariableOp_38^save/AssignVariableOp_39^save/AssignVariableOp_4^save/AssignVariableOp_40^save/AssignVariableOp_41^save/AssignVariableOp_42^save/AssignVariableOp_43^save/AssignVariableOp_44^save/AssignVariableOp_45^save/AssignVariableOp_46^save/AssignVariableOp_47^save/AssignVariableOp_48^save/AssignVariableOp_49^save/AssignVariableOp_5^save/AssignVariableOp_50^save/AssignVariableOp_51^save/AssignVariableOp_52^save/AssignVariableOp_53^save/AssignVariableOp_54^save/AssignVariableOp_55^save/AssignVariableOp_56^save/AssignVariableOp_57^save/AssignVariableOp_58^save/AssignVariableOp_59^save/AssignVariableOp_6^save/AssignVariableOp_60^save/AssignVariableOp_61^save/AssignVariableOp_62^save/AssignVariableOp_63^save/AssignVariableOp_64^save/AssignVariableOp_65^save/AssignVariableOp_66^save/AssignVariableOp_67^save/AssignVariableOp_68^save/AssignVariableOp_69^save/AssignVariableOp_7^save/AssignVariableOp_70^save/AssignVariableOp_71^save/AssignVariableOp_72^save/AssignVariableOp_73^save/AssignVariableOp_74^save/AssignVariableOp_75^save/AssignVariableOp_76^save/AssignVariableOp_77^save/AssignVariableOp_78^save/AssignVariableOp_79^save/AssignVariableOp_8^save/AssignVariableOp_80^save/AssignVariableOp_81^save/AssignVariableOp_82^save/AssignVariableOp_83^save/AssignVariableOp_84^save/AssignVariableOp_85^save/AssignVariableOp_86^save/AssignVariableOp_87^save/AssignVariableOp_88^save/AssignVariableOp_89^save/AssignVariableOp_9^save/AssignVariableOp_90^save/AssignVariableOp_91^save/AssignVariableOp_92^save/AssignVariableOp_93^save/AssignVariableOp_94^save/AssignVariableOp_95^save/AssignVariableOp_96^save/AssignVariableOp_97^save/AssignVariableOp_98^save/AssignVariableOp_99

init_all_tablesNoOp
[
save_1/filename/inputConst*
_output_shapes
: *
dtype0*
valueB Bmodel
r
save_1/filenamePlaceholderWithDefaultsave_1/filename/input*
_output_shapes
: *
dtype0*
shape: 
i
save_1/ConstPlaceholderWithDefaultsave_1/filename*
_output_shapes
: *
dtype0*
shape: 

save_1/StaticRegexFullMatchStaticRegexFullMatchsave_1/Const"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*
c
save_1/Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part
h
save_1/Const_2Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part

save_1/SelectSelectsave_1/StaticRegexFullMatchsave_1/Const_1save_1/Const_2"/device:CPU:**
T0*
_output_shapes
: 
l
save_1/StringJoin
StringJoinsave_1/Constsave_1/Select"/device:CPU:**
N*
_output_shapes
: 
S
save_1/num_shardsConst*
_output_shapes
: *
dtype0*
value	B :
m
save_1/ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 

save_1/ShardedFilenameShardedFilenamesave_1/StringJoinsave_1/ShardedFilename/shardsave_1/num_shards"/device:CPU:0*
_output_shapes
: 

save_1/SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes	
:*
dtype0*˛
value¨BĽBVariableBbatch_normalization/betaBbatch_normalization/beta/AdamBbatch_normalization/beta/Adam_1Bbatch_normalization/gammaBbatch_normalization/gamma/AdamB batch_normalization/gamma/Adam_1Bbatch_normalization/moving_meanB#batch_normalization/moving_varianceBbatch_normalization_1/betaBbatch_normalization_1/beta/AdamB!batch_normalization_1/beta/Adam_1Bbatch_normalization_1/gammaB batch_normalization_1/gamma/AdamB"batch_normalization_1/gamma/Adam_1B!batch_normalization_1/moving_meanB%batch_normalization_1/moving_varianceBbatch_normalization_2/betaBbatch_normalization_2/beta/AdamB!batch_normalization_2/beta/Adam_1Bbatch_normalization_2/gammaB batch_normalization_2/gamma/AdamB"batch_normalization_2/gamma/Adam_1B!batch_normalization_2/moving_meanB%batch_normalization_2/moving_varianceBbeta1_powerBbeta2_powerBconv1/bBconv1/b/AdamBconv1/b/Adam_1Bconv1/wBconv1/w/AdamBconv1/w/Adam_1B	conv_d1/bBconv_d1/b/AdamBconv_d1/b/Adam_1B	conv_d1/wBconv_d1/w/AdamBconv_d1/w/Adam_1B	conv_d2/bBconv_d2/b/AdamBconv_d2/b/Adam_1B	conv_d2/wBconv_d2/w/AdamBconv_d2/w/Adam_1B
conv_out/bBconv_out/b/AdamBconv_out/b/Adam_1B
conv_out/wBconv_out/w/AdamBconv_out/w/Adam_1Bsbb1/batch_normalization/betaB"sbb1/batch_normalization/beta/AdamB$sbb1/batch_normalization/beta/Adam_1Bsbb1/batch_normalization/gammaB#sbb1/batch_normalization/gamma/AdamB%sbb1/batch_normalization/gamma/Adam_1B$sbb1/batch_normalization/moving_meanB(sbb1/batch_normalization/moving_varianceBsbb1/conv1/bBsbb1/conv1/b/AdamBsbb1/conv1/b/Adam_1Bsbb1/conv1/wBsbb1/conv1/w/AdamBsbb1/conv1/w/Adam_1Bsbb1/conv2/bBsbb1/conv2/b/AdamBsbb1/conv2/b/Adam_1Bsbb1/conv2/wBsbb1/conv2/w/AdamBsbb1/conv2/w/Adam_1Bsbb1/conv3/bBsbb1/conv3/b/AdamBsbb1/conv3/b/Adam_1Bsbb1/conv3/wBsbb1/conv3/w/AdamBsbb1/conv3/w/Adam_1Bsbb1/conv4/bBsbb1/conv4/b/AdamBsbb1/conv4/b/Adam_1Bsbb1/conv4/wBsbb1/conv4/w/AdamBsbb1/conv4/w/Adam_1Bsbb2/batch_normalization/betaB"sbb2/batch_normalization/beta/AdamB$sbb2/batch_normalization/beta/Adam_1Bsbb2/batch_normalization/gammaB#sbb2/batch_normalization/gamma/AdamB%sbb2/batch_normalization/gamma/Adam_1B$sbb2/batch_normalization/moving_meanB(sbb2/batch_normalization/moving_varianceBsbb2/conv1/bBsbb2/conv1/b/AdamBsbb2/conv1/b/Adam_1Bsbb2/conv1/wBsbb2/conv1/w/AdamBsbb2/conv1/w/Adam_1Bsbb2/conv2/bBsbb2/conv2/b/AdamBsbb2/conv2/b/Adam_1Bsbb2/conv2/wBsbb2/conv2/w/AdamBsbb2/conv2/w/Adam_1Bsbb2/conv3/bBsbb2/conv3/b/AdamBsbb2/conv3/b/Adam_1Bsbb2/conv3/wBsbb2/conv3/w/AdamBsbb2/conv3/w/Adam_1Bsbb2/conv4/bBsbb2/conv4/b/AdamBsbb2/conv4/b/Adam_1Bsbb2/conv4/wBsbb2/conv4/w/AdamBsbb2/conv4/w/Adam_1Bsbb3/batch_normalization/betaB"sbb3/batch_normalization/beta/AdamB$sbb3/batch_normalization/beta/Adam_1Bsbb3/batch_normalization/gammaB#sbb3/batch_normalization/gamma/AdamB%sbb3/batch_normalization/gamma/Adam_1B$sbb3/batch_normalization/moving_meanB(sbb3/batch_normalization/moving_varianceBsbb3/conv1/bBsbb3/conv1/b/AdamBsbb3/conv1/b/Adam_1Bsbb3/conv1/wBsbb3/conv1/w/AdamBsbb3/conv1/w/Adam_1Bsbb3/conv2/bBsbb3/conv2/b/AdamBsbb3/conv2/b/Adam_1Bsbb3/conv2/wBsbb3/conv2/w/AdamBsbb3/conv2/w/Adam_1Bsbb3/conv3/bBsbb3/conv3/b/AdamBsbb3/conv3/b/Adam_1Bsbb3/conv3/wBsbb3/conv3/w/AdamBsbb3/conv3/w/Adam_1Bsbb3/conv4/bBsbb3/conv4/b/AdamBsbb3/conv4/b/Adam_1Bsbb3/conv4/wBsbb3/conv4/w/AdamBsbb3/conv4/w/Adam_1

save_1/SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes	
:*
dtype0*ź
value˛BŻB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
á3
save_1/SaveV2SaveV2save_1/ShardedFilenamesave_1/SaveV2/tensor_namessave_1/SaveV2/shape_and_slicesVariable/Read/ReadVariableOp,batch_normalization/beta/Read/ReadVariableOp1batch_normalization/beta/Adam/Read/ReadVariableOp3batch_normalization/beta/Adam_1/Read/ReadVariableOp-batch_normalization/gamma/Read/ReadVariableOp2batch_normalization/gamma/Adam/Read/ReadVariableOp4batch_normalization/gamma/Adam_1/Read/ReadVariableOp3batch_normalization/moving_mean/Read/ReadVariableOp7batch_normalization/moving_variance/Read/ReadVariableOp.batch_normalization_1/beta/Read/ReadVariableOp3batch_normalization_1/beta/Adam/Read/ReadVariableOp5batch_normalization_1/beta/Adam_1/Read/ReadVariableOp/batch_normalization_1/gamma/Read/ReadVariableOp4batch_normalization_1/gamma/Adam/Read/ReadVariableOp6batch_normalization_1/gamma/Adam_1/Read/ReadVariableOp5batch_normalization_1/moving_mean/Read/ReadVariableOp9batch_normalization_1/moving_variance/Read/ReadVariableOp.batch_normalization_2/beta/Read/ReadVariableOp3batch_normalization_2/beta/Adam/Read/ReadVariableOp5batch_normalization_2/beta/Adam_1/Read/ReadVariableOp/batch_normalization_2/gamma/Read/ReadVariableOp4batch_normalization_2/gamma/Adam/Read/ReadVariableOp6batch_normalization_2/gamma/Adam_1/Read/ReadVariableOp5batch_normalization_2/moving_mean/Read/ReadVariableOp9batch_normalization_2/moving_variance/Read/ReadVariableOpbeta1_power/Read/ReadVariableOpbeta2_power/Read/ReadVariableOpconv1/b/Read/ReadVariableOp conv1/b/Adam/Read/ReadVariableOp"conv1/b/Adam_1/Read/ReadVariableOpconv1/w/Read/ReadVariableOp conv1/w/Adam/Read/ReadVariableOp"conv1/w/Adam_1/Read/ReadVariableOpconv_d1/b/Read/ReadVariableOp"conv_d1/b/Adam/Read/ReadVariableOp$conv_d1/b/Adam_1/Read/ReadVariableOpconv_d1/w/Read/ReadVariableOp"conv_d1/w/Adam/Read/ReadVariableOp$conv_d1/w/Adam_1/Read/ReadVariableOpconv_d2/b/Read/ReadVariableOp"conv_d2/b/Adam/Read/ReadVariableOp$conv_d2/b/Adam_1/Read/ReadVariableOpconv_d2/w/Read/ReadVariableOp"conv_d2/w/Adam/Read/ReadVariableOp$conv_d2/w/Adam_1/Read/ReadVariableOpconv_out/b/Read/ReadVariableOp#conv_out/b/Adam/Read/ReadVariableOp%conv_out/b/Adam_1/Read/ReadVariableOpconv_out/w/Read/ReadVariableOp#conv_out/w/Adam/Read/ReadVariableOp%conv_out/w/Adam_1/Read/ReadVariableOp1sbb1/batch_normalization/beta/Read/ReadVariableOp6sbb1/batch_normalization/beta/Adam/Read/ReadVariableOp8sbb1/batch_normalization/beta/Adam_1/Read/ReadVariableOp2sbb1/batch_normalization/gamma/Read/ReadVariableOp7sbb1/batch_normalization/gamma/Adam/Read/ReadVariableOp9sbb1/batch_normalization/gamma/Adam_1/Read/ReadVariableOp8sbb1/batch_normalization/moving_mean/Read/ReadVariableOp<sbb1/batch_normalization/moving_variance/Read/ReadVariableOp sbb1/conv1/b/Read/ReadVariableOp%sbb1/conv1/b/Adam/Read/ReadVariableOp'sbb1/conv1/b/Adam_1/Read/ReadVariableOp sbb1/conv1/w/Read/ReadVariableOp%sbb1/conv1/w/Adam/Read/ReadVariableOp'sbb1/conv1/w/Adam_1/Read/ReadVariableOp sbb1/conv2/b/Read/ReadVariableOp%sbb1/conv2/b/Adam/Read/ReadVariableOp'sbb1/conv2/b/Adam_1/Read/ReadVariableOp sbb1/conv2/w/Read/ReadVariableOp%sbb1/conv2/w/Adam/Read/ReadVariableOp'sbb1/conv2/w/Adam_1/Read/ReadVariableOp sbb1/conv3/b/Read/ReadVariableOp%sbb1/conv3/b/Adam/Read/ReadVariableOp'sbb1/conv3/b/Adam_1/Read/ReadVariableOp sbb1/conv3/w/Read/ReadVariableOp%sbb1/conv3/w/Adam/Read/ReadVariableOp'sbb1/conv3/w/Adam_1/Read/ReadVariableOp sbb1/conv4/b/Read/ReadVariableOp%sbb1/conv4/b/Adam/Read/ReadVariableOp'sbb1/conv4/b/Adam_1/Read/ReadVariableOp sbb1/conv4/w/Read/ReadVariableOp%sbb1/conv4/w/Adam/Read/ReadVariableOp'sbb1/conv4/w/Adam_1/Read/ReadVariableOp1sbb2/batch_normalization/beta/Read/ReadVariableOp6sbb2/batch_normalization/beta/Adam/Read/ReadVariableOp8sbb2/batch_normalization/beta/Adam_1/Read/ReadVariableOp2sbb2/batch_normalization/gamma/Read/ReadVariableOp7sbb2/batch_normalization/gamma/Adam/Read/ReadVariableOp9sbb2/batch_normalization/gamma/Adam_1/Read/ReadVariableOp8sbb2/batch_normalization/moving_mean/Read/ReadVariableOp<sbb2/batch_normalization/moving_variance/Read/ReadVariableOp sbb2/conv1/b/Read/ReadVariableOp%sbb2/conv1/b/Adam/Read/ReadVariableOp'sbb2/conv1/b/Adam_1/Read/ReadVariableOp sbb2/conv1/w/Read/ReadVariableOp%sbb2/conv1/w/Adam/Read/ReadVariableOp'sbb2/conv1/w/Adam_1/Read/ReadVariableOp sbb2/conv2/b/Read/ReadVariableOp%sbb2/conv2/b/Adam/Read/ReadVariableOp'sbb2/conv2/b/Adam_1/Read/ReadVariableOp sbb2/conv2/w/Read/ReadVariableOp%sbb2/conv2/w/Adam/Read/ReadVariableOp'sbb2/conv2/w/Adam_1/Read/ReadVariableOp sbb2/conv3/b/Read/ReadVariableOp%sbb2/conv3/b/Adam/Read/ReadVariableOp'sbb2/conv3/b/Adam_1/Read/ReadVariableOp sbb2/conv3/w/Read/ReadVariableOp%sbb2/conv3/w/Adam/Read/ReadVariableOp'sbb2/conv3/w/Adam_1/Read/ReadVariableOp sbb2/conv4/b/Read/ReadVariableOp%sbb2/conv4/b/Adam/Read/ReadVariableOp'sbb2/conv4/b/Adam_1/Read/ReadVariableOp sbb2/conv4/w/Read/ReadVariableOp%sbb2/conv4/w/Adam/Read/ReadVariableOp'sbb2/conv4/w/Adam_1/Read/ReadVariableOp1sbb3/batch_normalization/beta/Read/ReadVariableOp6sbb3/batch_normalization/beta/Adam/Read/ReadVariableOp8sbb3/batch_normalization/beta/Adam_1/Read/ReadVariableOp2sbb3/batch_normalization/gamma/Read/ReadVariableOp7sbb3/batch_normalization/gamma/Adam/Read/ReadVariableOp9sbb3/batch_normalization/gamma/Adam_1/Read/ReadVariableOp8sbb3/batch_normalization/moving_mean/Read/ReadVariableOp<sbb3/batch_normalization/moving_variance/Read/ReadVariableOp sbb3/conv1/b/Read/ReadVariableOp%sbb3/conv1/b/Adam/Read/ReadVariableOp'sbb3/conv1/b/Adam_1/Read/ReadVariableOp sbb3/conv1/w/Read/ReadVariableOp%sbb3/conv1/w/Adam/Read/ReadVariableOp'sbb3/conv1/w/Adam_1/Read/ReadVariableOp sbb3/conv2/b/Read/ReadVariableOp%sbb3/conv2/b/Adam/Read/ReadVariableOp'sbb3/conv2/b/Adam_1/Read/ReadVariableOp sbb3/conv2/w/Read/ReadVariableOp%sbb3/conv2/w/Adam/Read/ReadVariableOp'sbb3/conv2/w/Adam_1/Read/ReadVariableOp sbb3/conv3/b/Read/ReadVariableOp%sbb3/conv3/b/Adam/Read/ReadVariableOp'sbb3/conv3/b/Adam_1/Read/ReadVariableOp sbb3/conv3/w/Read/ReadVariableOp%sbb3/conv3/w/Adam/Read/ReadVariableOp'sbb3/conv3/w/Adam_1/Read/ReadVariableOp sbb3/conv4/b/Read/ReadVariableOp%sbb3/conv4/b/Adam/Read/ReadVariableOp'sbb3/conv4/b/Adam_1/Read/ReadVariableOp sbb3/conv4/w/Read/ReadVariableOp%sbb3/conv4/w/Adam/Read/ReadVariableOp'sbb3/conv4/w/Adam_1/Read/ReadVariableOp"/device:CPU:0*&
 _has_manual_control_dependencies(*¤
dtypes
2
Đ
save_1/control_dependencyIdentitysave_1/ShardedFilename^save_1/SaveV2"/device:CPU:0*
T0*)
_class
loc:@save_1/ShardedFilename*&
 _has_manual_control_dependencies(*
_output_shapes
: 
Ś
-save_1/MergeV2Checkpoints/checkpoint_prefixesPacksave_1/ShardedFilename^save_1/control_dependency"/device:CPU:0*
N*
T0*
_output_shapes
:
Ł
save_1/MergeV2CheckpointsMergeV2Checkpoints-save_1/MergeV2Checkpoints/checkpoint_prefixessave_1/Const"/device:CPU:0*&
 _has_manual_control_dependencies(

save_1/IdentityIdentitysave_1/Const^save_1/MergeV2Checkpoints^save_1/control_dependency"/device:CPU:0*
T0*
_output_shapes
: 

save_1/RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes	
:*
dtype0*˛
value¨BĽBVariableBbatch_normalization/betaBbatch_normalization/beta/AdamBbatch_normalization/beta/Adam_1Bbatch_normalization/gammaBbatch_normalization/gamma/AdamB batch_normalization/gamma/Adam_1Bbatch_normalization/moving_meanB#batch_normalization/moving_varianceBbatch_normalization_1/betaBbatch_normalization_1/beta/AdamB!batch_normalization_1/beta/Adam_1Bbatch_normalization_1/gammaB batch_normalization_1/gamma/AdamB"batch_normalization_1/gamma/Adam_1B!batch_normalization_1/moving_meanB%batch_normalization_1/moving_varianceBbatch_normalization_2/betaBbatch_normalization_2/beta/AdamB!batch_normalization_2/beta/Adam_1Bbatch_normalization_2/gammaB batch_normalization_2/gamma/AdamB"batch_normalization_2/gamma/Adam_1B!batch_normalization_2/moving_meanB%batch_normalization_2/moving_varianceBbeta1_powerBbeta2_powerBconv1/bBconv1/b/AdamBconv1/b/Adam_1Bconv1/wBconv1/w/AdamBconv1/w/Adam_1B	conv_d1/bBconv_d1/b/AdamBconv_d1/b/Adam_1B	conv_d1/wBconv_d1/w/AdamBconv_d1/w/Adam_1B	conv_d2/bBconv_d2/b/AdamBconv_d2/b/Adam_1B	conv_d2/wBconv_d2/w/AdamBconv_d2/w/Adam_1B
conv_out/bBconv_out/b/AdamBconv_out/b/Adam_1B
conv_out/wBconv_out/w/AdamBconv_out/w/Adam_1Bsbb1/batch_normalization/betaB"sbb1/batch_normalization/beta/AdamB$sbb1/batch_normalization/beta/Adam_1Bsbb1/batch_normalization/gammaB#sbb1/batch_normalization/gamma/AdamB%sbb1/batch_normalization/gamma/Adam_1B$sbb1/batch_normalization/moving_meanB(sbb1/batch_normalization/moving_varianceBsbb1/conv1/bBsbb1/conv1/b/AdamBsbb1/conv1/b/Adam_1Bsbb1/conv1/wBsbb1/conv1/w/AdamBsbb1/conv1/w/Adam_1Bsbb1/conv2/bBsbb1/conv2/b/AdamBsbb1/conv2/b/Adam_1Bsbb1/conv2/wBsbb1/conv2/w/AdamBsbb1/conv2/w/Adam_1Bsbb1/conv3/bBsbb1/conv3/b/AdamBsbb1/conv3/b/Adam_1Bsbb1/conv3/wBsbb1/conv3/w/AdamBsbb1/conv3/w/Adam_1Bsbb1/conv4/bBsbb1/conv4/b/AdamBsbb1/conv4/b/Adam_1Bsbb1/conv4/wBsbb1/conv4/w/AdamBsbb1/conv4/w/Adam_1Bsbb2/batch_normalization/betaB"sbb2/batch_normalization/beta/AdamB$sbb2/batch_normalization/beta/Adam_1Bsbb2/batch_normalization/gammaB#sbb2/batch_normalization/gamma/AdamB%sbb2/batch_normalization/gamma/Adam_1B$sbb2/batch_normalization/moving_meanB(sbb2/batch_normalization/moving_varianceBsbb2/conv1/bBsbb2/conv1/b/AdamBsbb2/conv1/b/Adam_1Bsbb2/conv1/wBsbb2/conv1/w/AdamBsbb2/conv1/w/Adam_1Bsbb2/conv2/bBsbb2/conv2/b/AdamBsbb2/conv2/b/Adam_1Bsbb2/conv2/wBsbb2/conv2/w/AdamBsbb2/conv2/w/Adam_1Bsbb2/conv3/bBsbb2/conv3/b/AdamBsbb2/conv3/b/Adam_1Bsbb2/conv3/wBsbb2/conv3/w/AdamBsbb2/conv3/w/Adam_1Bsbb2/conv4/bBsbb2/conv4/b/AdamBsbb2/conv4/b/Adam_1Bsbb2/conv4/wBsbb2/conv4/w/AdamBsbb2/conv4/w/Adam_1Bsbb3/batch_normalization/betaB"sbb3/batch_normalization/beta/AdamB$sbb3/batch_normalization/beta/Adam_1Bsbb3/batch_normalization/gammaB#sbb3/batch_normalization/gamma/AdamB%sbb3/batch_normalization/gamma/Adam_1B$sbb3/batch_normalization/moving_meanB(sbb3/batch_normalization/moving_varianceBsbb3/conv1/bBsbb3/conv1/b/AdamBsbb3/conv1/b/Adam_1Bsbb3/conv1/wBsbb3/conv1/w/AdamBsbb3/conv1/w/Adam_1Bsbb3/conv2/bBsbb3/conv2/b/AdamBsbb3/conv2/b/Adam_1Bsbb3/conv2/wBsbb3/conv2/w/AdamBsbb3/conv2/w/Adam_1Bsbb3/conv3/bBsbb3/conv3/b/AdamBsbb3/conv3/b/Adam_1Bsbb3/conv3/wBsbb3/conv3/w/AdamBsbb3/conv3/w/Adam_1Bsbb3/conv4/bBsbb3/conv4/b/AdamBsbb3/conv4/b/Adam_1Bsbb3/conv4/wBsbb3/conv4/w/AdamBsbb3/conv4/w/Adam_1
˘
!save_1/RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes	
:*
dtype0*ź
value˛BŻB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 

save_1/RestoreV2	RestoreV2save_1/Constsave_1/RestoreV2/tensor_names!save_1/RestoreV2/shape_and_slices"/device:CPU:0*â
_output_shapesĎ
Ě:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*¤
dtypes
2
R
save_1/Identity_1Identitysave_1/RestoreV2*
T0*
_output_shapes
:
}
save_1/AssignVariableOpAssignVariableOpVariablesave_1/Identity_1*&
 _has_manual_control_dependencies(*
dtype0
T
save_1/Identity_2Identitysave_1/RestoreV2:1*
T0*
_output_shapes
:

save_1/AssignVariableOp_1AssignVariableOpbatch_normalization/betasave_1/Identity_2*&
 _has_manual_control_dependencies(*
dtype0
T
save_1/Identity_3Identitysave_1/RestoreV2:2*
T0*
_output_shapes
:

save_1/AssignVariableOp_2AssignVariableOpbatch_normalization/beta/Adamsave_1/Identity_3*&
 _has_manual_control_dependencies(*
dtype0
T
save_1/Identity_4Identitysave_1/RestoreV2:3*
T0*
_output_shapes
:

save_1/AssignVariableOp_3AssignVariableOpbatch_normalization/beta/Adam_1save_1/Identity_4*&
 _has_manual_control_dependencies(*
dtype0
T
save_1/Identity_5Identitysave_1/RestoreV2:4*
T0*
_output_shapes
:

save_1/AssignVariableOp_4AssignVariableOpbatch_normalization/gammasave_1/Identity_5*&
 _has_manual_control_dependencies(*
dtype0
T
save_1/Identity_6Identitysave_1/RestoreV2:5*
T0*
_output_shapes
:

save_1/AssignVariableOp_5AssignVariableOpbatch_normalization/gamma/Adamsave_1/Identity_6*&
 _has_manual_control_dependencies(*
dtype0
T
save_1/Identity_7Identitysave_1/RestoreV2:6*
T0*
_output_shapes
:

save_1/AssignVariableOp_6AssignVariableOp batch_normalization/gamma/Adam_1save_1/Identity_7*&
 _has_manual_control_dependencies(*
dtype0
T
save_1/Identity_8Identitysave_1/RestoreV2:7*
T0*
_output_shapes
:

save_1/AssignVariableOp_7AssignVariableOpbatch_normalization/moving_meansave_1/Identity_8*&
 _has_manual_control_dependencies(*
dtype0
T
save_1/Identity_9Identitysave_1/RestoreV2:8*
T0*
_output_shapes
:

save_1/AssignVariableOp_8AssignVariableOp#batch_normalization/moving_variancesave_1/Identity_9*&
 _has_manual_control_dependencies(*
dtype0
U
save_1/Identity_10Identitysave_1/RestoreV2:9*
T0*
_output_shapes
:

save_1/AssignVariableOp_9AssignVariableOpbatch_normalization_1/betasave_1/Identity_10*&
 _has_manual_control_dependencies(*
dtype0
V
save_1/Identity_11Identitysave_1/RestoreV2:10*
T0*
_output_shapes
:

save_1/AssignVariableOp_10AssignVariableOpbatch_normalization_1/beta/Adamsave_1/Identity_11*&
 _has_manual_control_dependencies(*
dtype0
V
save_1/Identity_12Identitysave_1/RestoreV2:11*
T0*
_output_shapes
:

save_1/AssignVariableOp_11AssignVariableOp!batch_normalization_1/beta/Adam_1save_1/Identity_12*&
 _has_manual_control_dependencies(*
dtype0
V
save_1/Identity_13Identitysave_1/RestoreV2:12*
T0*
_output_shapes
:

save_1/AssignVariableOp_12AssignVariableOpbatch_normalization_1/gammasave_1/Identity_13*&
 _has_manual_control_dependencies(*
dtype0
V
save_1/Identity_14Identitysave_1/RestoreV2:13*
T0*
_output_shapes
:

save_1/AssignVariableOp_13AssignVariableOp batch_normalization_1/gamma/Adamsave_1/Identity_14*&
 _has_manual_control_dependencies(*
dtype0
V
save_1/Identity_15Identitysave_1/RestoreV2:14*
T0*
_output_shapes
:

save_1/AssignVariableOp_14AssignVariableOp"batch_normalization_1/gamma/Adam_1save_1/Identity_15*&
 _has_manual_control_dependencies(*
dtype0
V
save_1/Identity_16Identitysave_1/RestoreV2:15*
T0*
_output_shapes
:

save_1/AssignVariableOp_15AssignVariableOp!batch_normalization_1/moving_meansave_1/Identity_16*&
 _has_manual_control_dependencies(*
dtype0
V
save_1/Identity_17Identitysave_1/RestoreV2:16*
T0*
_output_shapes
:

save_1/AssignVariableOp_16AssignVariableOp%batch_normalization_1/moving_variancesave_1/Identity_17*&
 _has_manual_control_dependencies(*
dtype0
V
save_1/Identity_18Identitysave_1/RestoreV2:17*
T0*
_output_shapes
:

save_1/AssignVariableOp_17AssignVariableOpbatch_normalization_2/betasave_1/Identity_18*&
 _has_manual_control_dependencies(*
dtype0
V
save_1/Identity_19Identitysave_1/RestoreV2:18*
T0*
_output_shapes
:

save_1/AssignVariableOp_18AssignVariableOpbatch_normalization_2/beta/Adamsave_1/Identity_19*&
 _has_manual_control_dependencies(*
dtype0
V
save_1/Identity_20Identitysave_1/RestoreV2:19*
T0*
_output_shapes
:

save_1/AssignVariableOp_19AssignVariableOp!batch_normalization_2/beta/Adam_1save_1/Identity_20*&
 _has_manual_control_dependencies(*
dtype0
V
save_1/Identity_21Identitysave_1/RestoreV2:20*
T0*
_output_shapes
:

save_1/AssignVariableOp_20AssignVariableOpbatch_normalization_2/gammasave_1/Identity_21*&
 _has_manual_control_dependencies(*
dtype0
V
save_1/Identity_22Identitysave_1/RestoreV2:21*
T0*
_output_shapes
:

save_1/AssignVariableOp_21AssignVariableOp batch_normalization_2/gamma/Adamsave_1/Identity_22*&
 _has_manual_control_dependencies(*
dtype0
V
save_1/Identity_23Identitysave_1/RestoreV2:22*
T0*
_output_shapes
:

save_1/AssignVariableOp_22AssignVariableOp"batch_normalization_2/gamma/Adam_1save_1/Identity_23*&
 _has_manual_control_dependencies(*
dtype0
V
save_1/Identity_24Identitysave_1/RestoreV2:23*
T0*
_output_shapes
:

save_1/AssignVariableOp_23AssignVariableOp!batch_normalization_2/moving_meansave_1/Identity_24*&
 _has_manual_control_dependencies(*
dtype0
V
save_1/Identity_25Identitysave_1/RestoreV2:24*
T0*
_output_shapes
:

save_1/AssignVariableOp_24AssignVariableOp%batch_normalization_2/moving_variancesave_1/Identity_25*&
 _has_manual_control_dependencies(*
dtype0
V
save_1/Identity_26Identitysave_1/RestoreV2:25*
T0*
_output_shapes
:

save_1/AssignVariableOp_25AssignVariableOpbeta1_powersave_1/Identity_26*&
 _has_manual_control_dependencies(*
dtype0
V
save_1/Identity_27Identitysave_1/RestoreV2:26*
T0*
_output_shapes
:

save_1/AssignVariableOp_26AssignVariableOpbeta2_powersave_1/Identity_27*&
 _has_manual_control_dependencies(*
dtype0
V
save_1/Identity_28Identitysave_1/RestoreV2:27*
T0*
_output_shapes
:

save_1/AssignVariableOp_27AssignVariableOpconv1/bsave_1/Identity_28*&
 _has_manual_control_dependencies(*
dtype0
V
save_1/Identity_29Identitysave_1/RestoreV2:28*
T0*
_output_shapes
:

save_1/AssignVariableOp_28AssignVariableOpconv1/b/Adamsave_1/Identity_29*&
 _has_manual_control_dependencies(*
dtype0
V
save_1/Identity_30Identitysave_1/RestoreV2:29*
T0*
_output_shapes
:

save_1/AssignVariableOp_29AssignVariableOpconv1/b/Adam_1save_1/Identity_30*&
 _has_manual_control_dependencies(*
dtype0
V
save_1/Identity_31Identitysave_1/RestoreV2:30*
T0*
_output_shapes
:

save_1/AssignVariableOp_30AssignVariableOpconv1/wsave_1/Identity_31*&
 _has_manual_control_dependencies(*
dtype0
V
save_1/Identity_32Identitysave_1/RestoreV2:31*
T0*
_output_shapes
:

save_1/AssignVariableOp_31AssignVariableOpconv1/w/Adamsave_1/Identity_32*&
 _has_manual_control_dependencies(*
dtype0
V
save_1/Identity_33Identitysave_1/RestoreV2:32*
T0*
_output_shapes
:

save_1/AssignVariableOp_32AssignVariableOpconv1/w/Adam_1save_1/Identity_33*&
 _has_manual_control_dependencies(*
dtype0
V
save_1/Identity_34Identitysave_1/RestoreV2:33*
T0*
_output_shapes
:

save_1/AssignVariableOp_33AssignVariableOp	conv_d1/bsave_1/Identity_34*&
 _has_manual_control_dependencies(*
dtype0
V
save_1/Identity_35Identitysave_1/RestoreV2:34*
T0*
_output_shapes
:

save_1/AssignVariableOp_34AssignVariableOpconv_d1/b/Adamsave_1/Identity_35*&
 _has_manual_control_dependencies(*
dtype0
V
save_1/Identity_36Identitysave_1/RestoreV2:35*
T0*
_output_shapes
:

save_1/AssignVariableOp_35AssignVariableOpconv_d1/b/Adam_1save_1/Identity_36*&
 _has_manual_control_dependencies(*
dtype0
V
save_1/Identity_37Identitysave_1/RestoreV2:36*
T0*
_output_shapes
:

save_1/AssignVariableOp_36AssignVariableOp	conv_d1/wsave_1/Identity_37*&
 _has_manual_control_dependencies(*
dtype0
V
save_1/Identity_38Identitysave_1/RestoreV2:37*
T0*
_output_shapes
:

save_1/AssignVariableOp_37AssignVariableOpconv_d1/w/Adamsave_1/Identity_38*&
 _has_manual_control_dependencies(*
dtype0
V
save_1/Identity_39Identitysave_1/RestoreV2:38*
T0*
_output_shapes
:

save_1/AssignVariableOp_38AssignVariableOpconv_d1/w/Adam_1save_1/Identity_39*&
 _has_manual_control_dependencies(*
dtype0
V
save_1/Identity_40Identitysave_1/RestoreV2:39*
T0*
_output_shapes
:

save_1/AssignVariableOp_39AssignVariableOp	conv_d2/bsave_1/Identity_40*&
 _has_manual_control_dependencies(*
dtype0
V
save_1/Identity_41Identitysave_1/RestoreV2:40*
T0*
_output_shapes
:

save_1/AssignVariableOp_40AssignVariableOpconv_d2/b/Adamsave_1/Identity_41*&
 _has_manual_control_dependencies(*
dtype0
V
save_1/Identity_42Identitysave_1/RestoreV2:41*
T0*
_output_shapes
:

save_1/AssignVariableOp_41AssignVariableOpconv_d2/b/Adam_1save_1/Identity_42*&
 _has_manual_control_dependencies(*
dtype0
V
save_1/Identity_43Identitysave_1/RestoreV2:42*
T0*
_output_shapes
:

save_1/AssignVariableOp_42AssignVariableOp	conv_d2/wsave_1/Identity_43*&
 _has_manual_control_dependencies(*
dtype0
V
save_1/Identity_44Identitysave_1/RestoreV2:43*
T0*
_output_shapes
:

save_1/AssignVariableOp_43AssignVariableOpconv_d2/w/Adamsave_1/Identity_44*&
 _has_manual_control_dependencies(*
dtype0
V
save_1/Identity_45Identitysave_1/RestoreV2:44*
T0*
_output_shapes
:

save_1/AssignVariableOp_44AssignVariableOpconv_d2/w/Adam_1save_1/Identity_45*&
 _has_manual_control_dependencies(*
dtype0
V
save_1/Identity_46Identitysave_1/RestoreV2:45*
T0*
_output_shapes
:

save_1/AssignVariableOp_45AssignVariableOp
conv_out/bsave_1/Identity_46*&
 _has_manual_control_dependencies(*
dtype0
V
save_1/Identity_47Identitysave_1/RestoreV2:46*
T0*
_output_shapes
:

save_1/AssignVariableOp_46AssignVariableOpconv_out/b/Adamsave_1/Identity_47*&
 _has_manual_control_dependencies(*
dtype0
V
save_1/Identity_48Identitysave_1/RestoreV2:47*
T0*
_output_shapes
:

save_1/AssignVariableOp_47AssignVariableOpconv_out/b/Adam_1save_1/Identity_48*&
 _has_manual_control_dependencies(*
dtype0
V
save_1/Identity_49Identitysave_1/RestoreV2:48*
T0*
_output_shapes
:

save_1/AssignVariableOp_48AssignVariableOp
conv_out/wsave_1/Identity_49*&
 _has_manual_control_dependencies(*
dtype0
V
save_1/Identity_50Identitysave_1/RestoreV2:49*
T0*
_output_shapes
:

save_1/AssignVariableOp_49AssignVariableOpconv_out/w/Adamsave_1/Identity_50*&
 _has_manual_control_dependencies(*
dtype0
V
save_1/Identity_51Identitysave_1/RestoreV2:50*
T0*
_output_shapes
:

save_1/AssignVariableOp_50AssignVariableOpconv_out/w/Adam_1save_1/Identity_51*&
 _has_manual_control_dependencies(*
dtype0
V
save_1/Identity_52Identitysave_1/RestoreV2:51*
T0*
_output_shapes
:

save_1/AssignVariableOp_51AssignVariableOpsbb1/batch_normalization/betasave_1/Identity_52*&
 _has_manual_control_dependencies(*
dtype0
V
save_1/Identity_53Identitysave_1/RestoreV2:52*
T0*
_output_shapes
:

save_1/AssignVariableOp_52AssignVariableOp"sbb1/batch_normalization/beta/Adamsave_1/Identity_53*&
 _has_manual_control_dependencies(*
dtype0
V
save_1/Identity_54Identitysave_1/RestoreV2:53*
T0*
_output_shapes
:

save_1/AssignVariableOp_53AssignVariableOp$sbb1/batch_normalization/beta/Adam_1save_1/Identity_54*&
 _has_manual_control_dependencies(*
dtype0
V
save_1/Identity_55Identitysave_1/RestoreV2:54*
T0*
_output_shapes
:

save_1/AssignVariableOp_54AssignVariableOpsbb1/batch_normalization/gammasave_1/Identity_55*&
 _has_manual_control_dependencies(*
dtype0
V
save_1/Identity_56Identitysave_1/RestoreV2:55*
T0*
_output_shapes
:

save_1/AssignVariableOp_55AssignVariableOp#sbb1/batch_normalization/gamma/Adamsave_1/Identity_56*&
 _has_manual_control_dependencies(*
dtype0
V
save_1/Identity_57Identitysave_1/RestoreV2:56*
T0*
_output_shapes
:

save_1/AssignVariableOp_56AssignVariableOp%sbb1/batch_normalization/gamma/Adam_1save_1/Identity_57*&
 _has_manual_control_dependencies(*
dtype0
V
save_1/Identity_58Identitysave_1/RestoreV2:57*
T0*
_output_shapes
:

save_1/AssignVariableOp_57AssignVariableOp$sbb1/batch_normalization/moving_meansave_1/Identity_58*&
 _has_manual_control_dependencies(*
dtype0
V
save_1/Identity_59Identitysave_1/RestoreV2:58*
T0*
_output_shapes
:
Ą
save_1/AssignVariableOp_58AssignVariableOp(sbb1/batch_normalization/moving_variancesave_1/Identity_59*&
 _has_manual_control_dependencies(*
dtype0
V
save_1/Identity_60Identitysave_1/RestoreV2:59*
T0*
_output_shapes
:

save_1/AssignVariableOp_59AssignVariableOpsbb1/conv1/bsave_1/Identity_60*&
 _has_manual_control_dependencies(*
dtype0
V
save_1/Identity_61Identitysave_1/RestoreV2:60*
T0*
_output_shapes
:

save_1/AssignVariableOp_60AssignVariableOpsbb1/conv1/b/Adamsave_1/Identity_61*&
 _has_manual_control_dependencies(*
dtype0
V
save_1/Identity_62Identitysave_1/RestoreV2:61*
T0*
_output_shapes
:

save_1/AssignVariableOp_61AssignVariableOpsbb1/conv1/b/Adam_1save_1/Identity_62*&
 _has_manual_control_dependencies(*
dtype0
V
save_1/Identity_63Identitysave_1/RestoreV2:62*
T0*
_output_shapes
:

save_1/AssignVariableOp_62AssignVariableOpsbb1/conv1/wsave_1/Identity_63*&
 _has_manual_control_dependencies(*
dtype0
V
save_1/Identity_64Identitysave_1/RestoreV2:63*
T0*
_output_shapes
:

save_1/AssignVariableOp_63AssignVariableOpsbb1/conv1/w/Adamsave_1/Identity_64*&
 _has_manual_control_dependencies(*
dtype0
V
save_1/Identity_65Identitysave_1/RestoreV2:64*
T0*
_output_shapes
:

save_1/AssignVariableOp_64AssignVariableOpsbb1/conv1/w/Adam_1save_1/Identity_65*&
 _has_manual_control_dependencies(*
dtype0
V
save_1/Identity_66Identitysave_1/RestoreV2:65*
T0*
_output_shapes
:

save_1/AssignVariableOp_65AssignVariableOpsbb1/conv2/bsave_1/Identity_66*&
 _has_manual_control_dependencies(*
dtype0
V
save_1/Identity_67Identitysave_1/RestoreV2:66*
T0*
_output_shapes
:

save_1/AssignVariableOp_66AssignVariableOpsbb1/conv2/b/Adamsave_1/Identity_67*&
 _has_manual_control_dependencies(*
dtype0
V
save_1/Identity_68Identitysave_1/RestoreV2:67*
T0*
_output_shapes
:

save_1/AssignVariableOp_67AssignVariableOpsbb1/conv2/b/Adam_1save_1/Identity_68*&
 _has_manual_control_dependencies(*
dtype0
V
save_1/Identity_69Identitysave_1/RestoreV2:68*
T0*
_output_shapes
:

save_1/AssignVariableOp_68AssignVariableOpsbb1/conv2/wsave_1/Identity_69*&
 _has_manual_control_dependencies(*
dtype0
V
save_1/Identity_70Identitysave_1/RestoreV2:69*
T0*
_output_shapes
:

save_1/AssignVariableOp_69AssignVariableOpsbb1/conv2/w/Adamsave_1/Identity_70*&
 _has_manual_control_dependencies(*
dtype0
V
save_1/Identity_71Identitysave_1/RestoreV2:70*
T0*
_output_shapes
:

save_1/AssignVariableOp_70AssignVariableOpsbb1/conv2/w/Adam_1save_1/Identity_71*&
 _has_manual_control_dependencies(*
dtype0
V
save_1/Identity_72Identitysave_1/RestoreV2:71*
T0*
_output_shapes
:

save_1/AssignVariableOp_71AssignVariableOpsbb1/conv3/bsave_1/Identity_72*&
 _has_manual_control_dependencies(*
dtype0
V
save_1/Identity_73Identitysave_1/RestoreV2:72*
T0*
_output_shapes
:

save_1/AssignVariableOp_72AssignVariableOpsbb1/conv3/b/Adamsave_1/Identity_73*&
 _has_manual_control_dependencies(*
dtype0
V
save_1/Identity_74Identitysave_1/RestoreV2:73*
T0*
_output_shapes
:

save_1/AssignVariableOp_73AssignVariableOpsbb1/conv3/b/Adam_1save_1/Identity_74*&
 _has_manual_control_dependencies(*
dtype0
V
save_1/Identity_75Identitysave_1/RestoreV2:74*
T0*
_output_shapes
:

save_1/AssignVariableOp_74AssignVariableOpsbb1/conv3/wsave_1/Identity_75*&
 _has_manual_control_dependencies(*
dtype0
V
save_1/Identity_76Identitysave_1/RestoreV2:75*
T0*
_output_shapes
:

save_1/AssignVariableOp_75AssignVariableOpsbb1/conv3/w/Adamsave_1/Identity_76*&
 _has_manual_control_dependencies(*
dtype0
V
save_1/Identity_77Identitysave_1/RestoreV2:76*
T0*
_output_shapes
:

save_1/AssignVariableOp_76AssignVariableOpsbb1/conv3/w/Adam_1save_1/Identity_77*&
 _has_manual_control_dependencies(*
dtype0
V
save_1/Identity_78Identitysave_1/RestoreV2:77*
T0*
_output_shapes
:

save_1/AssignVariableOp_77AssignVariableOpsbb1/conv4/bsave_1/Identity_78*&
 _has_manual_control_dependencies(*
dtype0
V
save_1/Identity_79Identitysave_1/RestoreV2:78*
T0*
_output_shapes
:

save_1/AssignVariableOp_78AssignVariableOpsbb1/conv4/b/Adamsave_1/Identity_79*&
 _has_manual_control_dependencies(*
dtype0
V
save_1/Identity_80Identitysave_1/RestoreV2:79*
T0*
_output_shapes
:

save_1/AssignVariableOp_79AssignVariableOpsbb1/conv4/b/Adam_1save_1/Identity_80*&
 _has_manual_control_dependencies(*
dtype0
V
save_1/Identity_81Identitysave_1/RestoreV2:80*
T0*
_output_shapes
:

save_1/AssignVariableOp_80AssignVariableOpsbb1/conv4/wsave_1/Identity_81*&
 _has_manual_control_dependencies(*
dtype0
V
save_1/Identity_82Identitysave_1/RestoreV2:81*
T0*
_output_shapes
:

save_1/AssignVariableOp_81AssignVariableOpsbb1/conv4/w/Adamsave_1/Identity_82*&
 _has_manual_control_dependencies(*
dtype0
V
save_1/Identity_83Identitysave_1/RestoreV2:82*
T0*
_output_shapes
:

save_1/AssignVariableOp_82AssignVariableOpsbb1/conv4/w/Adam_1save_1/Identity_83*&
 _has_manual_control_dependencies(*
dtype0
V
save_1/Identity_84Identitysave_1/RestoreV2:83*
T0*
_output_shapes
:

save_1/AssignVariableOp_83AssignVariableOpsbb2/batch_normalization/betasave_1/Identity_84*&
 _has_manual_control_dependencies(*
dtype0
V
save_1/Identity_85Identitysave_1/RestoreV2:84*
T0*
_output_shapes
:

save_1/AssignVariableOp_84AssignVariableOp"sbb2/batch_normalization/beta/Adamsave_1/Identity_85*&
 _has_manual_control_dependencies(*
dtype0
V
save_1/Identity_86Identitysave_1/RestoreV2:85*
T0*
_output_shapes
:

save_1/AssignVariableOp_85AssignVariableOp$sbb2/batch_normalization/beta/Adam_1save_1/Identity_86*&
 _has_manual_control_dependencies(*
dtype0
V
save_1/Identity_87Identitysave_1/RestoreV2:86*
T0*
_output_shapes
:

save_1/AssignVariableOp_86AssignVariableOpsbb2/batch_normalization/gammasave_1/Identity_87*&
 _has_manual_control_dependencies(*
dtype0
V
save_1/Identity_88Identitysave_1/RestoreV2:87*
T0*
_output_shapes
:

save_1/AssignVariableOp_87AssignVariableOp#sbb2/batch_normalization/gamma/Adamsave_1/Identity_88*&
 _has_manual_control_dependencies(*
dtype0
V
save_1/Identity_89Identitysave_1/RestoreV2:88*
T0*
_output_shapes
:

save_1/AssignVariableOp_88AssignVariableOp%sbb2/batch_normalization/gamma/Adam_1save_1/Identity_89*&
 _has_manual_control_dependencies(*
dtype0
V
save_1/Identity_90Identitysave_1/RestoreV2:89*
T0*
_output_shapes
:

save_1/AssignVariableOp_89AssignVariableOp$sbb2/batch_normalization/moving_meansave_1/Identity_90*&
 _has_manual_control_dependencies(*
dtype0
V
save_1/Identity_91Identitysave_1/RestoreV2:90*
T0*
_output_shapes
:
Ą
save_1/AssignVariableOp_90AssignVariableOp(sbb2/batch_normalization/moving_variancesave_1/Identity_91*&
 _has_manual_control_dependencies(*
dtype0
V
save_1/Identity_92Identitysave_1/RestoreV2:91*
T0*
_output_shapes
:

save_1/AssignVariableOp_91AssignVariableOpsbb2/conv1/bsave_1/Identity_92*&
 _has_manual_control_dependencies(*
dtype0
V
save_1/Identity_93Identitysave_1/RestoreV2:92*
T0*
_output_shapes
:

save_1/AssignVariableOp_92AssignVariableOpsbb2/conv1/b/Adamsave_1/Identity_93*&
 _has_manual_control_dependencies(*
dtype0
V
save_1/Identity_94Identitysave_1/RestoreV2:93*
T0*
_output_shapes
:

save_1/AssignVariableOp_93AssignVariableOpsbb2/conv1/b/Adam_1save_1/Identity_94*&
 _has_manual_control_dependencies(*
dtype0
V
save_1/Identity_95Identitysave_1/RestoreV2:94*
T0*
_output_shapes
:

save_1/AssignVariableOp_94AssignVariableOpsbb2/conv1/wsave_1/Identity_95*&
 _has_manual_control_dependencies(*
dtype0
V
save_1/Identity_96Identitysave_1/RestoreV2:95*
T0*
_output_shapes
:

save_1/AssignVariableOp_95AssignVariableOpsbb2/conv1/w/Adamsave_1/Identity_96*&
 _has_manual_control_dependencies(*
dtype0
V
save_1/Identity_97Identitysave_1/RestoreV2:96*
T0*
_output_shapes
:

save_1/AssignVariableOp_96AssignVariableOpsbb2/conv1/w/Adam_1save_1/Identity_97*&
 _has_manual_control_dependencies(*
dtype0
V
save_1/Identity_98Identitysave_1/RestoreV2:97*
T0*
_output_shapes
:

save_1/AssignVariableOp_97AssignVariableOpsbb2/conv2/bsave_1/Identity_98*&
 _has_manual_control_dependencies(*
dtype0
V
save_1/Identity_99Identitysave_1/RestoreV2:98*
T0*
_output_shapes
:

save_1/AssignVariableOp_98AssignVariableOpsbb2/conv2/b/Adamsave_1/Identity_99*&
 _has_manual_control_dependencies(*
dtype0
W
save_1/Identity_100Identitysave_1/RestoreV2:99*
T0*
_output_shapes
:

save_1/AssignVariableOp_99AssignVariableOpsbb2/conv2/b/Adam_1save_1/Identity_100*&
 _has_manual_control_dependencies(*
dtype0
X
save_1/Identity_101Identitysave_1/RestoreV2:100*
T0*
_output_shapes
:

save_1/AssignVariableOp_100AssignVariableOpsbb2/conv2/wsave_1/Identity_101*&
 _has_manual_control_dependencies(*
dtype0
X
save_1/Identity_102Identitysave_1/RestoreV2:101*
T0*
_output_shapes
:

save_1/AssignVariableOp_101AssignVariableOpsbb2/conv2/w/Adamsave_1/Identity_102*&
 _has_manual_control_dependencies(*
dtype0
X
save_1/Identity_103Identitysave_1/RestoreV2:102*
T0*
_output_shapes
:

save_1/AssignVariableOp_102AssignVariableOpsbb2/conv2/w/Adam_1save_1/Identity_103*&
 _has_manual_control_dependencies(*
dtype0
X
save_1/Identity_104Identitysave_1/RestoreV2:103*
T0*
_output_shapes
:

save_1/AssignVariableOp_103AssignVariableOpsbb2/conv3/bsave_1/Identity_104*&
 _has_manual_control_dependencies(*
dtype0
X
save_1/Identity_105Identitysave_1/RestoreV2:104*
T0*
_output_shapes
:

save_1/AssignVariableOp_104AssignVariableOpsbb2/conv3/b/Adamsave_1/Identity_105*&
 _has_manual_control_dependencies(*
dtype0
X
save_1/Identity_106Identitysave_1/RestoreV2:105*
T0*
_output_shapes
:

save_1/AssignVariableOp_105AssignVariableOpsbb2/conv3/b/Adam_1save_1/Identity_106*&
 _has_manual_control_dependencies(*
dtype0
X
save_1/Identity_107Identitysave_1/RestoreV2:106*
T0*
_output_shapes
:

save_1/AssignVariableOp_106AssignVariableOpsbb2/conv3/wsave_1/Identity_107*&
 _has_manual_control_dependencies(*
dtype0
X
save_1/Identity_108Identitysave_1/RestoreV2:107*
T0*
_output_shapes
:

save_1/AssignVariableOp_107AssignVariableOpsbb2/conv3/w/Adamsave_1/Identity_108*&
 _has_manual_control_dependencies(*
dtype0
X
save_1/Identity_109Identitysave_1/RestoreV2:108*
T0*
_output_shapes
:

save_1/AssignVariableOp_108AssignVariableOpsbb2/conv3/w/Adam_1save_1/Identity_109*&
 _has_manual_control_dependencies(*
dtype0
X
save_1/Identity_110Identitysave_1/RestoreV2:109*
T0*
_output_shapes
:

save_1/AssignVariableOp_109AssignVariableOpsbb2/conv4/bsave_1/Identity_110*&
 _has_manual_control_dependencies(*
dtype0
X
save_1/Identity_111Identitysave_1/RestoreV2:110*
T0*
_output_shapes
:

save_1/AssignVariableOp_110AssignVariableOpsbb2/conv4/b/Adamsave_1/Identity_111*&
 _has_manual_control_dependencies(*
dtype0
X
save_1/Identity_112Identitysave_1/RestoreV2:111*
T0*
_output_shapes
:

save_1/AssignVariableOp_111AssignVariableOpsbb2/conv4/b/Adam_1save_1/Identity_112*&
 _has_manual_control_dependencies(*
dtype0
X
save_1/Identity_113Identitysave_1/RestoreV2:112*
T0*
_output_shapes
:

save_1/AssignVariableOp_112AssignVariableOpsbb2/conv4/wsave_1/Identity_113*&
 _has_manual_control_dependencies(*
dtype0
X
save_1/Identity_114Identitysave_1/RestoreV2:113*
T0*
_output_shapes
:

save_1/AssignVariableOp_113AssignVariableOpsbb2/conv4/w/Adamsave_1/Identity_114*&
 _has_manual_control_dependencies(*
dtype0
X
save_1/Identity_115Identitysave_1/RestoreV2:114*
T0*
_output_shapes
:

save_1/AssignVariableOp_114AssignVariableOpsbb2/conv4/w/Adam_1save_1/Identity_115*&
 _has_manual_control_dependencies(*
dtype0
X
save_1/Identity_116Identitysave_1/RestoreV2:115*
T0*
_output_shapes
:

save_1/AssignVariableOp_115AssignVariableOpsbb3/batch_normalization/betasave_1/Identity_116*&
 _has_manual_control_dependencies(*
dtype0
X
save_1/Identity_117Identitysave_1/RestoreV2:116*
T0*
_output_shapes
:

save_1/AssignVariableOp_116AssignVariableOp"sbb3/batch_normalization/beta/Adamsave_1/Identity_117*&
 _has_manual_control_dependencies(*
dtype0
X
save_1/Identity_118Identitysave_1/RestoreV2:117*
T0*
_output_shapes
:

save_1/AssignVariableOp_117AssignVariableOp$sbb3/batch_normalization/beta/Adam_1save_1/Identity_118*&
 _has_manual_control_dependencies(*
dtype0
X
save_1/Identity_119Identitysave_1/RestoreV2:118*
T0*
_output_shapes
:

save_1/AssignVariableOp_118AssignVariableOpsbb3/batch_normalization/gammasave_1/Identity_119*&
 _has_manual_control_dependencies(*
dtype0
X
save_1/Identity_120Identitysave_1/RestoreV2:119*
T0*
_output_shapes
:

save_1/AssignVariableOp_119AssignVariableOp#sbb3/batch_normalization/gamma/Adamsave_1/Identity_120*&
 _has_manual_control_dependencies(*
dtype0
X
save_1/Identity_121Identitysave_1/RestoreV2:120*
T0*
_output_shapes
:
 
save_1/AssignVariableOp_120AssignVariableOp%sbb3/batch_normalization/gamma/Adam_1save_1/Identity_121*&
 _has_manual_control_dependencies(*
dtype0
X
save_1/Identity_122Identitysave_1/RestoreV2:121*
T0*
_output_shapes
:

save_1/AssignVariableOp_121AssignVariableOp$sbb3/batch_normalization/moving_meansave_1/Identity_122*&
 _has_manual_control_dependencies(*
dtype0
X
save_1/Identity_123Identitysave_1/RestoreV2:122*
T0*
_output_shapes
:
Ł
save_1/AssignVariableOp_122AssignVariableOp(sbb3/batch_normalization/moving_variancesave_1/Identity_123*&
 _has_manual_control_dependencies(*
dtype0
X
save_1/Identity_124Identitysave_1/RestoreV2:123*
T0*
_output_shapes
:

save_1/AssignVariableOp_123AssignVariableOpsbb3/conv1/bsave_1/Identity_124*&
 _has_manual_control_dependencies(*
dtype0
X
save_1/Identity_125Identitysave_1/RestoreV2:124*
T0*
_output_shapes
:

save_1/AssignVariableOp_124AssignVariableOpsbb3/conv1/b/Adamsave_1/Identity_125*&
 _has_manual_control_dependencies(*
dtype0
X
save_1/Identity_126Identitysave_1/RestoreV2:125*
T0*
_output_shapes
:

save_1/AssignVariableOp_125AssignVariableOpsbb3/conv1/b/Adam_1save_1/Identity_126*&
 _has_manual_control_dependencies(*
dtype0
X
save_1/Identity_127Identitysave_1/RestoreV2:126*
T0*
_output_shapes
:

save_1/AssignVariableOp_126AssignVariableOpsbb3/conv1/wsave_1/Identity_127*&
 _has_manual_control_dependencies(*
dtype0
X
save_1/Identity_128Identitysave_1/RestoreV2:127*
T0*
_output_shapes
:

save_1/AssignVariableOp_127AssignVariableOpsbb3/conv1/w/Adamsave_1/Identity_128*&
 _has_manual_control_dependencies(*
dtype0
X
save_1/Identity_129Identitysave_1/RestoreV2:128*
T0*
_output_shapes
:

save_1/AssignVariableOp_128AssignVariableOpsbb3/conv1/w/Adam_1save_1/Identity_129*&
 _has_manual_control_dependencies(*
dtype0
X
save_1/Identity_130Identitysave_1/RestoreV2:129*
T0*
_output_shapes
:

save_1/AssignVariableOp_129AssignVariableOpsbb3/conv2/bsave_1/Identity_130*&
 _has_manual_control_dependencies(*
dtype0
X
save_1/Identity_131Identitysave_1/RestoreV2:130*
T0*
_output_shapes
:

save_1/AssignVariableOp_130AssignVariableOpsbb3/conv2/b/Adamsave_1/Identity_131*&
 _has_manual_control_dependencies(*
dtype0
X
save_1/Identity_132Identitysave_1/RestoreV2:131*
T0*
_output_shapes
:

save_1/AssignVariableOp_131AssignVariableOpsbb3/conv2/b/Adam_1save_1/Identity_132*&
 _has_manual_control_dependencies(*
dtype0
X
save_1/Identity_133Identitysave_1/RestoreV2:132*
T0*
_output_shapes
:

save_1/AssignVariableOp_132AssignVariableOpsbb3/conv2/wsave_1/Identity_133*&
 _has_manual_control_dependencies(*
dtype0
X
save_1/Identity_134Identitysave_1/RestoreV2:133*
T0*
_output_shapes
:

save_1/AssignVariableOp_133AssignVariableOpsbb3/conv2/w/Adamsave_1/Identity_134*&
 _has_manual_control_dependencies(*
dtype0
X
save_1/Identity_135Identitysave_1/RestoreV2:134*
T0*
_output_shapes
:

save_1/AssignVariableOp_134AssignVariableOpsbb3/conv2/w/Adam_1save_1/Identity_135*&
 _has_manual_control_dependencies(*
dtype0
X
save_1/Identity_136Identitysave_1/RestoreV2:135*
T0*
_output_shapes
:

save_1/AssignVariableOp_135AssignVariableOpsbb3/conv3/bsave_1/Identity_136*&
 _has_manual_control_dependencies(*
dtype0
X
save_1/Identity_137Identitysave_1/RestoreV2:136*
T0*
_output_shapes
:

save_1/AssignVariableOp_136AssignVariableOpsbb3/conv3/b/Adamsave_1/Identity_137*&
 _has_manual_control_dependencies(*
dtype0
X
save_1/Identity_138Identitysave_1/RestoreV2:137*
T0*
_output_shapes
:

save_1/AssignVariableOp_137AssignVariableOpsbb3/conv3/b/Adam_1save_1/Identity_138*&
 _has_manual_control_dependencies(*
dtype0
X
save_1/Identity_139Identitysave_1/RestoreV2:138*
T0*
_output_shapes
:

save_1/AssignVariableOp_138AssignVariableOpsbb3/conv3/wsave_1/Identity_139*&
 _has_manual_control_dependencies(*
dtype0
X
save_1/Identity_140Identitysave_1/RestoreV2:139*
T0*
_output_shapes
:

save_1/AssignVariableOp_139AssignVariableOpsbb3/conv3/w/Adamsave_1/Identity_140*&
 _has_manual_control_dependencies(*
dtype0
X
save_1/Identity_141Identitysave_1/RestoreV2:140*
T0*
_output_shapes
:

save_1/AssignVariableOp_140AssignVariableOpsbb3/conv3/w/Adam_1save_1/Identity_141*&
 _has_manual_control_dependencies(*
dtype0
X
save_1/Identity_142Identitysave_1/RestoreV2:141*
T0*
_output_shapes
:

save_1/AssignVariableOp_141AssignVariableOpsbb3/conv4/bsave_1/Identity_142*&
 _has_manual_control_dependencies(*
dtype0
X
save_1/Identity_143Identitysave_1/RestoreV2:142*
T0*
_output_shapes
:

save_1/AssignVariableOp_142AssignVariableOpsbb3/conv4/b/Adamsave_1/Identity_143*&
 _has_manual_control_dependencies(*
dtype0
X
save_1/Identity_144Identitysave_1/RestoreV2:143*
T0*
_output_shapes
:

save_1/AssignVariableOp_143AssignVariableOpsbb3/conv4/b/Adam_1save_1/Identity_144*&
 _has_manual_control_dependencies(*
dtype0
X
save_1/Identity_145Identitysave_1/RestoreV2:144*
T0*
_output_shapes
:

save_1/AssignVariableOp_144AssignVariableOpsbb3/conv4/wsave_1/Identity_145*&
 _has_manual_control_dependencies(*
dtype0
X
save_1/Identity_146Identitysave_1/RestoreV2:145*
T0*
_output_shapes
:

save_1/AssignVariableOp_145AssignVariableOpsbb3/conv4/w/Adamsave_1/Identity_146*&
 _has_manual_control_dependencies(*
dtype0
X
save_1/Identity_147Identitysave_1/RestoreV2:146*
T0*
_output_shapes
:

save_1/AssignVariableOp_146AssignVariableOpsbb3/conv4/w/Adam_1save_1/Identity_147*&
 _has_manual_control_dependencies(*
dtype0
"
save_1/restore_shardNoOp^save_1/AssignVariableOp^save_1/AssignVariableOp_1^save_1/AssignVariableOp_10^save_1/AssignVariableOp_100^save_1/AssignVariableOp_101^save_1/AssignVariableOp_102^save_1/AssignVariableOp_103^save_1/AssignVariableOp_104^save_1/AssignVariableOp_105^save_1/AssignVariableOp_106^save_1/AssignVariableOp_107^save_1/AssignVariableOp_108^save_1/AssignVariableOp_109^save_1/AssignVariableOp_11^save_1/AssignVariableOp_110^save_1/AssignVariableOp_111^save_1/AssignVariableOp_112^save_1/AssignVariableOp_113^save_1/AssignVariableOp_114^save_1/AssignVariableOp_115^save_1/AssignVariableOp_116^save_1/AssignVariableOp_117^save_1/AssignVariableOp_118^save_1/AssignVariableOp_119^save_1/AssignVariableOp_12^save_1/AssignVariableOp_120^save_1/AssignVariableOp_121^save_1/AssignVariableOp_122^save_1/AssignVariableOp_123^save_1/AssignVariableOp_124^save_1/AssignVariableOp_125^save_1/AssignVariableOp_126^save_1/AssignVariableOp_127^save_1/AssignVariableOp_128^save_1/AssignVariableOp_129^save_1/AssignVariableOp_13^save_1/AssignVariableOp_130^save_1/AssignVariableOp_131^save_1/AssignVariableOp_132^save_1/AssignVariableOp_133^save_1/AssignVariableOp_134^save_1/AssignVariableOp_135^save_1/AssignVariableOp_136^save_1/AssignVariableOp_137^save_1/AssignVariableOp_138^save_1/AssignVariableOp_139^save_1/AssignVariableOp_14^save_1/AssignVariableOp_140^save_1/AssignVariableOp_141^save_1/AssignVariableOp_142^save_1/AssignVariableOp_143^save_1/AssignVariableOp_144^save_1/AssignVariableOp_145^save_1/AssignVariableOp_146^save_1/AssignVariableOp_15^save_1/AssignVariableOp_16^save_1/AssignVariableOp_17^save_1/AssignVariableOp_18^save_1/AssignVariableOp_19^save_1/AssignVariableOp_2^save_1/AssignVariableOp_20^save_1/AssignVariableOp_21^save_1/AssignVariableOp_22^save_1/AssignVariableOp_23^save_1/AssignVariableOp_24^save_1/AssignVariableOp_25^save_1/AssignVariableOp_26^save_1/AssignVariableOp_27^save_1/AssignVariableOp_28^save_1/AssignVariableOp_29^save_1/AssignVariableOp_3^save_1/AssignVariableOp_30^save_1/AssignVariableOp_31^save_1/AssignVariableOp_32^save_1/AssignVariableOp_33^save_1/AssignVariableOp_34^save_1/AssignVariableOp_35^save_1/AssignVariableOp_36^save_1/AssignVariableOp_37^save_1/AssignVariableOp_38^save_1/AssignVariableOp_39^save_1/AssignVariableOp_4^save_1/AssignVariableOp_40^save_1/AssignVariableOp_41^save_1/AssignVariableOp_42^save_1/AssignVariableOp_43^save_1/AssignVariableOp_44^save_1/AssignVariableOp_45^save_1/AssignVariableOp_46^save_1/AssignVariableOp_47^save_1/AssignVariableOp_48^save_1/AssignVariableOp_49^save_1/AssignVariableOp_5^save_1/AssignVariableOp_50^save_1/AssignVariableOp_51^save_1/AssignVariableOp_52^save_1/AssignVariableOp_53^save_1/AssignVariableOp_54^save_1/AssignVariableOp_55^save_1/AssignVariableOp_56^save_1/AssignVariableOp_57^save_1/AssignVariableOp_58^save_1/AssignVariableOp_59^save_1/AssignVariableOp_6^save_1/AssignVariableOp_60^save_1/AssignVariableOp_61^save_1/AssignVariableOp_62^save_1/AssignVariableOp_63^save_1/AssignVariableOp_64^save_1/AssignVariableOp_65^save_1/AssignVariableOp_66^save_1/AssignVariableOp_67^save_1/AssignVariableOp_68^save_1/AssignVariableOp_69^save_1/AssignVariableOp_7^save_1/AssignVariableOp_70^save_1/AssignVariableOp_71^save_1/AssignVariableOp_72^save_1/AssignVariableOp_73^save_1/AssignVariableOp_74^save_1/AssignVariableOp_75^save_1/AssignVariableOp_76^save_1/AssignVariableOp_77^save_1/AssignVariableOp_78^save_1/AssignVariableOp_79^save_1/AssignVariableOp_8^save_1/AssignVariableOp_80^save_1/AssignVariableOp_81^save_1/AssignVariableOp_82^save_1/AssignVariableOp_83^save_1/AssignVariableOp_84^save_1/AssignVariableOp_85^save_1/AssignVariableOp_86^save_1/AssignVariableOp_87^save_1/AssignVariableOp_88^save_1/AssignVariableOp_89^save_1/AssignVariableOp_9^save_1/AssignVariableOp_90^save_1/AssignVariableOp_91^save_1/AssignVariableOp_92^save_1/AssignVariableOp_93^save_1/AssignVariableOp_94^save_1/AssignVariableOp_95^save_1/AssignVariableOp_96^save_1/AssignVariableOp_97^save_1/AssignVariableOp_98^save_1/AssignVariableOp_99*&
 _has_manual_control_dependencies(
1
save_1/restore_allNoOp^save_1/restore_shard"
B
save_1/Const:0save_1/Identity:0save_1/restore_all (5 @F8"*
saved_model_main_op

init_all_tables"
train_op

Adam"Ľ-
trainable_variables--
d
	conv1/w:0conv1/w/Assignconv1/w/Read/ReadVariableOp:0(2$conv1/w/Initializer/random_uniform:08
[
	conv1/b:0conv1/b/Assignconv1/b/Read/ReadVariableOp:0(2conv1/b/Initializer/Const:08
˘
batch_normalization/gamma:0 batch_normalization/gamma/Assign/batch_normalization/gamma/Read/ReadVariableOp:0(2,batch_normalization/gamma/Initializer/ones:08

batch_normalization/beta:0batch_normalization/beta/Assign.batch_normalization/beta/Read/ReadVariableOp:0(2,batch_normalization/beta/Initializer/zeros:08
x
sbb1/conv1/w:0sbb1/conv1/w/Assign"sbb1/conv1/w/Read/ReadVariableOp:0(2)sbb1/conv1/w/Initializer/random_uniform:08
o
sbb1/conv1/b:0sbb1/conv1/b/Assign"sbb1/conv1/b/Read/ReadVariableOp:0(2 sbb1/conv1/b/Initializer/Const:08
x
sbb1/conv2/w:0sbb1/conv2/w/Assign"sbb1/conv2/w/Read/ReadVariableOp:0(2)sbb1/conv2/w/Initializer/random_uniform:08
o
sbb1/conv2/b:0sbb1/conv2/b/Assign"sbb1/conv2/b/Read/ReadVariableOp:0(2 sbb1/conv2/b/Initializer/Const:08
x
sbb1/conv3/w:0sbb1/conv3/w/Assign"sbb1/conv3/w/Read/ReadVariableOp:0(2)sbb1/conv3/w/Initializer/random_uniform:08
o
sbb1/conv3/b:0sbb1/conv3/b/Assign"sbb1/conv3/b/Read/ReadVariableOp:0(2 sbb1/conv3/b/Initializer/Const:08
x
sbb1/conv4/w:0sbb1/conv4/w/Assign"sbb1/conv4/w/Read/ReadVariableOp:0(2)sbb1/conv4/w/Initializer/random_uniform:08
o
sbb1/conv4/b:0sbb1/conv4/b/Assign"sbb1/conv4/b/Read/ReadVariableOp:0(2 sbb1/conv4/b/Initializer/Const:08
ś
 sbb1/batch_normalization/gamma:0%sbb1/batch_normalization/gamma/Assign4sbb1/batch_normalization/gamma/Read/ReadVariableOp:0(21sbb1/batch_normalization/gamma/Initializer/ones:08
ł
sbb1/batch_normalization/beta:0$sbb1/batch_normalization/beta/Assign3sbb1/batch_normalization/beta/Read/ReadVariableOp:0(21sbb1/batch_normalization/beta/Initializer/zeros:08
x
sbb2/conv1/w:0sbb2/conv1/w/Assign"sbb2/conv1/w/Read/ReadVariableOp:0(2)sbb2/conv1/w/Initializer/random_uniform:08
o
sbb2/conv1/b:0sbb2/conv1/b/Assign"sbb2/conv1/b/Read/ReadVariableOp:0(2 sbb2/conv1/b/Initializer/Const:08
x
sbb2/conv2/w:0sbb2/conv2/w/Assign"sbb2/conv2/w/Read/ReadVariableOp:0(2)sbb2/conv2/w/Initializer/random_uniform:08
o
sbb2/conv2/b:0sbb2/conv2/b/Assign"sbb2/conv2/b/Read/ReadVariableOp:0(2 sbb2/conv2/b/Initializer/Const:08
x
sbb2/conv3/w:0sbb2/conv3/w/Assign"sbb2/conv3/w/Read/ReadVariableOp:0(2)sbb2/conv3/w/Initializer/random_uniform:08
o
sbb2/conv3/b:0sbb2/conv3/b/Assign"sbb2/conv3/b/Read/ReadVariableOp:0(2 sbb2/conv3/b/Initializer/Const:08
x
sbb2/conv4/w:0sbb2/conv4/w/Assign"sbb2/conv4/w/Read/ReadVariableOp:0(2)sbb2/conv4/w/Initializer/random_uniform:08
o
sbb2/conv4/b:0sbb2/conv4/b/Assign"sbb2/conv4/b/Read/ReadVariableOp:0(2 sbb2/conv4/b/Initializer/Const:08
ś
 sbb2/batch_normalization/gamma:0%sbb2/batch_normalization/gamma/Assign4sbb2/batch_normalization/gamma/Read/ReadVariableOp:0(21sbb2/batch_normalization/gamma/Initializer/ones:08
ł
sbb2/batch_normalization/beta:0$sbb2/batch_normalization/beta/Assign3sbb2/batch_normalization/beta/Read/ReadVariableOp:0(21sbb2/batch_normalization/beta/Initializer/zeros:08
x
sbb3/conv1/w:0sbb3/conv1/w/Assign"sbb3/conv1/w/Read/ReadVariableOp:0(2)sbb3/conv1/w/Initializer/random_uniform:08
o
sbb3/conv1/b:0sbb3/conv1/b/Assign"sbb3/conv1/b/Read/ReadVariableOp:0(2 sbb3/conv1/b/Initializer/Const:08
x
sbb3/conv2/w:0sbb3/conv2/w/Assign"sbb3/conv2/w/Read/ReadVariableOp:0(2)sbb3/conv2/w/Initializer/random_uniform:08
o
sbb3/conv2/b:0sbb3/conv2/b/Assign"sbb3/conv2/b/Read/ReadVariableOp:0(2 sbb3/conv2/b/Initializer/Const:08
x
sbb3/conv3/w:0sbb3/conv3/w/Assign"sbb3/conv3/w/Read/ReadVariableOp:0(2)sbb3/conv3/w/Initializer/random_uniform:08
o
sbb3/conv3/b:0sbb3/conv3/b/Assign"sbb3/conv3/b/Read/ReadVariableOp:0(2 sbb3/conv3/b/Initializer/Const:08
x
sbb3/conv4/w:0sbb3/conv4/w/Assign"sbb3/conv4/w/Read/ReadVariableOp:0(2)sbb3/conv4/w/Initializer/random_uniform:08
o
sbb3/conv4/b:0sbb3/conv4/b/Assign"sbb3/conv4/b/Read/ReadVariableOp:0(2 sbb3/conv4/b/Initializer/Const:08
ś
 sbb3/batch_normalization/gamma:0%sbb3/batch_normalization/gamma/Assign4sbb3/batch_normalization/gamma/Read/ReadVariableOp:0(21sbb3/batch_normalization/gamma/Initializer/ones:08
ł
sbb3/batch_normalization/beta:0$sbb3/batch_normalization/beta/Assign3sbb3/batch_normalization/beta/Read/ReadVariableOp:0(21sbb3/batch_normalization/beta/Initializer/zeros:08
l
conv_d1/w:0conv_d1/w/Assignconv_d1/w/Read/ReadVariableOp:0(2&conv_d1/w/Initializer/random_uniform:08
c
conv_d1/b:0conv_d1/b/Assignconv_d1/b/Read/ReadVariableOp:0(2conv_d1/b/Initializer/Const:08
Ş
batch_normalization_1/gamma:0"batch_normalization_1/gamma/Assign1batch_normalization_1/gamma/Read/ReadVariableOp:0(2.batch_normalization_1/gamma/Initializer/ones:08
§
batch_normalization_1/beta:0!batch_normalization_1/beta/Assign0batch_normalization_1/beta/Read/ReadVariableOp:0(2.batch_normalization_1/beta/Initializer/zeros:08
l
conv_d2/w:0conv_d2/w/Assignconv_d2/w/Read/ReadVariableOp:0(2&conv_d2/w/Initializer/random_uniform:08
c
conv_d2/b:0conv_d2/b/Assignconv_d2/b/Read/ReadVariableOp:0(2conv_d2/b/Initializer/Const:08
Ş
batch_normalization_2/gamma:0"batch_normalization_2/gamma/Assign1batch_normalization_2/gamma/Read/ReadVariableOp:0(2.batch_normalization_2/gamma/Initializer/ones:08
§
batch_normalization_2/beta:0!batch_normalization_2/beta/Assign0batch_normalization_2/beta/Read/ReadVariableOp:0(2.batch_normalization_2/beta/Initializer/zeros:08
p
conv_out/w:0conv_out/w/Assign conv_out/w/Read/ReadVariableOp:0(2'conv_out/w/Initializer/random_uniform:08
g
conv_out/b:0conv_out/b/Assign conv_out/b/Read/ReadVariableOp:0(2conv_out/b/Initializer/Const:08"Ť
	variablesŤŤ
d
	conv1/w:0conv1/w/Assignconv1/w/Read/ReadVariableOp:0(2$conv1/w/Initializer/random_uniform:08
[
	conv1/b:0conv1/b/Assignconv1/b/Read/ReadVariableOp:0(2conv1/b/Initializer/Const:08
˘
batch_normalization/gamma:0 batch_normalization/gamma/Assign/batch_normalization/gamma/Read/ReadVariableOp:0(2,batch_normalization/gamma/Initializer/ones:08

batch_normalization/beta:0batch_normalization/beta/Assign.batch_normalization/beta/Read/ReadVariableOp:0(2,batch_normalization/beta/Initializer/zeros:08
˝
!batch_normalization/moving_mean:0&batch_normalization/moving_mean/Assign5batch_normalization/moving_mean/Read/ReadVariableOp:0(23batch_normalization/moving_mean/Initializer/zeros:0@H
Ě
%batch_normalization/moving_variance:0*batch_normalization/moving_variance/Assign9batch_normalization/moving_variance/Read/ReadVariableOp:0(26batch_normalization/moving_variance/Initializer/ones:0@H
x
sbb1/conv1/w:0sbb1/conv1/w/Assign"sbb1/conv1/w/Read/ReadVariableOp:0(2)sbb1/conv1/w/Initializer/random_uniform:08
o
sbb1/conv1/b:0sbb1/conv1/b/Assign"sbb1/conv1/b/Read/ReadVariableOp:0(2 sbb1/conv1/b/Initializer/Const:08
x
sbb1/conv2/w:0sbb1/conv2/w/Assign"sbb1/conv2/w/Read/ReadVariableOp:0(2)sbb1/conv2/w/Initializer/random_uniform:08
o
sbb1/conv2/b:0sbb1/conv2/b/Assign"sbb1/conv2/b/Read/ReadVariableOp:0(2 sbb1/conv2/b/Initializer/Const:08
x
sbb1/conv3/w:0sbb1/conv3/w/Assign"sbb1/conv3/w/Read/ReadVariableOp:0(2)sbb1/conv3/w/Initializer/random_uniform:08
o
sbb1/conv3/b:0sbb1/conv3/b/Assign"sbb1/conv3/b/Read/ReadVariableOp:0(2 sbb1/conv3/b/Initializer/Const:08
x
sbb1/conv4/w:0sbb1/conv4/w/Assign"sbb1/conv4/w/Read/ReadVariableOp:0(2)sbb1/conv4/w/Initializer/random_uniform:08
o
sbb1/conv4/b:0sbb1/conv4/b/Assign"sbb1/conv4/b/Read/ReadVariableOp:0(2 sbb1/conv4/b/Initializer/Const:08
ś
 sbb1/batch_normalization/gamma:0%sbb1/batch_normalization/gamma/Assign4sbb1/batch_normalization/gamma/Read/ReadVariableOp:0(21sbb1/batch_normalization/gamma/Initializer/ones:08
ł
sbb1/batch_normalization/beta:0$sbb1/batch_normalization/beta/Assign3sbb1/batch_normalization/beta/Read/ReadVariableOp:0(21sbb1/batch_normalization/beta/Initializer/zeros:08
Ń
&sbb1/batch_normalization/moving_mean:0+sbb1/batch_normalization/moving_mean/Assign:sbb1/batch_normalization/moving_mean/Read/ReadVariableOp:0(28sbb1/batch_normalization/moving_mean/Initializer/zeros:0@H
ŕ
*sbb1/batch_normalization/moving_variance:0/sbb1/batch_normalization/moving_variance/Assign>sbb1/batch_normalization/moving_variance/Read/ReadVariableOp:0(2;sbb1/batch_normalization/moving_variance/Initializer/ones:0@H
x
sbb2/conv1/w:0sbb2/conv1/w/Assign"sbb2/conv1/w/Read/ReadVariableOp:0(2)sbb2/conv1/w/Initializer/random_uniform:08
o
sbb2/conv1/b:0sbb2/conv1/b/Assign"sbb2/conv1/b/Read/ReadVariableOp:0(2 sbb2/conv1/b/Initializer/Const:08
x
sbb2/conv2/w:0sbb2/conv2/w/Assign"sbb2/conv2/w/Read/ReadVariableOp:0(2)sbb2/conv2/w/Initializer/random_uniform:08
o
sbb2/conv2/b:0sbb2/conv2/b/Assign"sbb2/conv2/b/Read/ReadVariableOp:0(2 sbb2/conv2/b/Initializer/Const:08
x
sbb2/conv3/w:0sbb2/conv3/w/Assign"sbb2/conv3/w/Read/ReadVariableOp:0(2)sbb2/conv3/w/Initializer/random_uniform:08
o
sbb2/conv3/b:0sbb2/conv3/b/Assign"sbb2/conv3/b/Read/ReadVariableOp:0(2 sbb2/conv3/b/Initializer/Const:08
x
sbb2/conv4/w:0sbb2/conv4/w/Assign"sbb2/conv4/w/Read/ReadVariableOp:0(2)sbb2/conv4/w/Initializer/random_uniform:08
o
sbb2/conv4/b:0sbb2/conv4/b/Assign"sbb2/conv4/b/Read/ReadVariableOp:0(2 sbb2/conv4/b/Initializer/Const:08
ś
 sbb2/batch_normalization/gamma:0%sbb2/batch_normalization/gamma/Assign4sbb2/batch_normalization/gamma/Read/ReadVariableOp:0(21sbb2/batch_normalization/gamma/Initializer/ones:08
ł
sbb2/batch_normalization/beta:0$sbb2/batch_normalization/beta/Assign3sbb2/batch_normalization/beta/Read/ReadVariableOp:0(21sbb2/batch_normalization/beta/Initializer/zeros:08
Ń
&sbb2/batch_normalization/moving_mean:0+sbb2/batch_normalization/moving_mean/Assign:sbb2/batch_normalization/moving_mean/Read/ReadVariableOp:0(28sbb2/batch_normalization/moving_mean/Initializer/zeros:0@H
ŕ
*sbb2/batch_normalization/moving_variance:0/sbb2/batch_normalization/moving_variance/Assign>sbb2/batch_normalization/moving_variance/Read/ReadVariableOp:0(2;sbb2/batch_normalization/moving_variance/Initializer/ones:0@H
x
sbb3/conv1/w:0sbb3/conv1/w/Assign"sbb3/conv1/w/Read/ReadVariableOp:0(2)sbb3/conv1/w/Initializer/random_uniform:08
o
sbb3/conv1/b:0sbb3/conv1/b/Assign"sbb3/conv1/b/Read/ReadVariableOp:0(2 sbb3/conv1/b/Initializer/Const:08
x
sbb3/conv2/w:0sbb3/conv2/w/Assign"sbb3/conv2/w/Read/ReadVariableOp:0(2)sbb3/conv2/w/Initializer/random_uniform:08
o
sbb3/conv2/b:0sbb3/conv2/b/Assign"sbb3/conv2/b/Read/ReadVariableOp:0(2 sbb3/conv2/b/Initializer/Const:08
x
sbb3/conv3/w:0sbb3/conv3/w/Assign"sbb3/conv3/w/Read/ReadVariableOp:0(2)sbb3/conv3/w/Initializer/random_uniform:08
o
sbb3/conv3/b:0sbb3/conv3/b/Assign"sbb3/conv3/b/Read/ReadVariableOp:0(2 sbb3/conv3/b/Initializer/Const:08
x
sbb3/conv4/w:0sbb3/conv4/w/Assign"sbb3/conv4/w/Read/ReadVariableOp:0(2)sbb3/conv4/w/Initializer/random_uniform:08
o
sbb3/conv4/b:0sbb3/conv4/b/Assign"sbb3/conv4/b/Read/ReadVariableOp:0(2 sbb3/conv4/b/Initializer/Const:08
ś
 sbb3/batch_normalization/gamma:0%sbb3/batch_normalization/gamma/Assign4sbb3/batch_normalization/gamma/Read/ReadVariableOp:0(21sbb3/batch_normalization/gamma/Initializer/ones:08
ł
sbb3/batch_normalization/beta:0$sbb3/batch_normalization/beta/Assign3sbb3/batch_normalization/beta/Read/ReadVariableOp:0(21sbb3/batch_normalization/beta/Initializer/zeros:08
Ń
&sbb3/batch_normalization/moving_mean:0+sbb3/batch_normalization/moving_mean/Assign:sbb3/batch_normalization/moving_mean/Read/ReadVariableOp:0(28sbb3/batch_normalization/moving_mean/Initializer/zeros:0@H
ŕ
*sbb3/batch_normalization/moving_variance:0/sbb3/batch_normalization/moving_variance/Assign>sbb3/batch_normalization/moving_variance/Read/ReadVariableOp:0(2;sbb3/batch_normalization/moving_variance/Initializer/ones:0@H
l
conv_d1/w:0conv_d1/w/Assignconv_d1/w/Read/ReadVariableOp:0(2&conv_d1/w/Initializer/random_uniform:08
c
conv_d1/b:0conv_d1/b/Assignconv_d1/b/Read/ReadVariableOp:0(2conv_d1/b/Initializer/Const:08
Ş
batch_normalization_1/gamma:0"batch_normalization_1/gamma/Assign1batch_normalization_1/gamma/Read/ReadVariableOp:0(2.batch_normalization_1/gamma/Initializer/ones:08
§
batch_normalization_1/beta:0!batch_normalization_1/beta/Assign0batch_normalization_1/beta/Read/ReadVariableOp:0(2.batch_normalization_1/beta/Initializer/zeros:08
Ĺ
#batch_normalization_1/moving_mean:0(batch_normalization_1/moving_mean/Assign7batch_normalization_1/moving_mean/Read/ReadVariableOp:0(25batch_normalization_1/moving_mean/Initializer/zeros:0@H
Ô
'batch_normalization_1/moving_variance:0,batch_normalization_1/moving_variance/Assign;batch_normalization_1/moving_variance/Read/ReadVariableOp:0(28batch_normalization_1/moving_variance/Initializer/ones:0@H
l
conv_d2/w:0conv_d2/w/Assignconv_d2/w/Read/ReadVariableOp:0(2&conv_d2/w/Initializer/random_uniform:08
c
conv_d2/b:0conv_d2/b/Assignconv_d2/b/Read/ReadVariableOp:0(2conv_d2/b/Initializer/Const:08
Ş
batch_normalization_2/gamma:0"batch_normalization_2/gamma/Assign1batch_normalization_2/gamma/Read/ReadVariableOp:0(2.batch_normalization_2/gamma/Initializer/ones:08
§
batch_normalization_2/beta:0!batch_normalization_2/beta/Assign0batch_normalization_2/beta/Read/ReadVariableOp:0(2.batch_normalization_2/beta/Initializer/zeros:08
Ĺ
#batch_normalization_2/moving_mean:0(batch_normalization_2/moving_mean/Assign7batch_normalization_2/moving_mean/Read/ReadVariableOp:0(25batch_normalization_2/moving_mean/Initializer/zeros:0@H
Ô
'batch_normalization_2/moving_variance:0,batch_normalization_2/moving_variance/Assign;batch_normalization_2/moving_variance/Read/ReadVariableOp:0(28batch_normalization_2/moving_variance/Initializer/ones:0@H
p
conv_out/w:0conv_out/w/Assign conv_out/w/Read/ReadVariableOp:0(2'conv_out/w/Initializer/random_uniform:08
g
conv_out/b:0conv_out/b/Assign conv_out/b/Read/ReadVariableOp:0(2conv_out/b/Initializer/Const:08
e

Variable:0Variable/AssignVariable/Read/ReadVariableOp:0(2$Variable/Initializer/initial_value:0
q
beta1_power:0beta1_power/Assign!beta1_power/Read/ReadVariableOp:0(2'beta1_power/Initializer/initial_value:0
q
beta2_power:0beta2_power/Assign!beta2_power/Read/ReadVariableOp:0(2'beta2_power/Initializer/initial_value:0
m
conv1/w/Adam:0conv1/w/Adam/Assign"conv1/w/Adam/Read/ReadVariableOp:0(2 conv1/w/Adam/Initializer/zeros:0
u
conv1/w/Adam_1:0conv1/w/Adam_1/Assign$conv1/w/Adam_1/Read/ReadVariableOp:0(2"conv1/w/Adam_1/Initializer/zeros:0
m
conv1/b/Adam:0conv1/b/Adam/Assign"conv1/b/Adam/Read/ReadVariableOp:0(2 conv1/b/Adam/Initializer/zeros:0
u
conv1/b/Adam_1:0conv1/b/Adam_1/Assign$conv1/b/Adam_1/Read/ReadVariableOp:0(2"conv1/b/Adam_1/Initializer/zeros:0
ľ
 batch_normalization/gamma/Adam:0%batch_normalization/gamma/Adam/Assign4batch_normalization/gamma/Adam/Read/ReadVariableOp:0(22batch_normalization/gamma/Adam/Initializer/zeros:0
˝
"batch_normalization/gamma/Adam_1:0'batch_normalization/gamma/Adam_1/Assign6batch_normalization/gamma/Adam_1/Read/ReadVariableOp:0(24batch_normalization/gamma/Adam_1/Initializer/zeros:0
ą
batch_normalization/beta/Adam:0$batch_normalization/beta/Adam/Assign3batch_normalization/beta/Adam/Read/ReadVariableOp:0(21batch_normalization/beta/Adam/Initializer/zeros:0
š
!batch_normalization/beta/Adam_1:0&batch_normalization/beta/Adam_1/Assign5batch_normalization/beta/Adam_1/Read/ReadVariableOp:0(23batch_normalization/beta/Adam_1/Initializer/zeros:0

sbb1/conv1/w/Adam:0sbb1/conv1/w/Adam/Assign'sbb1/conv1/w/Adam/Read/ReadVariableOp:0(2%sbb1/conv1/w/Adam/Initializer/zeros:0

sbb1/conv1/w/Adam_1:0sbb1/conv1/w/Adam_1/Assign)sbb1/conv1/w/Adam_1/Read/ReadVariableOp:0(2'sbb1/conv1/w/Adam_1/Initializer/zeros:0

sbb1/conv1/b/Adam:0sbb1/conv1/b/Adam/Assign'sbb1/conv1/b/Adam/Read/ReadVariableOp:0(2%sbb1/conv1/b/Adam/Initializer/zeros:0

sbb1/conv1/b/Adam_1:0sbb1/conv1/b/Adam_1/Assign)sbb1/conv1/b/Adam_1/Read/ReadVariableOp:0(2'sbb1/conv1/b/Adam_1/Initializer/zeros:0

sbb1/conv2/w/Adam:0sbb1/conv2/w/Adam/Assign'sbb1/conv2/w/Adam/Read/ReadVariableOp:0(2%sbb1/conv2/w/Adam/Initializer/zeros:0

sbb1/conv2/w/Adam_1:0sbb1/conv2/w/Adam_1/Assign)sbb1/conv2/w/Adam_1/Read/ReadVariableOp:0(2'sbb1/conv2/w/Adam_1/Initializer/zeros:0

sbb1/conv2/b/Adam:0sbb1/conv2/b/Adam/Assign'sbb1/conv2/b/Adam/Read/ReadVariableOp:0(2%sbb1/conv2/b/Adam/Initializer/zeros:0

sbb1/conv2/b/Adam_1:0sbb1/conv2/b/Adam_1/Assign)sbb1/conv2/b/Adam_1/Read/ReadVariableOp:0(2'sbb1/conv2/b/Adam_1/Initializer/zeros:0

sbb1/conv3/w/Adam:0sbb1/conv3/w/Adam/Assign'sbb1/conv3/w/Adam/Read/ReadVariableOp:0(2%sbb1/conv3/w/Adam/Initializer/zeros:0

sbb1/conv3/w/Adam_1:0sbb1/conv3/w/Adam_1/Assign)sbb1/conv3/w/Adam_1/Read/ReadVariableOp:0(2'sbb1/conv3/w/Adam_1/Initializer/zeros:0

sbb1/conv3/b/Adam:0sbb1/conv3/b/Adam/Assign'sbb1/conv3/b/Adam/Read/ReadVariableOp:0(2%sbb1/conv3/b/Adam/Initializer/zeros:0

sbb1/conv3/b/Adam_1:0sbb1/conv3/b/Adam_1/Assign)sbb1/conv3/b/Adam_1/Read/ReadVariableOp:0(2'sbb1/conv3/b/Adam_1/Initializer/zeros:0

sbb1/conv4/w/Adam:0sbb1/conv4/w/Adam/Assign'sbb1/conv4/w/Adam/Read/ReadVariableOp:0(2%sbb1/conv4/w/Adam/Initializer/zeros:0

sbb1/conv4/w/Adam_1:0sbb1/conv4/w/Adam_1/Assign)sbb1/conv4/w/Adam_1/Read/ReadVariableOp:0(2'sbb1/conv4/w/Adam_1/Initializer/zeros:0

sbb1/conv4/b/Adam:0sbb1/conv4/b/Adam/Assign'sbb1/conv4/b/Adam/Read/ReadVariableOp:0(2%sbb1/conv4/b/Adam/Initializer/zeros:0

sbb1/conv4/b/Adam_1:0sbb1/conv4/b/Adam_1/Assign)sbb1/conv4/b/Adam_1/Read/ReadVariableOp:0(2'sbb1/conv4/b/Adam_1/Initializer/zeros:0
É
%sbb1/batch_normalization/gamma/Adam:0*sbb1/batch_normalization/gamma/Adam/Assign9sbb1/batch_normalization/gamma/Adam/Read/ReadVariableOp:0(27sbb1/batch_normalization/gamma/Adam/Initializer/zeros:0
Ń
'sbb1/batch_normalization/gamma/Adam_1:0,sbb1/batch_normalization/gamma/Adam_1/Assign;sbb1/batch_normalization/gamma/Adam_1/Read/ReadVariableOp:0(29sbb1/batch_normalization/gamma/Adam_1/Initializer/zeros:0
Ĺ
$sbb1/batch_normalization/beta/Adam:0)sbb1/batch_normalization/beta/Adam/Assign8sbb1/batch_normalization/beta/Adam/Read/ReadVariableOp:0(26sbb1/batch_normalization/beta/Adam/Initializer/zeros:0
Í
&sbb1/batch_normalization/beta/Adam_1:0+sbb1/batch_normalization/beta/Adam_1/Assign:sbb1/batch_normalization/beta/Adam_1/Read/ReadVariableOp:0(28sbb1/batch_normalization/beta/Adam_1/Initializer/zeros:0

sbb2/conv1/w/Adam:0sbb2/conv1/w/Adam/Assign'sbb2/conv1/w/Adam/Read/ReadVariableOp:0(2%sbb2/conv1/w/Adam/Initializer/zeros:0

sbb2/conv1/w/Adam_1:0sbb2/conv1/w/Adam_1/Assign)sbb2/conv1/w/Adam_1/Read/ReadVariableOp:0(2'sbb2/conv1/w/Adam_1/Initializer/zeros:0

sbb2/conv1/b/Adam:0sbb2/conv1/b/Adam/Assign'sbb2/conv1/b/Adam/Read/ReadVariableOp:0(2%sbb2/conv1/b/Adam/Initializer/zeros:0

sbb2/conv1/b/Adam_1:0sbb2/conv1/b/Adam_1/Assign)sbb2/conv1/b/Adam_1/Read/ReadVariableOp:0(2'sbb2/conv1/b/Adam_1/Initializer/zeros:0

sbb2/conv2/w/Adam:0sbb2/conv2/w/Adam/Assign'sbb2/conv2/w/Adam/Read/ReadVariableOp:0(2%sbb2/conv2/w/Adam/Initializer/zeros:0

sbb2/conv2/w/Adam_1:0sbb2/conv2/w/Adam_1/Assign)sbb2/conv2/w/Adam_1/Read/ReadVariableOp:0(2'sbb2/conv2/w/Adam_1/Initializer/zeros:0

sbb2/conv2/b/Adam:0sbb2/conv2/b/Adam/Assign'sbb2/conv2/b/Adam/Read/ReadVariableOp:0(2%sbb2/conv2/b/Adam/Initializer/zeros:0

sbb2/conv2/b/Adam_1:0sbb2/conv2/b/Adam_1/Assign)sbb2/conv2/b/Adam_1/Read/ReadVariableOp:0(2'sbb2/conv2/b/Adam_1/Initializer/zeros:0

sbb2/conv3/w/Adam:0sbb2/conv3/w/Adam/Assign'sbb2/conv3/w/Adam/Read/ReadVariableOp:0(2%sbb2/conv3/w/Adam/Initializer/zeros:0

sbb2/conv3/w/Adam_1:0sbb2/conv3/w/Adam_1/Assign)sbb2/conv3/w/Adam_1/Read/ReadVariableOp:0(2'sbb2/conv3/w/Adam_1/Initializer/zeros:0

sbb2/conv3/b/Adam:0sbb2/conv3/b/Adam/Assign'sbb2/conv3/b/Adam/Read/ReadVariableOp:0(2%sbb2/conv3/b/Adam/Initializer/zeros:0

sbb2/conv3/b/Adam_1:0sbb2/conv3/b/Adam_1/Assign)sbb2/conv3/b/Adam_1/Read/ReadVariableOp:0(2'sbb2/conv3/b/Adam_1/Initializer/zeros:0

sbb2/conv4/w/Adam:0sbb2/conv4/w/Adam/Assign'sbb2/conv4/w/Adam/Read/ReadVariableOp:0(2%sbb2/conv4/w/Adam/Initializer/zeros:0

sbb2/conv4/w/Adam_1:0sbb2/conv4/w/Adam_1/Assign)sbb2/conv4/w/Adam_1/Read/ReadVariableOp:0(2'sbb2/conv4/w/Adam_1/Initializer/zeros:0

sbb2/conv4/b/Adam:0sbb2/conv4/b/Adam/Assign'sbb2/conv4/b/Adam/Read/ReadVariableOp:0(2%sbb2/conv4/b/Adam/Initializer/zeros:0

sbb2/conv4/b/Adam_1:0sbb2/conv4/b/Adam_1/Assign)sbb2/conv4/b/Adam_1/Read/ReadVariableOp:0(2'sbb2/conv4/b/Adam_1/Initializer/zeros:0
É
%sbb2/batch_normalization/gamma/Adam:0*sbb2/batch_normalization/gamma/Adam/Assign9sbb2/batch_normalization/gamma/Adam/Read/ReadVariableOp:0(27sbb2/batch_normalization/gamma/Adam/Initializer/zeros:0
Ń
'sbb2/batch_normalization/gamma/Adam_1:0,sbb2/batch_normalization/gamma/Adam_1/Assign;sbb2/batch_normalization/gamma/Adam_1/Read/ReadVariableOp:0(29sbb2/batch_normalization/gamma/Adam_1/Initializer/zeros:0
Ĺ
$sbb2/batch_normalization/beta/Adam:0)sbb2/batch_normalization/beta/Adam/Assign8sbb2/batch_normalization/beta/Adam/Read/ReadVariableOp:0(26sbb2/batch_normalization/beta/Adam/Initializer/zeros:0
Í
&sbb2/batch_normalization/beta/Adam_1:0+sbb2/batch_normalization/beta/Adam_1/Assign:sbb2/batch_normalization/beta/Adam_1/Read/ReadVariableOp:0(28sbb2/batch_normalization/beta/Adam_1/Initializer/zeros:0

sbb3/conv1/w/Adam:0sbb3/conv1/w/Adam/Assign'sbb3/conv1/w/Adam/Read/ReadVariableOp:0(2%sbb3/conv1/w/Adam/Initializer/zeros:0

sbb3/conv1/w/Adam_1:0sbb3/conv1/w/Adam_1/Assign)sbb3/conv1/w/Adam_1/Read/ReadVariableOp:0(2'sbb3/conv1/w/Adam_1/Initializer/zeros:0

sbb3/conv1/b/Adam:0sbb3/conv1/b/Adam/Assign'sbb3/conv1/b/Adam/Read/ReadVariableOp:0(2%sbb3/conv1/b/Adam/Initializer/zeros:0

sbb3/conv1/b/Adam_1:0sbb3/conv1/b/Adam_1/Assign)sbb3/conv1/b/Adam_1/Read/ReadVariableOp:0(2'sbb3/conv1/b/Adam_1/Initializer/zeros:0

sbb3/conv2/w/Adam:0sbb3/conv2/w/Adam/Assign'sbb3/conv2/w/Adam/Read/ReadVariableOp:0(2%sbb3/conv2/w/Adam/Initializer/zeros:0

sbb3/conv2/w/Adam_1:0sbb3/conv2/w/Adam_1/Assign)sbb3/conv2/w/Adam_1/Read/ReadVariableOp:0(2'sbb3/conv2/w/Adam_1/Initializer/zeros:0

sbb3/conv2/b/Adam:0sbb3/conv2/b/Adam/Assign'sbb3/conv2/b/Adam/Read/ReadVariableOp:0(2%sbb3/conv2/b/Adam/Initializer/zeros:0

sbb3/conv2/b/Adam_1:0sbb3/conv2/b/Adam_1/Assign)sbb3/conv2/b/Adam_1/Read/ReadVariableOp:0(2'sbb3/conv2/b/Adam_1/Initializer/zeros:0

sbb3/conv3/w/Adam:0sbb3/conv3/w/Adam/Assign'sbb3/conv3/w/Adam/Read/ReadVariableOp:0(2%sbb3/conv3/w/Adam/Initializer/zeros:0

sbb3/conv3/w/Adam_1:0sbb3/conv3/w/Adam_1/Assign)sbb3/conv3/w/Adam_1/Read/ReadVariableOp:0(2'sbb3/conv3/w/Adam_1/Initializer/zeros:0

sbb3/conv3/b/Adam:0sbb3/conv3/b/Adam/Assign'sbb3/conv3/b/Adam/Read/ReadVariableOp:0(2%sbb3/conv3/b/Adam/Initializer/zeros:0

sbb3/conv3/b/Adam_1:0sbb3/conv3/b/Adam_1/Assign)sbb3/conv3/b/Adam_1/Read/ReadVariableOp:0(2'sbb3/conv3/b/Adam_1/Initializer/zeros:0

sbb3/conv4/w/Adam:0sbb3/conv4/w/Adam/Assign'sbb3/conv4/w/Adam/Read/ReadVariableOp:0(2%sbb3/conv4/w/Adam/Initializer/zeros:0

sbb3/conv4/w/Adam_1:0sbb3/conv4/w/Adam_1/Assign)sbb3/conv4/w/Adam_1/Read/ReadVariableOp:0(2'sbb3/conv4/w/Adam_1/Initializer/zeros:0

sbb3/conv4/b/Adam:0sbb3/conv4/b/Adam/Assign'sbb3/conv4/b/Adam/Read/ReadVariableOp:0(2%sbb3/conv4/b/Adam/Initializer/zeros:0

sbb3/conv4/b/Adam_1:0sbb3/conv4/b/Adam_1/Assign)sbb3/conv4/b/Adam_1/Read/ReadVariableOp:0(2'sbb3/conv4/b/Adam_1/Initializer/zeros:0
É
%sbb3/batch_normalization/gamma/Adam:0*sbb3/batch_normalization/gamma/Adam/Assign9sbb3/batch_normalization/gamma/Adam/Read/ReadVariableOp:0(27sbb3/batch_normalization/gamma/Adam/Initializer/zeros:0
Ń
'sbb3/batch_normalization/gamma/Adam_1:0,sbb3/batch_normalization/gamma/Adam_1/Assign;sbb3/batch_normalization/gamma/Adam_1/Read/ReadVariableOp:0(29sbb3/batch_normalization/gamma/Adam_1/Initializer/zeros:0
Ĺ
$sbb3/batch_normalization/beta/Adam:0)sbb3/batch_normalization/beta/Adam/Assign8sbb3/batch_normalization/beta/Adam/Read/ReadVariableOp:0(26sbb3/batch_normalization/beta/Adam/Initializer/zeros:0
Í
&sbb3/batch_normalization/beta/Adam_1:0+sbb3/batch_normalization/beta/Adam_1/Assign:sbb3/batch_normalization/beta/Adam_1/Read/ReadVariableOp:0(28sbb3/batch_normalization/beta/Adam_1/Initializer/zeros:0
u
conv_d1/w/Adam:0conv_d1/w/Adam/Assign$conv_d1/w/Adam/Read/ReadVariableOp:0(2"conv_d1/w/Adam/Initializer/zeros:0
}
conv_d1/w/Adam_1:0conv_d1/w/Adam_1/Assign&conv_d1/w/Adam_1/Read/ReadVariableOp:0(2$conv_d1/w/Adam_1/Initializer/zeros:0
u
conv_d1/b/Adam:0conv_d1/b/Adam/Assign$conv_d1/b/Adam/Read/ReadVariableOp:0(2"conv_d1/b/Adam/Initializer/zeros:0
}
conv_d1/b/Adam_1:0conv_d1/b/Adam_1/Assign&conv_d1/b/Adam_1/Read/ReadVariableOp:0(2$conv_d1/b/Adam_1/Initializer/zeros:0
˝
"batch_normalization_1/gamma/Adam:0'batch_normalization_1/gamma/Adam/Assign6batch_normalization_1/gamma/Adam/Read/ReadVariableOp:0(24batch_normalization_1/gamma/Adam/Initializer/zeros:0
Ĺ
$batch_normalization_1/gamma/Adam_1:0)batch_normalization_1/gamma/Adam_1/Assign8batch_normalization_1/gamma/Adam_1/Read/ReadVariableOp:0(26batch_normalization_1/gamma/Adam_1/Initializer/zeros:0
š
!batch_normalization_1/beta/Adam:0&batch_normalization_1/beta/Adam/Assign5batch_normalization_1/beta/Adam/Read/ReadVariableOp:0(23batch_normalization_1/beta/Adam/Initializer/zeros:0
Á
#batch_normalization_1/beta/Adam_1:0(batch_normalization_1/beta/Adam_1/Assign7batch_normalization_1/beta/Adam_1/Read/ReadVariableOp:0(25batch_normalization_1/beta/Adam_1/Initializer/zeros:0
u
conv_d2/w/Adam:0conv_d2/w/Adam/Assign$conv_d2/w/Adam/Read/ReadVariableOp:0(2"conv_d2/w/Adam/Initializer/zeros:0
}
conv_d2/w/Adam_1:0conv_d2/w/Adam_1/Assign&conv_d2/w/Adam_1/Read/ReadVariableOp:0(2$conv_d2/w/Adam_1/Initializer/zeros:0
u
conv_d2/b/Adam:0conv_d2/b/Adam/Assign$conv_d2/b/Adam/Read/ReadVariableOp:0(2"conv_d2/b/Adam/Initializer/zeros:0
}
conv_d2/b/Adam_1:0conv_d2/b/Adam_1/Assign&conv_d2/b/Adam_1/Read/ReadVariableOp:0(2$conv_d2/b/Adam_1/Initializer/zeros:0
˝
"batch_normalization_2/gamma/Adam:0'batch_normalization_2/gamma/Adam/Assign6batch_normalization_2/gamma/Adam/Read/ReadVariableOp:0(24batch_normalization_2/gamma/Adam/Initializer/zeros:0
Ĺ
$batch_normalization_2/gamma/Adam_1:0)batch_normalization_2/gamma/Adam_1/Assign8batch_normalization_2/gamma/Adam_1/Read/ReadVariableOp:0(26batch_normalization_2/gamma/Adam_1/Initializer/zeros:0
š
!batch_normalization_2/beta/Adam:0&batch_normalization_2/beta/Adam/Assign5batch_normalization_2/beta/Adam/Read/ReadVariableOp:0(23batch_normalization_2/beta/Adam/Initializer/zeros:0
Á
#batch_normalization_2/beta/Adam_1:0(batch_normalization_2/beta/Adam_1/Assign7batch_normalization_2/beta/Adam_1/Read/ReadVariableOp:0(25batch_normalization_2/beta/Adam_1/Initializer/zeros:0
y
conv_out/w/Adam:0conv_out/w/Adam/Assign%conv_out/w/Adam/Read/ReadVariableOp:0(2#conv_out/w/Adam/Initializer/zeros:0

conv_out/w/Adam_1:0conv_out/w/Adam_1/Assign'conv_out/w/Adam_1/Read/ReadVariableOp:0(2%conv_out/w/Adam_1/Initializer/zeros:0
y
conv_out/b/Adam:0conv_out/b/Adam/Assign%conv_out/b/Adam/Read/ReadVariableOp:0(2#conv_out/b/Adam/Initializer/zeros:0

conv_out/b/Adam_1:0conv_out/b/Adam_1/Assign'conv_out/b/Adam_1/Read/ReadVariableOp:0(2%conv_out/b/Adam_1/Initializer/zeros:0*
classification
1
inputs'
inputs:0˙˙˙˙˙˙˙˙˙^4
outputs)
	decoded:0	˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙tensorflow/serving/predict