ҏ+
��
D
AddV2
x"T
y"T
z"T"
Ttype:
2	��
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
�
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
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
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
�
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
�
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
.
Identity

input"T
output"T"	
Ttype
�
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( �
?
Mul
x"T
y"T
z"T"
Ttype:
2	�
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
8
Pow
x"T
y"T
z"T"
Ttype:
2
	
@
ReadVariableOp
resource
value"dtype"
dtypetype�
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
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
H
ShardedFilename
basename	
shard

num_shards
filename
�
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
executor_typestring ��
@
StaticRegexFullMatch	
input

output
"
patternstring
�
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
-
Tanh
x"T
y"T"
Ttype:

2
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.10.12v2.10.0-76-gfdfc646704c8ϡ&
�
cond_1/Adam/conv2d_16/bias/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!cond_1/Adam/conv2d_16/bias/vhat
�
3cond_1/Adam/conv2d_16/bias/vhat/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv2d_16/bias/vhat*
_output_shapes
:*
dtype0
�
!cond_1/Adam/conv2d_16/kernel/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:<*2
shared_name#!cond_1/Adam/conv2d_16/kernel/vhat
�
5cond_1/Adam/conv2d_16/kernel/vhat/Read/ReadVariableOpReadVariableOp!cond_1/Adam/conv2d_16/kernel/vhat*&
_output_shapes
:<*
dtype0
�
cond_1/Adam/conv2d_15/bias/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:<*0
shared_name!cond_1/Adam/conv2d_15/bias/vhat
�
3cond_1/Adam/conv2d_15/bias/vhat/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv2d_15/bias/vhat*
_output_shapes
:<*
dtype0
�
!cond_1/Adam/conv2d_15/kernel/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:<<*2
shared_name#!cond_1/Adam/conv2d_15/kernel/vhat
�
5cond_1/Adam/conv2d_15/kernel/vhat/Read/ReadVariableOpReadVariableOp!cond_1/Adam/conv2d_15/kernel/vhat*&
_output_shapes
:<<*
dtype0
�
cond_1/Adam/conv2d_14/bias/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:<*0
shared_name!cond_1/Adam/conv2d_14/bias/vhat
�
3cond_1/Adam/conv2d_14/bias/vhat/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv2d_14/bias/vhat*
_output_shapes
:<*
dtype0
�
!cond_1/Adam/conv2d_14/kernel/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:<<*2
shared_name#!cond_1/Adam/conv2d_14/kernel/vhat
�
5cond_1/Adam/conv2d_14/kernel/vhat/Read/ReadVariableOpReadVariableOp!cond_1/Adam/conv2d_14/kernel/vhat*&
_output_shapes
:<<*
dtype0
�
(cond_1/Adam/conv2d_transpose_1/bias/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:<*9
shared_name*(cond_1/Adam/conv2d_transpose_1/bias/vhat
�
<cond_1/Adam/conv2d_transpose_1/bias/vhat/Read/ReadVariableOpReadVariableOp(cond_1/Adam/conv2d_transpose_1/bias/vhat*
_output_shapes
:<*
dtype0
�
*cond_1/Adam/conv2d_transpose_1/kernel/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:<x*;
shared_name,*cond_1/Adam/conv2d_transpose_1/kernel/vhat
�
>cond_1/Adam/conv2d_transpose_1/kernel/vhat/Read/ReadVariableOpReadVariableOp*cond_1/Adam/conv2d_transpose_1/kernel/vhat*&
_output_shapes
:<x*
dtype0
�
cond_1/Adam/conv2d_13/bias/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:x*0
shared_name!cond_1/Adam/conv2d_13/bias/vhat
�
3cond_1/Adam/conv2d_13/bias/vhat/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv2d_13/bias/vhat*
_output_shapes
:x*
dtype0
�
!cond_1/Adam/conv2d_13/kernel/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:xx*2
shared_name#!cond_1/Adam/conv2d_13/kernel/vhat
�
5cond_1/Adam/conv2d_13/kernel/vhat/Read/ReadVariableOpReadVariableOp!cond_1/Adam/conv2d_13/kernel/vhat*&
_output_shapes
:xx*
dtype0
�
cond_1/Adam/conv2d_12/bias/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:x*0
shared_name!cond_1/Adam/conv2d_12/bias/vhat
�
3cond_1/Adam/conv2d_12/bias/vhat/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv2d_12/bias/vhat*
_output_shapes
:x*
dtype0
�
!cond_1/Adam/conv2d_12/kernel/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:xx*2
shared_name#!cond_1/Adam/conv2d_12/kernel/vhat
�
5cond_1/Adam/conv2d_12/kernel/vhat/Read/ReadVariableOpReadVariableOp!cond_1/Adam/conv2d_12/kernel/vhat*&
_output_shapes
:xx*
dtype0
�
&cond_1/Adam/conv2d_transpose/bias/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:x*7
shared_name(&cond_1/Adam/conv2d_transpose/bias/vhat
�
:cond_1/Adam/conv2d_transpose/bias/vhat/Read/ReadVariableOpReadVariableOp&cond_1/Adam/conv2d_transpose/bias/vhat*
_output_shapes
:x*
dtype0
�
(cond_1/Adam/conv2d_transpose/kernel/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:x�*9
shared_name*(cond_1/Adam/conv2d_transpose/kernel/vhat
�
<cond_1/Adam/conv2d_transpose/kernel/vhat/Read/ReadVariableOpReadVariableOp(cond_1/Adam/conv2d_transpose/kernel/vhat*'
_output_shapes
:x�*
dtype0
�
cond_1/Adam/conv2d_11/bias/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*0
shared_name!cond_1/Adam/conv2d_11/bias/vhat
�
3cond_1/Adam/conv2d_11/bias/vhat/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv2d_11/bias/vhat*
_output_shapes	
:�*
dtype0
�
!cond_1/Adam/conv2d_11/kernel/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*2
shared_name#!cond_1/Adam/conv2d_11/kernel/vhat
�
5cond_1/Adam/conv2d_11/kernel/vhat/Read/ReadVariableOpReadVariableOp!cond_1/Adam/conv2d_11/kernel/vhat*(
_output_shapes
:��*
dtype0
�
cond_1/Adam/conv2d_10/bias/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*0
shared_name!cond_1/Adam/conv2d_10/bias/vhat
�
3cond_1/Adam/conv2d_10/bias/vhat/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv2d_10/bias/vhat*
_output_shapes	
:�*
dtype0
�
!cond_1/Adam/conv2d_10/kernel/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*2
shared_name#!cond_1/Adam/conv2d_10/kernel/vhat
�
5cond_1/Adam/conv2d_10/kernel/vhat/Read/ReadVariableOpReadVariableOp!cond_1/Adam/conv2d_10/kernel/vhat*(
_output_shapes
:��*
dtype0
�
cond_1/Adam/conv2d_9/bias/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*/
shared_name cond_1/Adam/conv2d_9/bias/vhat
�
2cond_1/Adam/conv2d_9/bias/vhat/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv2d_9/bias/vhat*
_output_shapes	
:�*
dtype0
�
 cond_1/Adam/conv2d_9/kernel/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*1
shared_name" cond_1/Adam/conv2d_9/kernel/vhat
�
4cond_1/Adam/conv2d_9/kernel/vhat/Read/ReadVariableOpReadVariableOp cond_1/Adam/conv2d_9/kernel/vhat*(
_output_shapes
:��*
dtype0
�
cond_1/Adam/conv2d_8/bias/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*/
shared_name cond_1/Adam/conv2d_8/bias/vhat
�
2cond_1/Adam/conv2d_8/bias/vhat/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv2d_8/bias/vhat*
_output_shapes	
:�*
dtype0
�
 cond_1/Adam/conv2d_8/kernel/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:x�*1
shared_name" cond_1/Adam/conv2d_8/kernel/vhat
�
4cond_1/Adam/conv2d_8/kernel/vhat/Read/ReadVariableOpReadVariableOp cond_1/Adam/conv2d_8/kernel/vhat*'
_output_shapes
:x�*
dtype0
�
cond_1/Adam/conv2d_7/bias/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:x*/
shared_name cond_1/Adam/conv2d_7/bias/vhat
�
2cond_1/Adam/conv2d_7/bias/vhat/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv2d_7/bias/vhat*
_output_shapes
:x*
dtype0
�
 cond_1/Adam/conv2d_7/kernel/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:xx*1
shared_name" cond_1/Adam/conv2d_7/kernel/vhat
�
4cond_1/Adam/conv2d_7/kernel/vhat/Read/ReadVariableOpReadVariableOp cond_1/Adam/conv2d_7/kernel/vhat*&
_output_shapes
:xx*
dtype0
�
cond_1/Adam/conv2d_6/bias/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:x*/
shared_name cond_1/Adam/conv2d_6/bias/vhat
�
2cond_1/Adam/conv2d_6/bias/vhat/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv2d_6/bias/vhat*
_output_shapes
:x*
dtype0
�
 cond_1/Adam/conv2d_6/kernel/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:<x*1
shared_name" cond_1/Adam/conv2d_6/kernel/vhat
�
4cond_1/Adam/conv2d_6/kernel/vhat/Read/ReadVariableOpReadVariableOp cond_1/Adam/conv2d_6/kernel/vhat*&
_output_shapes
:<x*
dtype0
�
cond_1/Adam/conv2d_5/bias/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:<*/
shared_name cond_1/Adam/conv2d_5/bias/vhat
�
2cond_1/Adam/conv2d_5/bias/vhat/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv2d_5/bias/vhat*
_output_shapes
:<*
dtype0
�
 cond_1/Adam/conv2d_5/kernel/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:<<*1
shared_name" cond_1/Adam/conv2d_5/kernel/vhat
�
4cond_1/Adam/conv2d_5/kernel/vhat/Read/ReadVariableOpReadVariableOp cond_1/Adam/conv2d_5/kernel/vhat*&
_output_shapes
:<<*
dtype0
�
cond_1/Adam/conv2d_4/bias/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:<*/
shared_name cond_1/Adam/conv2d_4/bias/vhat
�
2cond_1/Adam/conv2d_4/bias/vhat/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv2d_4/bias/vhat*
_output_shapes
:<*
dtype0
�
 cond_1/Adam/conv2d_4/kernel/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:<*1
shared_name" cond_1/Adam/conv2d_4/kernel/vhat
�
4cond_1/Adam/conv2d_4/kernel/vhat/Read/ReadVariableOpReadVariableOp cond_1/Adam/conv2d_4/kernel/vhat*&
_output_shapes
:<*
dtype0
�
cond_1/Adam/conv2d_3/bias/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name cond_1/Adam/conv2d_3/bias/vhat
�
2cond_1/Adam/conv2d_3/bias/vhat/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv2d_3/bias/vhat*
_output_shapes
:*
dtype0
�
 cond_1/Adam/conv2d_3/kernel/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" cond_1/Adam/conv2d_3/kernel/vhat
�
4cond_1/Adam/conv2d_3/kernel/vhat/Read/ReadVariableOpReadVariableOp cond_1/Adam/conv2d_3/kernel/vhat*&
_output_shapes
:*
dtype0
�
cond_1/Adam/conv2d_2/bias/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name cond_1/Adam/conv2d_2/bias/vhat
�
2cond_1/Adam/conv2d_2/bias/vhat/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv2d_2/bias/vhat*
_output_shapes
:*
dtype0
�
 cond_1/Adam/conv2d_2/kernel/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" cond_1/Adam/conv2d_2/kernel/vhat
�
4cond_1/Adam/conv2d_2/kernel/vhat/Read/ReadVariableOpReadVariableOp cond_1/Adam/conv2d_2/kernel/vhat*&
_output_shapes
:*
dtype0
�
cond_1/Adam/conv2d_1/bias/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name cond_1/Adam/conv2d_1/bias/vhat
�
2cond_1/Adam/conv2d_1/bias/vhat/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv2d_1/bias/vhat*
_output_shapes
:*
dtype0
�
 cond_1/Adam/conv2d_1/kernel/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" cond_1/Adam/conv2d_1/kernel/vhat
�
4cond_1/Adam/conv2d_1/kernel/vhat/Read/ReadVariableOpReadVariableOp cond_1/Adam/conv2d_1/kernel/vhat*&
_output_shapes
:*
dtype0
�
cond_1/Adam/conv2d/bias/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namecond_1/Adam/conv2d/bias/vhat
�
0cond_1/Adam/conv2d/bias/vhat/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv2d/bias/vhat*
_output_shapes
:*
dtype0
�
cond_1/Adam/conv2d/kernel/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name cond_1/Adam/conv2d/kernel/vhat
�
2cond_1/Adam/conv2d/kernel/vhat/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv2d/kernel/vhat*&
_output_shapes
:*
dtype0
�
cond_1/Adam/conv2d_16/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namecond_1/Adam/conv2d_16/bias/v
�
0cond_1/Adam/conv2d_16/bias/v/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv2d_16/bias/v*
_output_shapes
:*
dtype0
�
cond_1/Adam/conv2d_16/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:<*/
shared_name cond_1/Adam/conv2d_16/kernel/v
�
2cond_1/Adam/conv2d_16/kernel/v/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv2d_16/kernel/v*&
_output_shapes
:<*
dtype0
�
cond_1/Adam/conv2d_15/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:<*-
shared_namecond_1/Adam/conv2d_15/bias/v
�
0cond_1/Adam/conv2d_15/bias/v/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv2d_15/bias/v*
_output_shapes
:<*
dtype0
�
cond_1/Adam/conv2d_15/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:<<*/
shared_name cond_1/Adam/conv2d_15/kernel/v
�
2cond_1/Adam/conv2d_15/kernel/v/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv2d_15/kernel/v*&
_output_shapes
:<<*
dtype0
�
cond_1/Adam/conv2d_14/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:<*-
shared_namecond_1/Adam/conv2d_14/bias/v
�
0cond_1/Adam/conv2d_14/bias/v/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv2d_14/bias/v*
_output_shapes
:<*
dtype0
�
cond_1/Adam/conv2d_14/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:<<*/
shared_name cond_1/Adam/conv2d_14/kernel/v
�
2cond_1/Adam/conv2d_14/kernel/v/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv2d_14/kernel/v*&
_output_shapes
:<<*
dtype0
�
%cond_1/Adam/conv2d_transpose_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:<*6
shared_name'%cond_1/Adam/conv2d_transpose_1/bias/v
�
9cond_1/Adam/conv2d_transpose_1/bias/v/Read/ReadVariableOpReadVariableOp%cond_1/Adam/conv2d_transpose_1/bias/v*
_output_shapes
:<*
dtype0
�
'cond_1/Adam/conv2d_transpose_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:<x*8
shared_name)'cond_1/Adam/conv2d_transpose_1/kernel/v
�
;cond_1/Adam/conv2d_transpose_1/kernel/v/Read/ReadVariableOpReadVariableOp'cond_1/Adam/conv2d_transpose_1/kernel/v*&
_output_shapes
:<x*
dtype0
�
cond_1/Adam/conv2d_13/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:x*-
shared_namecond_1/Adam/conv2d_13/bias/v
�
0cond_1/Adam/conv2d_13/bias/v/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv2d_13/bias/v*
_output_shapes
:x*
dtype0
�
cond_1/Adam/conv2d_13/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:xx*/
shared_name cond_1/Adam/conv2d_13/kernel/v
�
2cond_1/Adam/conv2d_13/kernel/v/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv2d_13/kernel/v*&
_output_shapes
:xx*
dtype0
�
cond_1/Adam/conv2d_12/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:x*-
shared_namecond_1/Adam/conv2d_12/bias/v
�
0cond_1/Adam/conv2d_12/bias/v/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv2d_12/bias/v*
_output_shapes
:x*
dtype0
�
cond_1/Adam/conv2d_12/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:xx*/
shared_name cond_1/Adam/conv2d_12/kernel/v
�
2cond_1/Adam/conv2d_12/kernel/v/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv2d_12/kernel/v*&
_output_shapes
:xx*
dtype0
�
#cond_1/Adam/conv2d_transpose/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:x*4
shared_name%#cond_1/Adam/conv2d_transpose/bias/v
�
7cond_1/Adam/conv2d_transpose/bias/v/Read/ReadVariableOpReadVariableOp#cond_1/Adam/conv2d_transpose/bias/v*
_output_shapes
:x*
dtype0
�
%cond_1/Adam/conv2d_transpose/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:x�*6
shared_name'%cond_1/Adam/conv2d_transpose/kernel/v
�
9cond_1/Adam/conv2d_transpose/kernel/v/Read/ReadVariableOpReadVariableOp%cond_1/Adam/conv2d_transpose/kernel/v*'
_output_shapes
:x�*
dtype0
�
cond_1/Adam/conv2d_11/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*-
shared_namecond_1/Adam/conv2d_11/bias/v
�
0cond_1/Adam/conv2d_11/bias/v/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv2d_11/bias/v*
_output_shapes	
:�*
dtype0
�
cond_1/Adam/conv2d_11/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*/
shared_name cond_1/Adam/conv2d_11/kernel/v
�
2cond_1/Adam/conv2d_11/kernel/v/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv2d_11/kernel/v*(
_output_shapes
:��*
dtype0
�
cond_1/Adam/conv2d_10/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*-
shared_namecond_1/Adam/conv2d_10/bias/v
�
0cond_1/Adam/conv2d_10/bias/v/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv2d_10/bias/v*
_output_shapes	
:�*
dtype0
�
cond_1/Adam/conv2d_10/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*/
shared_name cond_1/Adam/conv2d_10/kernel/v
�
2cond_1/Adam/conv2d_10/kernel/v/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv2d_10/kernel/v*(
_output_shapes
:��*
dtype0
�
cond_1/Adam/conv2d_9/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*,
shared_namecond_1/Adam/conv2d_9/bias/v
�
/cond_1/Adam/conv2d_9/bias/v/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv2d_9/bias/v*
_output_shapes	
:�*
dtype0
�
cond_1/Adam/conv2d_9/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*.
shared_namecond_1/Adam/conv2d_9/kernel/v
�
1cond_1/Adam/conv2d_9/kernel/v/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv2d_9/kernel/v*(
_output_shapes
:��*
dtype0
�
cond_1/Adam/conv2d_8/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*,
shared_namecond_1/Adam/conv2d_8/bias/v
�
/cond_1/Adam/conv2d_8/bias/v/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv2d_8/bias/v*
_output_shapes	
:�*
dtype0
�
cond_1/Adam/conv2d_8/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:x�*.
shared_namecond_1/Adam/conv2d_8/kernel/v
�
1cond_1/Adam/conv2d_8/kernel/v/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv2d_8/kernel/v*'
_output_shapes
:x�*
dtype0
�
cond_1/Adam/conv2d_7/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:x*,
shared_namecond_1/Adam/conv2d_7/bias/v
�
/cond_1/Adam/conv2d_7/bias/v/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv2d_7/bias/v*
_output_shapes
:x*
dtype0
�
cond_1/Adam/conv2d_7/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:xx*.
shared_namecond_1/Adam/conv2d_7/kernel/v
�
1cond_1/Adam/conv2d_7/kernel/v/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv2d_7/kernel/v*&
_output_shapes
:xx*
dtype0
�
cond_1/Adam/conv2d_6/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:x*,
shared_namecond_1/Adam/conv2d_6/bias/v
�
/cond_1/Adam/conv2d_6/bias/v/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv2d_6/bias/v*
_output_shapes
:x*
dtype0
�
cond_1/Adam/conv2d_6/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:<x*.
shared_namecond_1/Adam/conv2d_6/kernel/v
�
1cond_1/Adam/conv2d_6/kernel/v/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv2d_6/kernel/v*&
_output_shapes
:<x*
dtype0
�
cond_1/Adam/conv2d_5/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:<*,
shared_namecond_1/Adam/conv2d_5/bias/v
�
/cond_1/Adam/conv2d_5/bias/v/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv2d_5/bias/v*
_output_shapes
:<*
dtype0
�
cond_1/Adam/conv2d_5/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:<<*.
shared_namecond_1/Adam/conv2d_5/kernel/v
�
1cond_1/Adam/conv2d_5/kernel/v/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv2d_5/kernel/v*&
_output_shapes
:<<*
dtype0
�
cond_1/Adam/conv2d_4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:<*,
shared_namecond_1/Adam/conv2d_4/bias/v
�
/cond_1/Adam/conv2d_4/bias/v/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv2d_4/bias/v*
_output_shapes
:<*
dtype0
�
cond_1/Adam/conv2d_4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:<*.
shared_namecond_1/Adam/conv2d_4/kernel/v
�
1cond_1/Adam/conv2d_4/kernel/v/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv2d_4/kernel/v*&
_output_shapes
:<*
dtype0
�
cond_1/Adam/conv2d_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namecond_1/Adam/conv2d_3/bias/v
�
/cond_1/Adam/conv2d_3/bias/v/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv2d_3/bias/v*
_output_shapes
:*
dtype0
�
cond_1/Adam/conv2d_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_namecond_1/Adam/conv2d_3/kernel/v
�
1cond_1/Adam/conv2d_3/kernel/v/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv2d_3/kernel/v*&
_output_shapes
:*
dtype0
�
cond_1/Adam/conv2d_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namecond_1/Adam/conv2d_2/bias/v
�
/cond_1/Adam/conv2d_2/bias/v/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv2d_2/bias/v*
_output_shapes
:*
dtype0
�
cond_1/Adam/conv2d_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_namecond_1/Adam/conv2d_2/kernel/v
�
1cond_1/Adam/conv2d_2/kernel/v/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv2d_2/kernel/v*&
_output_shapes
:*
dtype0
�
cond_1/Adam/conv2d_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namecond_1/Adam/conv2d_1/bias/v
�
/cond_1/Adam/conv2d_1/bias/v/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv2d_1/bias/v*
_output_shapes
:*
dtype0
�
cond_1/Adam/conv2d_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_namecond_1/Adam/conv2d_1/kernel/v
�
1cond_1/Adam/conv2d_1/kernel/v/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv2d_1/kernel/v*&
_output_shapes
:*
dtype0
�
cond_1/Adam/conv2d/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_namecond_1/Adam/conv2d/bias/v
�
-cond_1/Adam/conv2d/bias/v/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv2d/bias/v*
_output_shapes
:*
dtype0
�
cond_1/Adam/conv2d/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namecond_1/Adam/conv2d/kernel/v
�
/cond_1/Adam/conv2d/kernel/v/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv2d/kernel/v*&
_output_shapes
:*
dtype0
�
cond_1/Adam/conv2d_16/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namecond_1/Adam/conv2d_16/bias/m
�
0cond_1/Adam/conv2d_16/bias/m/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv2d_16/bias/m*
_output_shapes
:*
dtype0
�
cond_1/Adam/conv2d_16/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:<*/
shared_name cond_1/Adam/conv2d_16/kernel/m
�
2cond_1/Adam/conv2d_16/kernel/m/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv2d_16/kernel/m*&
_output_shapes
:<*
dtype0
�
cond_1/Adam/conv2d_15/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:<*-
shared_namecond_1/Adam/conv2d_15/bias/m
�
0cond_1/Adam/conv2d_15/bias/m/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv2d_15/bias/m*
_output_shapes
:<*
dtype0
�
cond_1/Adam/conv2d_15/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:<<*/
shared_name cond_1/Adam/conv2d_15/kernel/m
�
2cond_1/Adam/conv2d_15/kernel/m/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv2d_15/kernel/m*&
_output_shapes
:<<*
dtype0
�
cond_1/Adam/conv2d_14/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:<*-
shared_namecond_1/Adam/conv2d_14/bias/m
�
0cond_1/Adam/conv2d_14/bias/m/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv2d_14/bias/m*
_output_shapes
:<*
dtype0
�
cond_1/Adam/conv2d_14/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:<<*/
shared_name cond_1/Adam/conv2d_14/kernel/m
�
2cond_1/Adam/conv2d_14/kernel/m/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv2d_14/kernel/m*&
_output_shapes
:<<*
dtype0
�
%cond_1/Adam/conv2d_transpose_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:<*6
shared_name'%cond_1/Adam/conv2d_transpose_1/bias/m
�
9cond_1/Adam/conv2d_transpose_1/bias/m/Read/ReadVariableOpReadVariableOp%cond_1/Adam/conv2d_transpose_1/bias/m*
_output_shapes
:<*
dtype0
�
'cond_1/Adam/conv2d_transpose_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:<x*8
shared_name)'cond_1/Adam/conv2d_transpose_1/kernel/m
�
;cond_1/Adam/conv2d_transpose_1/kernel/m/Read/ReadVariableOpReadVariableOp'cond_1/Adam/conv2d_transpose_1/kernel/m*&
_output_shapes
:<x*
dtype0
�
cond_1/Adam/conv2d_13/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:x*-
shared_namecond_1/Adam/conv2d_13/bias/m
�
0cond_1/Adam/conv2d_13/bias/m/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv2d_13/bias/m*
_output_shapes
:x*
dtype0
�
cond_1/Adam/conv2d_13/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:xx*/
shared_name cond_1/Adam/conv2d_13/kernel/m
�
2cond_1/Adam/conv2d_13/kernel/m/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv2d_13/kernel/m*&
_output_shapes
:xx*
dtype0
�
cond_1/Adam/conv2d_12/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:x*-
shared_namecond_1/Adam/conv2d_12/bias/m
�
0cond_1/Adam/conv2d_12/bias/m/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv2d_12/bias/m*
_output_shapes
:x*
dtype0
�
cond_1/Adam/conv2d_12/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:xx*/
shared_name cond_1/Adam/conv2d_12/kernel/m
�
2cond_1/Adam/conv2d_12/kernel/m/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv2d_12/kernel/m*&
_output_shapes
:xx*
dtype0
�
#cond_1/Adam/conv2d_transpose/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:x*4
shared_name%#cond_1/Adam/conv2d_transpose/bias/m
�
7cond_1/Adam/conv2d_transpose/bias/m/Read/ReadVariableOpReadVariableOp#cond_1/Adam/conv2d_transpose/bias/m*
_output_shapes
:x*
dtype0
�
%cond_1/Adam/conv2d_transpose/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:x�*6
shared_name'%cond_1/Adam/conv2d_transpose/kernel/m
�
9cond_1/Adam/conv2d_transpose/kernel/m/Read/ReadVariableOpReadVariableOp%cond_1/Adam/conv2d_transpose/kernel/m*'
_output_shapes
:x�*
dtype0
�
cond_1/Adam/conv2d_11/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*-
shared_namecond_1/Adam/conv2d_11/bias/m
�
0cond_1/Adam/conv2d_11/bias/m/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv2d_11/bias/m*
_output_shapes	
:�*
dtype0
�
cond_1/Adam/conv2d_11/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*/
shared_name cond_1/Adam/conv2d_11/kernel/m
�
2cond_1/Adam/conv2d_11/kernel/m/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv2d_11/kernel/m*(
_output_shapes
:��*
dtype0
�
cond_1/Adam/conv2d_10/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*-
shared_namecond_1/Adam/conv2d_10/bias/m
�
0cond_1/Adam/conv2d_10/bias/m/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv2d_10/bias/m*
_output_shapes	
:�*
dtype0
�
cond_1/Adam/conv2d_10/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*/
shared_name cond_1/Adam/conv2d_10/kernel/m
�
2cond_1/Adam/conv2d_10/kernel/m/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv2d_10/kernel/m*(
_output_shapes
:��*
dtype0
�
cond_1/Adam/conv2d_9/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*,
shared_namecond_1/Adam/conv2d_9/bias/m
�
/cond_1/Adam/conv2d_9/bias/m/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv2d_9/bias/m*
_output_shapes	
:�*
dtype0
�
cond_1/Adam/conv2d_9/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*.
shared_namecond_1/Adam/conv2d_9/kernel/m
�
1cond_1/Adam/conv2d_9/kernel/m/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv2d_9/kernel/m*(
_output_shapes
:��*
dtype0
�
cond_1/Adam/conv2d_8/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*,
shared_namecond_1/Adam/conv2d_8/bias/m
�
/cond_1/Adam/conv2d_8/bias/m/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv2d_8/bias/m*
_output_shapes	
:�*
dtype0
�
cond_1/Adam/conv2d_8/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:x�*.
shared_namecond_1/Adam/conv2d_8/kernel/m
�
1cond_1/Adam/conv2d_8/kernel/m/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv2d_8/kernel/m*'
_output_shapes
:x�*
dtype0
�
cond_1/Adam/conv2d_7/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:x*,
shared_namecond_1/Adam/conv2d_7/bias/m
�
/cond_1/Adam/conv2d_7/bias/m/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv2d_7/bias/m*
_output_shapes
:x*
dtype0
�
cond_1/Adam/conv2d_7/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:xx*.
shared_namecond_1/Adam/conv2d_7/kernel/m
�
1cond_1/Adam/conv2d_7/kernel/m/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv2d_7/kernel/m*&
_output_shapes
:xx*
dtype0
�
cond_1/Adam/conv2d_6/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:x*,
shared_namecond_1/Adam/conv2d_6/bias/m
�
/cond_1/Adam/conv2d_6/bias/m/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv2d_6/bias/m*
_output_shapes
:x*
dtype0
�
cond_1/Adam/conv2d_6/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:<x*.
shared_namecond_1/Adam/conv2d_6/kernel/m
�
1cond_1/Adam/conv2d_6/kernel/m/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv2d_6/kernel/m*&
_output_shapes
:<x*
dtype0
�
cond_1/Adam/conv2d_5/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:<*,
shared_namecond_1/Adam/conv2d_5/bias/m
�
/cond_1/Adam/conv2d_5/bias/m/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv2d_5/bias/m*
_output_shapes
:<*
dtype0
�
cond_1/Adam/conv2d_5/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:<<*.
shared_namecond_1/Adam/conv2d_5/kernel/m
�
1cond_1/Adam/conv2d_5/kernel/m/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv2d_5/kernel/m*&
_output_shapes
:<<*
dtype0
�
cond_1/Adam/conv2d_4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:<*,
shared_namecond_1/Adam/conv2d_4/bias/m
�
/cond_1/Adam/conv2d_4/bias/m/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv2d_4/bias/m*
_output_shapes
:<*
dtype0
�
cond_1/Adam/conv2d_4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:<*.
shared_namecond_1/Adam/conv2d_4/kernel/m
�
1cond_1/Adam/conv2d_4/kernel/m/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv2d_4/kernel/m*&
_output_shapes
:<*
dtype0
�
cond_1/Adam/conv2d_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namecond_1/Adam/conv2d_3/bias/m
�
/cond_1/Adam/conv2d_3/bias/m/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv2d_3/bias/m*
_output_shapes
:*
dtype0
�
cond_1/Adam/conv2d_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_namecond_1/Adam/conv2d_3/kernel/m
�
1cond_1/Adam/conv2d_3/kernel/m/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv2d_3/kernel/m*&
_output_shapes
:*
dtype0
�
cond_1/Adam/conv2d_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namecond_1/Adam/conv2d_2/bias/m
�
/cond_1/Adam/conv2d_2/bias/m/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv2d_2/bias/m*
_output_shapes
:*
dtype0
�
cond_1/Adam/conv2d_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_namecond_1/Adam/conv2d_2/kernel/m
�
1cond_1/Adam/conv2d_2/kernel/m/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv2d_2/kernel/m*&
_output_shapes
:*
dtype0
�
cond_1/Adam/conv2d_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namecond_1/Adam/conv2d_1/bias/m
�
/cond_1/Adam/conv2d_1/bias/m/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv2d_1/bias/m*
_output_shapes
:*
dtype0
�
cond_1/Adam/conv2d_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_namecond_1/Adam/conv2d_1/kernel/m
�
1cond_1/Adam/conv2d_1/kernel/m/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv2d_1/kernel/m*&
_output_shapes
:*
dtype0
�
cond_1/Adam/conv2d/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_namecond_1/Adam/conv2d/bias/m
�
-cond_1/Adam/conv2d/bias/m/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv2d/bias/m*
_output_shapes
:*
dtype0
�
cond_1/Adam/conv2d/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namecond_1/Adam/conv2d/kernel/m
�
/cond_1/Adam/conv2d/kernel/m/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv2d/kernel/m*&
_output_shapes
:*
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
count_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_2
[
count_2/Read/ReadVariableOpReadVariableOpcount_2*
_output_shapes
: *
dtype0
b
total_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_2
[
total_2/Read/ReadVariableOpReadVariableOptotal_2*
_output_shapes
: *
dtype0
h

good_stepsVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name
good_steps
a
good_steps/Read/ReadVariableOpReadVariableOp
good_steps*
_output_shapes
: *
dtype0	
x
current_loss_scaleVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_namecurrent_loss_scale
q
&current_loss_scale/Read/ReadVariableOpReadVariableOpcurrent_loss_scale*
_output_shapes
: *
dtype0
�
cond_1/Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: **
shared_namecond_1/Adam/learning_rate

-cond_1/Adam/learning_rate/Read/ReadVariableOpReadVariableOpcond_1/Adam/learning_rate*
_output_shapes
: *
dtype0
v
cond_1/Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_namecond_1/Adam/decay
o
%cond_1/Adam/decay/Read/ReadVariableOpReadVariableOpcond_1/Adam/decay*
_output_shapes
: *
dtype0
x
cond_1/Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_namecond_1/Adam/beta_2
q
&cond_1/Adam/beta_2/Read/ReadVariableOpReadVariableOpcond_1/Adam/beta_2*
_output_shapes
: *
dtype0
x
cond_1/Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_namecond_1/Adam/beta_1
q
&cond_1/Adam/beta_1/Read/ReadVariableOpReadVariableOpcond_1/Adam/beta_1*
_output_shapes
: *
dtype0
t
cond_1/Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *!
shared_namecond_1/Adam/iter
m
$cond_1/Adam/iter/Read/ReadVariableOpReadVariableOpcond_1/Adam/iter*
_output_shapes
: *
dtype0	
t
conv2d_16/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_16/bias
m
"conv2d_16/bias/Read/ReadVariableOpReadVariableOpconv2d_16/bias*
_output_shapes
:*
dtype0
�
conv2d_16/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:<*!
shared_nameconv2d_16/kernel
}
$conv2d_16/kernel/Read/ReadVariableOpReadVariableOpconv2d_16/kernel*&
_output_shapes
:<*
dtype0
t
conv2d_15/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:<*
shared_nameconv2d_15/bias
m
"conv2d_15/bias/Read/ReadVariableOpReadVariableOpconv2d_15/bias*
_output_shapes
:<*
dtype0
�
conv2d_15/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:<<*!
shared_nameconv2d_15/kernel
}
$conv2d_15/kernel/Read/ReadVariableOpReadVariableOpconv2d_15/kernel*&
_output_shapes
:<<*
dtype0
t
conv2d_14/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:<*
shared_nameconv2d_14/bias
m
"conv2d_14/bias/Read/ReadVariableOpReadVariableOpconv2d_14/bias*
_output_shapes
:<*
dtype0
�
conv2d_14/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:<<*!
shared_nameconv2d_14/kernel
}
$conv2d_14/kernel/Read/ReadVariableOpReadVariableOpconv2d_14/kernel*&
_output_shapes
:<<*
dtype0
�
conv2d_transpose_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:<*(
shared_nameconv2d_transpose_1/bias

+conv2d_transpose_1/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_1/bias*
_output_shapes
:<*
dtype0
�
conv2d_transpose_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:<x**
shared_nameconv2d_transpose_1/kernel
�
-conv2d_transpose_1/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_1/kernel*&
_output_shapes
:<x*
dtype0
t
conv2d_13/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:x*
shared_nameconv2d_13/bias
m
"conv2d_13/bias/Read/ReadVariableOpReadVariableOpconv2d_13/bias*
_output_shapes
:x*
dtype0
�
conv2d_13/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:xx*!
shared_nameconv2d_13/kernel
}
$conv2d_13/kernel/Read/ReadVariableOpReadVariableOpconv2d_13/kernel*&
_output_shapes
:xx*
dtype0
t
conv2d_12/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:x*
shared_nameconv2d_12/bias
m
"conv2d_12/bias/Read/ReadVariableOpReadVariableOpconv2d_12/bias*
_output_shapes
:x*
dtype0
�
conv2d_12/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:xx*!
shared_nameconv2d_12/kernel
}
$conv2d_12/kernel/Read/ReadVariableOpReadVariableOpconv2d_12/kernel*&
_output_shapes
:xx*
dtype0
�
conv2d_transpose/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:x*&
shared_nameconv2d_transpose/bias
{
)conv2d_transpose/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose/bias*
_output_shapes
:x*
dtype0
�
conv2d_transpose/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:x�*(
shared_nameconv2d_transpose/kernel
�
+conv2d_transpose/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose/kernel*'
_output_shapes
:x�*
dtype0
u
conv2d_11/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_nameconv2d_11/bias
n
"conv2d_11/bias/Read/ReadVariableOpReadVariableOpconv2d_11/bias*
_output_shapes	
:�*
dtype0
�
conv2d_11/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*!
shared_nameconv2d_11/kernel

$conv2d_11/kernel/Read/ReadVariableOpReadVariableOpconv2d_11/kernel*(
_output_shapes
:��*
dtype0
u
conv2d_10/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_nameconv2d_10/bias
n
"conv2d_10/bias/Read/ReadVariableOpReadVariableOpconv2d_10/bias*
_output_shapes	
:�*
dtype0
�
conv2d_10/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*!
shared_nameconv2d_10/kernel

$conv2d_10/kernel/Read/ReadVariableOpReadVariableOpconv2d_10/kernel*(
_output_shapes
:��*
dtype0
s
conv2d_9/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_nameconv2d_9/bias
l
!conv2d_9/bias/Read/ReadVariableOpReadVariableOpconv2d_9/bias*
_output_shapes	
:�*
dtype0
�
conv2d_9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��* 
shared_nameconv2d_9/kernel
}
#conv2d_9/kernel/Read/ReadVariableOpReadVariableOpconv2d_9/kernel*(
_output_shapes
:��*
dtype0
s
conv2d_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_nameconv2d_8/bias
l
!conv2d_8/bias/Read/ReadVariableOpReadVariableOpconv2d_8/bias*
_output_shapes	
:�*
dtype0
�
conv2d_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:x�* 
shared_nameconv2d_8/kernel
|
#conv2d_8/kernel/Read/ReadVariableOpReadVariableOpconv2d_8/kernel*'
_output_shapes
:x�*
dtype0
r
conv2d_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:x*
shared_nameconv2d_7/bias
k
!conv2d_7/bias/Read/ReadVariableOpReadVariableOpconv2d_7/bias*
_output_shapes
:x*
dtype0
�
conv2d_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:xx* 
shared_nameconv2d_7/kernel
{
#conv2d_7/kernel/Read/ReadVariableOpReadVariableOpconv2d_7/kernel*&
_output_shapes
:xx*
dtype0
r
conv2d_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:x*
shared_nameconv2d_6/bias
k
!conv2d_6/bias/Read/ReadVariableOpReadVariableOpconv2d_6/bias*
_output_shapes
:x*
dtype0
�
conv2d_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:<x* 
shared_nameconv2d_6/kernel
{
#conv2d_6/kernel/Read/ReadVariableOpReadVariableOpconv2d_6/kernel*&
_output_shapes
:<x*
dtype0
r
conv2d_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:<*
shared_nameconv2d_5/bias
k
!conv2d_5/bias/Read/ReadVariableOpReadVariableOpconv2d_5/bias*
_output_shapes
:<*
dtype0
�
conv2d_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:<<* 
shared_nameconv2d_5/kernel
{
#conv2d_5/kernel/Read/ReadVariableOpReadVariableOpconv2d_5/kernel*&
_output_shapes
:<<*
dtype0
r
conv2d_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:<*
shared_nameconv2d_4/bias
k
!conv2d_4/bias/Read/ReadVariableOpReadVariableOpconv2d_4/bias*
_output_shapes
:<*
dtype0
�
conv2d_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:<* 
shared_nameconv2d_4/kernel
{
#conv2d_4/kernel/Read/ReadVariableOpReadVariableOpconv2d_4/kernel*&
_output_shapes
:<*
dtype0
r
conv2d_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_3/bias
k
!conv2d_3/bias/Read/ReadVariableOpReadVariableOpconv2d_3/bias*
_output_shapes
:*
dtype0
�
conv2d_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_3/kernel
{
#conv2d_3/kernel/Read/ReadVariableOpReadVariableOpconv2d_3/kernel*&
_output_shapes
:*
dtype0
r
conv2d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_2/bias
k
!conv2d_2/bias/Read/ReadVariableOpReadVariableOpconv2d_2/bias*
_output_shapes
:*
dtype0
�
conv2d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_2/kernel
{
#conv2d_2/kernel/Read/ReadVariableOpReadVariableOpconv2d_2/kernel*&
_output_shapes
:*
dtype0
r
conv2d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_1/bias
k
!conv2d_1/bias/Read/ReadVariableOpReadVariableOpconv2d_1/bias*
_output_shapes
:*
dtype0
�
conv2d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_1/kernel
{
#conv2d_1/kernel/Read/ReadVariableOpReadVariableOpconv2d_1/kernel*&
_output_shapes
:*
dtype0
n
conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d/bias
g
conv2d/bias/Read/ReadVariableOpReadVariableOpconv2d/bias*
_output_shapes
:*
dtype0
~
conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d/kernel
w
!conv2d/kernel/Read/ReadVariableOpReadVariableOpconv2d/kernel*&
_output_shapes
:*
dtype0
�
serving_default_input_1Placeholder*1
_output_shapes
:�����������*
dtype0*&
shape:�����������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1conv2d/kernelconv2d/biasconv2d_1/kernelconv2d_1/biasconv2d_2/kernelconv2d_2/biasconv2d_3/kernelconv2d_3/biasconv2d_4/kernelconv2d_4/biasconv2d_5/kernelconv2d_5/biasconv2d_6/kernelconv2d_6/biasconv2d_7/kernelconv2d_7/biasconv2d_8/kernelconv2d_8/biasconv2d_9/kernelconv2d_9/biasconv2d_10/kernelconv2d_10/biasconv2d_11/kernelconv2d_11/biasconv2d_transpose/kernelconv2d_transpose/biasconv2d_12/kernelconv2d_12/biasconv2d_13/kernelconv2d_13/biasconv2d_transpose_1/kernelconv2d_transpose_1/biasconv2d_14/kernelconv2d_14/biasconv2d_15/kernelconv2d_15/biasconv2d_16/kernelconv2d_16/bias*2
Tin+
)2'*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������*H
_read_only_resource_inputs*
(&	
 !"#$%&*0
config_proto 

CPU

GPU2*0J 8� *,
f'R%
#__inference_signature_wrapper_54656

NoOpNoOp
��
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�
value�B߸ B׸
�
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer-5
layer_with_weights-4
layer-6
layer_with_weights-5
layer-7
	layer-8

layer_with_weights-6

layer-9
layer_with_weights-7
layer-10
layer-11
layer_with_weights-8
layer-12
layer_with_weights-9
layer-13
layer_with_weights-10
layer-14
layer_with_weights-11
layer-15
layer_with_weights-12
layer-16
layer-17
layer_with_weights-13
layer-18
layer_with_weights-14
layer-19
layer_with_weights-15
layer-20
layer-21
layer_with_weights-16
layer-22
layer_with_weights-17
layer-23
layer_with_weights-18
layer-24
layer-25
	variables
trainable_variables
regularization_losses
	keras_api
__call__
* &call_and_return_all_conditional_losses
!_default_save_signature
"	optimizer
#
signatures*
* 
�
$	variables
%trainable_variables
&regularization_losses
'	keras_api
(__call__
*)&call_and_return_all_conditional_losses

*kernel
+bias
 ,_jit_compiled_convolution_op*
�
-	variables
.trainable_variables
/regularization_losses
0	keras_api
1__call__
*2&call_and_return_all_conditional_losses

3kernel
4bias
 5_jit_compiled_convolution_op*
�
6	variables
7trainable_variables
8regularization_losses
9	keras_api
:__call__
*;&call_and_return_all_conditional_losses

<kernel
=bias
 >_jit_compiled_convolution_op*
�
?	variables
@trainable_variables
Aregularization_losses
B	keras_api
C__call__
*D&call_and_return_all_conditional_losses

Ekernel
Fbias
 G_jit_compiled_convolution_op*
�
H	variables
Itrainable_variables
Jregularization_losses
K	keras_api
L__call__
*M&call_and_return_all_conditional_losses* 
�
N	variables
Otrainable_variables
Pregularization_losses
Q	keras_api
R__call__
*S&call_and_return_all_conditional_losses

Tkernel
Ubias
 V_jit_compiled_convolution_op*
�
W	variables
Xtrainable_variables
Yregularization_losses
Z	keras_api
[__call__
*\&call_and_return_all_conditional_losses

]kernel
^bias
 __jit_compiled_convolution_op*
�
`	variables
atrainable_variables
bregularization_losses
c	keras_api
d__call__
*e&call_and_return_all_conditional_losses* 
�
f	variables
gtrainable_variables
hregularization_losses
i	keras_api
j__call__
*k&call_and_return_all_conditional_losses

lkernel
mbias
 n_jit_compiled_convolution_op*
�
o	variables
ptrainable_variables
qregularization_losses
r	keras_api
s__call__
*t&call_and_return_all_conditional_losses

ukernel
vbias
 w_jit_compiled_convolution_op*
�
x	variables
ytrainable_variables
zregularization_losses
{	keras_api
|__call__
*}&call_and_return_all_conditional_losses* 
�
~	variables
trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
*0
+1
32
43
<4
=5
E6
F7
T8
U9
]10
^11
l12
m13
u14
v15
�16
�17
�18
�19
�20
�21
�22
�23
�24
�25
�26
�27
�28
�29
�30
�31
�32
�33
�34
�35
�36
�37*
�
*0
+1
32
43
<4
=5
E6
F7
T8
U9
]10
^11
l12
m13
u14
v15
�16
�17
�18
�19
�20
�21
�22
�23
�24
�25
�26
�27
�28
�29
�30
�31
�32
�33
�34
�35
�36
�37*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
__call__
!_default_save_signature
* &call_and_return_all_conditional_losses
& "call_and_return_conditional_losses*
:
�trace_0
�trace_1
�trace_2
�trace_3* 
:
�trace_0
�trace_1
�trace_2
�trace_3* 
* 
�
�
loss_scale
�base_optimizer
	�iter
�beta_1
�beta_2

�decay
�learning_rate*m�+m�3m�4m�<m�=m�Em�Fm�Tm�Um�]m�^m�lm�mm�um�vm�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�*v�+v�3v�4v�<v�=v�Ev�Fv�Tv�Uv�]v�^v�lv�mv�uv�vv�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�*vhat�+vhat�3vhat�4vhat�<vhat�=vhat�Evhat�Fvhat�Tvhat�Uvhat�]vhat�^vhat�lvhat�mvhat�uvhat�vvhat��vhat��vhat��vhat��vhat��vhat��vhat��vhat��vhat��vhat��vhat��vhat��vhat��vhat��vhat��vhat��vhat��vhat��vhat��vhat��vhat��vhat��vhat�*

�serving_default* 

*0
+1*

*0
+1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
$	variables
%trainable_variables
&regularization_losses
(__call__
*)&call_and_return_all_conditional_losses
&)"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
]W
VARIABLE_VALUEconv2d/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEconv2d/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

30
41*

30
41*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
-	variables
.trainable_variables
/regularization_losses
1__call__
*2&call_and_return_all_conditional_losses
&2"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
_Y
VARIABLE_VALUEconv2d_1/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEconv2d_1/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

<0
=1*

<0
=1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
6	variables
7trainable_variables
8regularization_losses
:__call__
*;&call_and_return_all_conditional_losses
&;"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
_Y
VARIABLE_VALUEconv2d_2/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEconv2d_2/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

E0
F1*

E0
F1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
?	variables
@trainable_variables
Aregularization_losses
C__call__
*D&call_and_return_all_conditional_losses
&D"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
_Y
VARIABLE_VALUEconv2d_3/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEconv2d_3/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
H	variables
Itrainable_variables
Jregularization_losses
L__call__
*M&call_and_return_all_conditional_losses
&M"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

T0
U1*

T0
U1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
N	variables
Otrainable_variables
Pregularization_losses
R__call__
*S&call_and_return_all_conditional_losses
&S"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
_Y
VARIABLE_VALUEconv2d_4/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEconv2d_4/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

]0
^1*

]0
^1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
W	variables
Xtrainable_variables
Yregularization_losses
[__call__
*\&call_and_return_all_conditional_losses
&\"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
_Y
VARIABLE_VALUEconv2d_5/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEconv2d_5/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
`	variables
atrainable_variables
bregularization_losses
d__call__
*e&call_and_return_all_conditional_losses
&e"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

l0
m1*

l0
m1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
f	variables
gtrainable_variables
hregularization_losses
j__call__
*k&call_and_return_all_conditional_losses
&k"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
_Y
VARIABLE_VALUEconv2d_6/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEconv2d_6/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

u0
v1*

u0
v1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
o	variables
ptrainable_variables
qregularization_losses
s__call__
*t&call_and_return_all_conditional_losses
&t"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
_Y
VARIABLE_VALUEconv2d_7/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEconv2d_7/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
x	variables
ytrainable_variables
zregularization_losses
|__call__
*}&call_and_return_all_conditional_losses
&}"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
~	variables
trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
_Y
VARIABLE_VALUEconv2d_8/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEconv2d_8/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
_Y
VARIABLE_VALUEconv2d_9/kernel6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEconv2d_9/bias4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
a[
VARIABLE_VALUEconv2d_10/kernel7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEconv2d_10/bias5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
a[
VARIABLE_VALUEconv2d_11/kernel7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEconv2d_11/bias5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
hb
VARIABLE_VALUEconv2d_transpose/kernel7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUEconv2d_transpose/bias5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
a[
VARIABLE_VALUEconv2d_12/kernel7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEconv2d_12/bias5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
a[
VARIABLE_VALUEconv2d_13/kernel7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEconv2d_13/bias5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
jd
VARIABLE_VALUEconv2d_transpose_1/kernel7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUEconv2d_transpose_1/bias5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
a[
VARIABLE_VALUEconv2d_14/kernel7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEconv2d_14/bias5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
a[
VARIABLE_VALUEconv2d_15/kernel7layer_with_weights-17/kernel/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEconv2d_15/bias5layer_with_weights-17/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
a[
VARIABLE_VALUEconv2d_16/kernel7layer_with_weights-18/kernel/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEconv2d_16/bias5layer_with_weights-18/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
* 
�
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25*

�0
�1
�2*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
.
�current_loss_scale
�
good_steps*
* 
SM
VARIABLE_VALUEcond_1/Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
WQ
VARIABLE_VALUEcond_1/Adam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
WQ
VARIABLE_VALUEcond_1/Adam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcond_1/Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUEcond_1/Adam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
<
�	variables
�	keras_api

�total

�count*
M
�	variables
�	keras_api

�total

�count
�
_fn_kwargs*
M
�	variables
�	keras_api

�total

�count
�
_fn_kwargs*
nh
VARIABLE_VALUEcurrent_loss_scaleBoptimizer/loss_scale/current_loss_scale/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUE
good_steps:optimizer/loss_scale/good_steps/.ATTRIBUTES/VARIABLE_VALUE*

�0
�1*

�	variables*
UO
VARIABLE_VALUEtotal_24keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_24keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

�0
�1*

�	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 

�0
�1*

�	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
��
VARIABLE_VALUEcond_1/Adam/conv2d/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
�}
VARIABLE_VALUEcond_1/Adam/conv2d/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEcond_1/Adam/conv2d_1/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
�
VARIABLE_VALUEcond_1/Adam/conv2d_1/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEcond_1/Adam/conv2d_2/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
�
VARIABLE_VALUEcond_1/Adam/conv2d_2/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEcond_1/Adam/conv2d_3/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
�
VARIABLE_VALUEcond_1/Adam/conv2d_3/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEcond_1/Adam/conv2d_4/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
�
VARIABLE_VALUEcond_1/Adam/conv2d_4/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEcond_1/Adam/conv2d_5/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
�
VARIABLE_VALUEcond_1/Adam/conv2d_5/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEcond_1/Adam/conv2d_6/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
�
VARIABLE_VALUEcond_1/Adam/conv2d_6/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEcond_1/Adam/conv2d_7/kernel/mRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
�
VARIABLE_VALUEcond_1/Adam/conv2d_7/bias/mPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEcond_1/Adam/conv2d_8/kernel/mRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
�
VARIABLE_VALUEcond_1/Adam/conv2d_8/bias/mPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEcond_1/Adam/conv2d_9/kernel/mRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
�
VARIABLE_VALUEcond_1/Adam/conv2d_9/bias/mPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEcond_1/Adam/conv2d_10/kernel/mSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEcond_1/Adam/conv2d_10/bias/mQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEcond_1/Adam/conv2d_11/kernel/mSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEcond_1/Adam/conv2d_11/bias/mQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE%cond_1/Adam/conv2d_transpose/kernel/mSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE#cond_1/Adam/conv2d_transpose/bias/mQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEcond_1/Adam/conv2d_12/kernel/mSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEcond_1/Adam/conv2d_12/bias/mQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEcond_1/Adam/conv2d_13/kernel/mSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEcond_1/Adam/conv2d_13/bias/mQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE'cond_1/Adam/conv2d_transpose_1/kernel/mSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE%cond_1/Adam/conv2d_transpose_1/bias/mQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEcond_1/Adam/conv2d_14/kernel/mSlayer_with_weights-16/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEcond_1/Adam/conv2d_14/bias/mQlayer_with_weights-16/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEcond_1/Adam/conv2d_15/kernel/mSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEcond_1/Adam/conv2d_15/bias/mQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEcond_1/Adam/conv2d_16/kernel/mSlayer_with_weights-18/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEcond_1/Adam/conv2d_16/bias/mQlayer_with_weights-18/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEcond_1/Adam/conv2d/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
�}
VARIABLE_VALUEcond_1/Adam/conv2d/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEcond_1/Adam/conv2d_1/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
�
VARIABLE_VALUEcond_1/Adam/conv2d_1/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEcond_1/Adam/conv2d_2/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
�
VARIABLE_VALUEcond_1/Adam/conv2d_2/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEcond_1/Adam/conv2d_3/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
�
VARIABLE_VALUEcond_1/Adam/conv2d_3/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEcond_1/Adam/conv2d_4/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
�
VARIABLE_VALUEcond_1/Adam/conv2d_4/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEcond_1/Adam/conv2d_5/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
�
VARIABLE_VALUEcond_1/Adam/conv2d_5/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEcond_1/Adam/conv2d_6/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
�
VARIABLE_VALUEcond_1/Adam/conv2d_6/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEcond_1/Adam/conv2d_7/kernel/vRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
�
VARIABLE_VALUEcond_1/Adam/conv2d_7/bias/vPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEcond_1/Adam/conv2d_8/kernel/vRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
�
VARIABLE_VALUEcond_1/Adam/conv2d_8/bias/vPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEcond_1/Adam/conv2d_9/kernel/vRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
�
VARIABLE_VALUEcond_1/Adam/conv2d_9/bias/vPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEcond_1/Adam/conv2d_10/kernel/vSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEcond_1/Adam/conv2d_10/bias/vQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEcond_1/Adam/conv2d_11/kernel/vSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEcond_1/Adam/conv2d_11/bias/vQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE%cond_1/Adam/conv2d_transpose/kernel/vSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE#cond_1/Adam/conv2d_transpose/bias/vQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEcond_1/Adam/conv2d_12/kernel/vSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEcond_1/Adam/conv2d_12/bias/vQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEcond_1/Adam/conv2d_13/kernel/vSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEcond_1/Adam/conv2d_13/bias/vQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE'cond_1/Adam/conv2d_transpose_1/kernel/vSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE%cond_1/Adam/conv2d_transpose_1/bias/vQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEcond_1/Adam/conv2d_14/kernel/vSlayer_with_weights-16/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEcond_1/Adam/conv2d_14/bias/vQlayer_with_weights-16/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEcond_1/Adam/conv2d_15/kernel/vSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEcond_1/Adam/conv2d_15/bias/vQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEcond_1/Adam/conv2d_16/kernel/vSlayer_with_weights-18/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEcond_1/Adam/conv2d_16/bias/vQlayer_with_weights-18/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEcond_1/Adam/conv2d/kernel/vhatUlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEcond_1/Adam/conv2d/bias/vhatSlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE cond_1/Adam/conv2d_1/kernel/vhatUlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEcond_1/Adam/conv2d_1/bias/vhatSlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE cond_1/Adam/conv2d_2/kernel/vhatUlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEcond_1/Adam/conv2d_2/bias/vhatSlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE cond_1/Adam/conv2d_3/kernel/vhatUlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEcond_1/Adam/conv2d_3/bias/vhatSlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE cond_1/Adam/conv2d_4/kernel/vhatUlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEcond_1/Adam/conv2d_4/bias/vhatSlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE cond_1/Adam/conv2d_5/kernel/vhatUlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEcond_1/Adam/conv2d_5/bias/vhatSlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE cond_1/Adam/conv2d_6/kernel/vhatUlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEcond_1/Adam/conv2d_6/bias/vhatSlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE cond_1/Adam/conv2d_7/kernel/vhatUlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEcond_1/Adam/conv2d_7/bias/vhatSlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE cond_1/Adam/conv2d_8/kernel/vhatUlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEcond_1/Adam/conv2d_8/bias/vhatSlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE cond_1/Adam/conv2d_9/kernel/vhatUlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEcond_1/Adam/conv2d_9/bias/vhatSlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE!cond_1/Adam/conv2d_10/kernel/vhatVlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEcond_1/Adam/conv2d_10/bias/vhatTlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE!cond_1/Adam/conv2d_11/kernel/vhatVlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEcond_1/Adam/conv2d_11/bias/vhatTlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE(cond_1/Adam/conv2d_transpose/kernel/vhatVlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE&cond_1/Adam/conv2d_transpose/bias/vhatTlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE!cond_1/Adam/conv2d_12/kernel/vhatVlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEcond_1/Adam/conv2d_12/bias/vhatTlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE!cond_1/Adam/conv2d_13/kernel/vhatVlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEcond_1/Adam/conv2d_13/bias/vhatTlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE*cond_1/Adam/conv2d_transpose_1/kernel/vhatVlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE(cond_1/Adam/conv2d_transpose_1/bias/vhatTlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE!cond_1/Adam/conv2d_14/kernel/vhatVlayer_with_weights-16/kernel/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEcond_1/Adam/conv2d_14/bias/vhatTlayer_with_weights-16/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE!cond_1/Adam/conv2d_15/kernel/vhatVlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEcond_1/Adam/conv2d_15/bias/vhatTlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE!cond_1/Adam/conv2d_16/kernel/vhatVlayer_with_weights-18/kernel/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEcond_1/Adam/conv2d_16/bias/vhatTlayer_with_weights-18/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE*
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�@
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename!conv2d/kernel/Read/ReadVariableOpconv2d/bias/Read/ReadVariableOp#conv2d_1/kernel/Read/ReadVariableOp!conv2d_1/bias/Read/ReadVariableOp#conv2d_2/kernel/Read/ReadVariableOp!conv2d_2/bias/Read/ReadVariableOp#conv2d_3/kernel/Read/ReadVariableOp!conv2d_3/bias/Read/ReadVariableOp#conv2d_4/kernel/Read/ReadVariableOp!conv2d_4/bias/Read/ReadVariableOp#conv2d_5/kernel/Read/ReadVariableOp!conv2d_5/bias/Read/ReadVariableOp#conv2d_6/kernel/Read/ReadVariableOp!conv2d_6/bias/Read/ReadVariableOp#conv2d_7/kernel/Read/ReadVariableOp!conv2d_7/bias/Read/ReadVariableOp#conv2d_8/kernel/Read/ReadVariableOp!conv2d_8/bias/Read/ReadVariableOp#conv2d_9/kernel/Read/ReadVariableOp!conv2d_9/bias/Read/ReadVariableOp$conv2d_10/kernel/Read/ReadVariableOp"conv2d_10/bias/Read/ReadVariableOp$conv2d_11/kernel/Read/ReadVariableOp"conv2d_11/bias/Read/ReadVariableOp+conv2d_transpose/kernel/Read/ReadVariableOp)conv2d_transpose/bias/Read/ReadVariableOp$conv2d_12/kernel/Read/ReadVariableOp"conv2d_12/bias/Read/ReadVariableOp$conv2d_13/kernel/Read/ReadVariableOp"conv2d_13/bias/Read/ReadVariableOp-conv2d_transpose_1/kernel/Read/ReadVariableOp+conv2d_transpose_1/bias/Read/ReadVariableOp$conv2d_14/kernel/Read/ReadVariableOp"conv2d_14/bias/Read/ReadVariableOp$conv2d_15/kernel/Read/ReadVariableOp"conv2d_15/bias/Read/ReadVariableOp$conv2d_16/kernel/Read/ReadVariableOp"conv2d_16/bias/Read/ReadVariableOp$cond_1/Adam/iter/Read/ReadVariableOp&cond_1/Adam/beta_1/Read/ReadVariableOp&cond_1/Adam/beta_2/Read/ReadVariableOp%cond_1/Adam/decay/Read/ReadVariableOp-cond_1/Adam/learning_rate/Read/ReadVariableOp&current_loss_scale/Read/ReadVariableOpgood_steps/Read/ReadVariableOptotal_2/Read/ReadVariableOpcount_2/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp/cond_1/Adam/conv2d/kernel/m/Read/ReadVariableOp-cond_1/Adam/conv2d/bias/m/Read/ReadVariableOp1cond_1/Adam/conv2d_1/kernel/m/Read/ReadVariableOp/cond_1/Adam/conv2d_1/bias/m/Read/ReadVariableOp1cond_1/Adam/conv2d_2/kernel/m/Read/ReadVariableOp/cond_1/Adam/conv2d_2/bias/m/Read/ReadVariableOp1cond_1/Adam/conv2d_3/kernel/m/Read/ReadVariableOp/cond_1/Adam/conv2d_3/bias/m/Read/ReadVariableOp1cond_1/Adam/conv2d_4/kernel/m/Read/ReadVariableOp/cond_1/Adam/conv2d_4/bias/m/Read/ReadVariableOp1cond_1/Adam/conv2d_5/kernel/m/Read/ReadVariableOp/cond_1/Adam/conv2d_5/bias/m/Read/ReadVariableOp1cond_1/Adam/conv2d_6/kernel/m/Read/ReadVariableOp/cond_1/Adam/conv2d_6/bias/m/Read/ReadVariableOp1cond_1/Adam/conv2d_7/kernel/m/Read/ReadVariableOp/cond_1/Adam/conv2d_7/bias/m/Read/ReadVariableOp1cond_1/Adam/conv2d_8/kernel/m/Read/ReadVariableOp/cond_1/Adam/conv2d_8/bias/m/Read/ReadVariableOp1cond_1/Adam/conv2d_9/kernel/m/Read/ReadVariableOp/cond_1/Adam/conv2d_9/bias/m/Read/ReadVariableOp2cond_1/Adam/conv2d_10/kernel/m/Read/ReadVariableOp0cond_1/Adam/conv2d_10/bias/m/Read/ReadVariableOp2cond_1/Adam/conv2d_11/kernel/m/Read/ReadVariableOp0cond_1/Adam/conv2d_11/bias/m/Read/ReadVariableOp9cond_1/Adam/conv2d_transpose/kernel/m/Read/ReadVariableOp7cond_1/Adam/conv2d_transpose/bias/m/Read/ReadVariableOp2cond_1/Adam/conv2d_12/kernel/m/Read/ReadVariableOp0cond_1/Adam/conv2d_12/bias/m/Read/ReadVariableOp2cond_1/Adam/conv2d_13/kernel/m/Read/ReadVariableOp0cond_1/Adam/conv2d_13/bias/m/Read/ReadVariableOp;cond_1/Adam/conv2d_transpose_1/kernel/m/Read/ReadVariableOp9cond_1/Adam/conv2d_transpose_1/bias/m/Read/ReadVariableOp2cond_1/Adam/conv2d_14/kernel/m/Read/ReadVariableOp0cond_1/Adam/conv2d_14/bias/m/Read/ReadVariableOp2cond_1/Adam/conv2d_15/kernel/m/Read/ReadVariableOp0cond_1/Adam/conv2d_15/bias/m/Read/ReadVariableOp2cond_1/Adam/conv2d_16/kernel/m/Read/ReadVariableOp0cond_1/Adam/conv2d_16/bias/m/Read/ReadVariableOp/cond_1/Adam/conv2d/kernel/v/Read/ReadVariableOp-cond_1/Adam/conv2d/bias/v/Read/ReadVariableOp1cond_1/Adam/conv2d_1/kernel/v/Read/ReadVariableOp/cond_1/Adam/conv2d_1/bias/v/Read/ReadVariableOp1cond_1/Adam/conv2d_2/kernel/v/Read/ReadVariableOp/cond_1/Adam/conv2d_2/bias/v/Read/ReadVariableOp1cond_1/Adam/conv2d_3/kernel/v/Read/ReadVariableOp/cond_1/Adam/conv2d_3/bias/v/Read/ReadVariableOp1cond_1/Adam/conv2d_4/kernel/v/Read/ReadVariableOp/cond_1/Adam/conv2d_4/bias/v/Read/ReadVariableOp1cond_1/Adam/conv2d_5/kernel/v/Read/ReadVariableOp/cond_1/Adam/conv2d_5/bias/v/Read/ReadVariableOp1cond_1/Adam/conv2d_6/kernel/v/Read/ReadVariableOp/cond_1/Adam/conv2d_6/bias/v/Read/ReadVariableOp1cond_1/Adam/conv2d_7/kernel/v/Read/ReadVariableOp/cond_1/Adam/conv2d_7/bias/v/Read/ReadVariableOp1cond_1/Adam/conv2d_8/kernel/v/Read/ReadVariableOp/cond_1/Adam/conv2d_8/bias/v/Read/ReadVariableOp1cond_1/Adam/conv2d_9/kernel/v/Read/ReadVariableOp/cond_1/Adam/conv2d_9/bias/v/Read/ReadVariableOp2cond_1/Adam/conv2d_10/kernel/v/Read/ReadVariableOp0cond_1/Adam/conv2d_10/bias/v/Read/ReadVariableOp2cond_1/Adam/conv2d_11/kernel/v/Read/ReadVariableOp0cond_1/Adam/conv2d_11/bias/v/Read/ReadVariableOp9cond_1/Adam/conv2d_transpose/kernel/v/Read/ReadVariableOp7cond_1/Adam/conv2d_transpose/bias/v/Read/ReadVariableOp2cond_1/Adam/conv2d_12/kernel/v/Read/ReadVariableOp0cond_1/Adam/conv2d_12/bias/v/Read/ReadVariableOp2cond_1/Adam/conv2d_13/kernel/v/Read/ReadVariableOp0cond_1/Adam/conv2d_13/bias/v/Read/ReadVariableOp;cond_1/Adam/conv2d_transpose_1/kernel/v/Read/ReadVariableOp9cond_1/Adam/conv2d_transpose_1/bias/v/Read/ReadVariableOp2cond_1/Adam/conv2d_14/kernel/v/Read/ReadVariableOp0cond_1/Adam/conv2d_14/bias/v/Read/ReadVariableOp2cond_1/Adam/conv2d_15/kernel/v/Read/ReadVariableOp0cond_1/Adam/conv2d_15/bias/v/Read/ReadVariableOp2cond_1/Adam/conv2d_16/kernel/v/Read/ReadVariableOp0cond_1/Adam/conv2d_16/bias/v/Read/ReadVariableOp2cond_1/Adam/conv2d/kernel/vhat/Read/ReadVariableOp0cond_1/Adam/conv2d/bias/vhat/Read/ReadVariableOp4cond_1/Adam/conv2d_1/kernel/vhat/Read/ReadVariableOp2cond_1/Adam/conv2d_1/bias/vhat/Read/ReadVariableOp4cond_1/Adam/conv2d_2/kernel/vhat/Read/ReadVariableOp2cond_1/Adam/conv2d_2/bias/vhat/Read/ReadVariableOp4cond_1/Adam/conv2d_3/kernel/vhat/Read/ReadVariableOp2cond_1/Adam/conv2d_3/bias/vhat/Read/ReadVariableOp4cond_1/Adam/conv2d_4/kernel/vhat/Read/ReadVariableOp2cond_1/Adam/conv2d_4/bias/vhat/Read/ReadVariableOp4cond_1/Adam/conv2d_5/kernel/vhat/Read/ReadVariableOp2cond_1/Adam/conv2d_5/bias/vhat/Read/ReadVariableOp4cond_1/Adam/conv2d_6/kernel/vhat/Read/ReadVariableOp2cond_1/Adam/conv2d_6/bias/vhat/Read/ReadVariableOp4cond_1/Adam/conv2d_7/kernel/vhat/Read/ReadVariableOp2cond_1/Adam/conv2d_7/bias/vhat/Read/ReadVariableOp4cond_1/Adam/conv2d_8/kernel/vhat/Read/ReadVariableOp2cond_1/Adam/conv2d_8/bias/vhat/Read/ReadVariableOp4cond_1/Adam/conv2d_9/kernel/vhat/Read/ReadVariableOp2cond_1/Adam/conv2d_9/bias/vhat/Read/ReadVariableOp5cond_1/Adam/conv2d_10/kernel/vhat/Read/ReadVariableOp3cond_1/Adam/conv2d_10/bias/vhat/Read/ReadVariableOp5cond_1/Adam/conv2d_11/kernel/vhat/Read/ReadVariableOp3cond_1/Adam/conv2d_11/bias/vhat/Read/ReadVariableOp<cond_1/Adam/conv2d_transpose/kernel/vhat/Read/ReadVariableOp:cond_1/Adam/conv2d_transpose/bias/vhat/Read/ReadVariableOp5cond_1/Adam/conv2d_12/kernel/vhat/Read/ReadVariableOp3cond_1/Adam/conv2d_12/bias/vhat/Read/ReadVariableOp5cond_1/Adam/conv2d_13/kernel/vhat/Read/ReadVariableOp3cond_1/Adam/conv2d_13/bias/vhat/Read/ReadVariableOp>cond_1/Adam/conv2d_transpose_1/kernel/vhat/Read/ReadVariableOp<cond_1/Adam/conv2d_transpose_1/bias/vhat/Read/ReadVariableOp5cond_1/Adam/conv2d_14/kernel/vhat/Read/ReadVariableOp3cond_1/Adam/conv2d_14/bias/vhat/Read/ReadVariableOp5cond_1/Adam/conv2d_15/kernel/vhat/Read/ReadVariableOp3cond_1/Adam/conv2d_15/bias/vhat/Read/ReadVariableOp5cond_1/Adam/conv2d_16/kernel/vhat/Read/ReadVariableOp3cond_1/Adam/conv2d_16/bias/vhat/Read/ReadVariableOpConst*�
Tin�
�2�		*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *'
f"R 
__inference__traced_save_56936
�&
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d/kernelconv2d/biasconv2d_1/kernelconv2d_1/biasconv2d_2/kernelconv2d_2/biasconv2d_3/kernelconv2d_3/biasconv2d_4/kernelconv2d_4/biasconv2d_5/kernelconv2d_5/biasconv2d_6/kernelconv2d_6/biasconv2d_7/kernelconv2d_7/biasconv2d_8/kernelconv2d_8/biasconv2d_9/kernelconv2d_9/biasconv2d_10/kernelconv2d_10/biasconv2d_11/kernelconv2d_11/biasconv2d_transpose/kernelconv2d_transpose/biasconv2d_12/kernelconv2d_12/biasconv2d_13/kernelconv2d_13/biasconv2d_transpose_1/kernelconv2d_transpose_1/biasconv2d_14/kernelconv2d_14/biasconv2d_15/kernelconv2d_15/biasconv2d_16/kernelconv2d_16/biascond_1/Adam/itercond_1/Adam/beta_1cond_1/Adam/beta_2cond_1/Adam/decaycond_1/Adam/learning_ratecurrent_loss_scale
good_stepstotal_2count_2total_1count_1totalcountcond_1/Adam/conv2d/kernel/mcond_1/Adam/conv2d/bias/mcond_1/Adam/conv2d_1/kernel/mcond_1/Adam/conv2d_1/bias/mcond_1/Adam/conv2d_2/kernel/mcond_1/Adam/conv2d_2/bias/mcond_1/Adam/conv2d_3/kernel/mcond_1/Adam/conv2d_3/bias/mcond_1/Adam/conv2d_4/kernel/mcond_1/Adam/conv2d_4/bias/mcond_1/Adam/conv2d_5/kernel/mcond_1/Adam/conv2d_5/bias/mcond_1/Adam/conv2d_6/kernel/mcond_1/Adam/conv2d_6/bias/mcond_1/Adam/conv2d_7/kernel/mcond_1/Adam/conv2d_7/bias/mcond_1/Adam/conv2d_8/kernel/mcond_1/Adam/conv2d_8/bias/mcond_1/Adam/conv2d_9/kernel/mcond_1/Adam/conv2d_9/bias/mcond_1/Adam/conv2d_10/kernel/mcond_1/Adam/conv2d_10/bias/mcond_1/Adam/conv2d_11/kernel/mcond_1/Adam/conv2d_11/bias/m%cond_1/Adam/conv2d_transpose/kernel/m#cond_1/Adam/conv2d_transpose/bias/mcond_1/Adam/conv2d_12/kernel/mcond_1/Adam/conv2d_12/bias/mcond_1/Adam/conv2d_13/kernel/mcond_1/Adam/conv2d_13/bias/m'cond_1/Adam/conv2d_transpose_1/kernel/m%cond_1/Adam/conv2d_transpose_1/bias/mcond_1/Adam/conv2d_14/kernel/mcond_1/Adam/conv2d_14/bias/mcond_1/Adam/conv2d_15/kernel/mcond_1/Adam/conv2d_15/bias/mcond_1/Adam/conv2d_16/kernel/mcond_1/Adam/conv2d_16/bias/mcond_1/Adam/conv2d/kernel/vcond_1/Adam/conv2d/bias/vcond_1/Adam/conv2d_1/kernel/vcond_1/Adam/conv2d_1/bias/vcond_1/Adam/conv2d_2/kernel/vcond_1/Adam/conv2d_2/bias/vcond_1/Adam/conv2d_3/kernel/vcond_1/Adam/conv2d_3/bias/vcond_1/Adam/conv2d_4/kernel/vcond_1/Adam/conv2d_4/bias/vcond_1/Adam/conv2d_5/kernel/vcond_1/Adam/conv2d_5/bias/vcond_1/Adam/conv2d_6/kernel/vcond_1/Adam/conv2d_6/bias/vcond_1/Adam/conv2d_7/kernel/vcond_1/Adam/conv2d_7/bias/vcond_1/Adam/conv2d_8/kernel/vcond_1/Adam/conv2d_8/bias/vcond_1/Adam/conv2d_9/kernel/vcond_1/Adam/conv2d_9/bias/vcond_1/Adam/conv2d_10/kernel/vcond_1/Adam/conv2d_10/bias/vcond_1/Adam/conv2d_11/kernel/vcond_1/Adam/conv2d_11/bias/v%cond_1/Adam/conv2d_transpose/kernel/v#cond_1/Adam/conv2d_transpose/bias/vcond_1/Adam/conv2d_12/kernel/vcond_1/Adam/conv2d_12/bias/vcond_1/Adam/conv2d_13/kernel/vcond_1/Adam/conv2d_13/bias/v'cond_1/Adam/conv2d_transpose_1/kernel/v%cond_1/Adam/conv2d_transpose_1/bias/vcond_1/Adam/conv2d_14/kernel/vcond_1/Adam/conv2d_14/bias/vcond_1/Adam/conv2d_15/kernel/vcond_1/Adam/conv2d_15/bias/vcond_1/Adam/conv2d_16/kernel/vcond_1/Adam/conv2d_16/bias/vcond_1/Adam/conv2d/kernel/vhatcond_1/Adam/conv2d/bias/vhat cond_1/Adam/conv2d_1/kernel/vhatcond_1/Adam/conv2d_1/bias/vhat cond_1/Adam/conv2d_2/kernel/vhatcond_1/Adam/conv2d_2/bias/vhat cond_1/Adam/conv2d_3/kernel/vhatcond_1/Adam/conv2d_3/bias/vhat cond_1/Adam/conv2d_4/kernel/vhatcond_1/Adam/conv2d_4/bias/vhat cond_1/Adam/conv2d_5/kernel/vhatcond_1/Adam/conv2d_5/bias/vhat cond_1/Adam/conv2d_6/kernel/vhatcond_1/Adam/conv2d_6/bias/vhat cond_1/Adam/conv2d_7/kernel/vhatcond_1/Adam/conv2d_7/bias/vhat cond_1/Adam/conv2d_8/kernel/vhatcond_1/Adam/conv2d_8/bias/vhat cond_1/Adam/conv2d_9/kernel/vhatcond_1/Adam/conv2d_9/bias/vhat!cond_1/Adam/conv2d_10/kernel/vhatcond_1/Adam/conv2d_10/bias/vhat!cond_1/Adam/conv2d_11/kernel/vhatcond_1/Adam/conv2d_11/bias/vhat(cond_1/Adam/conv2d_transpose/kernel/vhat&cond_1/Adam/conv2d_transpose/bias/vhat!cond_1/Adam/conv2d_12/kernel/vhatcond_1/Adam/conv2d_12/bias/vhat!cond_1/Adam/conv2d_13/kernel/vhatcond_1/Adam/conv2d_13/bias/vhat*cond_1/Adam/conv2d_transpose_1/kernel/vhat(cond_1/Adam/conv2d_transpose_1/bias/vhat!cond_1/Adam/conv2d_14/kernel/vhatcond_1/Adam/conv2d_14/bias/vhat!cond_1/Adam/conv2d_15/kernel/vhatcond_1/Adam/conv2d_15/bias/vhat!cond_1/Adam/conv2d_16/kernel/vhatcond_1/Adam/conv2d_16/bias/vhat*�
Tin�
�2�*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� **
f%R#
!__inference__traced_restore_57441� 
�w
�
@__inference_model_layer_call_and_return_conditional_losses_54459
input_1&
conv2d_54355:
conv2d_54357:(
conv2d_1_54360:
conv2d_1_54362:(
conv2d_2_54365:
conv2d_2_54367:(
conv2d_3_54370:
conv2d_3_54372:(
conv2d_4_54377:<
conv2d_4_54379:<(
conv2d_5_54382:<<
conv2d_5_54384:<(
conv2d_6_54388:<x
conv2d_6_54390:x(
conv2d_7_54393:xx
conv2d_7_54395:x)
conv2d_8_54399:x�
conv2d_8_54401:	�*
conv2d_9_54404:��
conv2d_9_54406:	�+
conv2d_10_54409:��
conv2d_10_54411:	�+
conv2d_11_54414:��
conv2d_11_54416:	�1
conv2d_transpose_54419:x�$
conv2d_transpose_54421:x)
conv2d_12_54425:xx
conv2d_12_54427:x)
conv2d_13_54430:xx
conv2d_13_54432:x2
conv2d_transpose_1_54435:<x&
conv2d_transpose_1_54437:<)
conv2d_14_54441:<<
conv2d_14_54443:<)
conv2d_15_54446:<<
conv2d_15_54448:<)
conv2d_16_54451:<
conv2d_16_54453:
identity��conv2d/StatefulPartitionedCall� conv2d_1/StatefulPartitionedCall�!conv2d_10/StatefulPartitionedCall�!conv2d_11/StatefulPartitionedCall�!conv2d_12/StatefulPartitionedCall�!conv2d_13/StatefulPartitionedCall�!conv2d_14/StatefulPartitionedCall�!conv2d_15/StatefulPartitionedCall�!conv2d_16/StatefulPartitionedCall� conv2d_2/StatefulPartitionedCall� conv2d_3/StatefulPartitionedCall� conv2d_4/StatefulPartitionedCall� conv2d_5/StatefulPartitionedCall� conv2d_6/StatefulPartitionedCall� conv2d_7/StatefulPartitionedCall� conv2d_8/StatefulPartitionedCall� conv2d_9/StatefulPartitionedCall�(conv2d_transpose/StatefulPartitionedCall�*conv2d_transpose_1/StatefulPartitionedCallg
conv2d/CastCastinput_1*

DstT0*

SrcT0*1
_output_shapes
:������������
conv2d/StatefulPartitionedCallStatefulPartitionedCallconv2d/Cast:y:0conv2d_54355conv2d_54357*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_53186�
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0conv2d_1_54360conv2d_1_54362*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_53217�
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0conv2d_2_54365conv2d_2_54367*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_conv2d_2_layer_call_and_return_conditional_losses_53248�
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0conv2d_3_54370conv2d_3_54372*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_conv2d_3_layer_call_and_return_conditional_losses_53279l
concatenate/CastCastinput_1*

DstT0*

SrcT0*1
_output_shapes
:������������
concatenate/PartitionedCallPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0concatenate/Cast:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_concatenate_layer_call_and_return_conditional_losses_53293�
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0conv2d_4_54377conv2d_4_54379*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������<*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_conv2d_4_layer_call_and_return_conditional_losses_53320�
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0conv2d_5_54382conv2d_5_54384*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������<*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_conv2d_5_layer_call_and_return_conditional_losses_53351�
!average_pooling2d/PartitionedCallPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������xx<* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_average_pooling2d_layer_call_and_return_conditional_losses_53020�
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCall*average_pooling2d/PartitionedCall:output:0conv2d_6_54388conv2d_6_54390*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������xxx*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_conv2d_6_layer_call_and_return_conditional_losses_53383�
 conv2d_7/StatefulPartitionedCallStatefulPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0conv2d_7_54393conv2d_7_54395*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������xxx*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_conv2d_7_layer_call_and_return_conditional_losses_53414�
#average_pooling2d_1/PartitionedCallPartitionedCall)conv2d_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������<<x* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *W
fRRP
N__inference_average_pooling2d_1_layer_call_and_return_conditional_losses_53032�
 conv2d_8/StatefulPartitionedCallStatefulPartitionedCall,average_pooling2d_1/PartitionedCall:output:0conv2d_8_54399conv2d_8_54401*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������<<�*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_conv2d_8_layer_call_and_return_conditional_losses_53446�
 conv2d_9/StatefulPartitionedCallStatefulPartitionedCall)conv2d_8/StatefulPartitionedCall:output:0conv2d_9_54404conv2d_9_54406*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������<<�*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_conv2d_9_layer_call_and_return_conditional_losses_53477�
!conv2d_10/StatefulPartitionedCallStatefulPartitionedCall)conv2d_9/StatefulPartitionedCall:output:0conv2d_10_54409conv2d_10_54411*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������<<�*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv2d_10_layer_call_and_return_conditional_losses_53508�
!conv2d_11/StatefulPartitionedCallStatefulPartitionedCall*conv2d_10/StatefulPartitionedCall:output:0conv2d_11_54414conv2d_11_54416*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������<<�*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv2d_11_layer_call_and_return_conditional_losses_53539�
(conv2d_transpose/StatefulPartitionedCallStatefulPartitionedCall*conv2d_11/StatefulPartitionedCall:output:0conv2d_transpose_54419conv2d_transpose_54421*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������xxx*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *T
fORM
K__inference_conv2d_transpose_layer_call_and_return_conditional_losses_53087�
add/PartitionedCallPartitionedCall1conv2d_transpose/StatefulPartitionedCall:output:0)conv2d_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������xxx* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *G
fBR@
>__inference_add_layer_call_and_return_conditional_losses_53556�
!conv2d_12/StatefulPartitionedCallStatefulPartitionedCalladd/PartitionedCall:output:0conv2d_12_54425conv2d_12_54427*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������xxx*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv2d_12_layer_call_and_return_conditional_losses_53583�
!conv2d_13/StatefulPartitionedCallStatefulPartitionedCall*conv2d_12/StatefulPartitionedCall:output:0conv2d_13_54430conv2d_13_54432*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������xxx*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv2d_13_layer_call_and_return_conditional_losses_53614�
*conv2d_transpose_1/StatefulPartitionedCallStatefulPartitionedCall*conv2d_13/StatefulPartitionedCall:output:0conv2d_transpose_1_54435conv2d_transpose_1_54437*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������<*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *V
fQRO
M__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_53146�
add_1/PartitionedCallPartitionedCall3conv2d_transpose_1/StatefulPartitionedCall:output:0)conv2d_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������<* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *I
fDRB
@__inference_add_1_layer_call_and_return_conditional_losses_53631�
!conv2d_14/StatefulPartitionedCallStatefulPartitionedCalladd_1/PartitionedCall:output:0conv2d_14_54441conv2d_14_54443*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������<*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv2d_14_layer_call_and_return_conditional_losses_53658�
!conv2d_15/StatefulPartitionedCallStatefulPartitionedCall*conv2d_14/StatefulPartitionedCall:output:0conv2d_15_54446conv2d_15_54448*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������<*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv2d_15_layer_call_and_return_conditional_losses_53689�
!conv2d_16/StatefulPartitionedCallStatefulPartitionedCall*conv2d_15/StatefulPartitionedCall:output:0conv2d_16_54451conv2d_16_54453*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv2d_16_layer_call_and_return_conditional_losses_53707�

add_2/CastCast*conv2d_16/StatefulPartitionedCall:output:0*

DstT0*

SrcT0*1
_output_shapes
:������������
add_2/PartitionedCallPartitionedCalladd_2/Cast:y:0input_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *I
fDRB
@__inference_add_2_layer_call_and_return_conditional_losses_53720w
IdentityIdentityadd_2/PartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:������������
NoOpNoOp^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall"^conv2d_10/StatefulPartitionedCall"^conv2d_11/StatefulPartitionedCall"^conv2d_12/StatefulPartitionedCall"^conv2d_13/StatefulPartitionedCall"^conv2d_14/StatefulPartitionedCall"^conv2d_15/StatefulPartitionedCall"^conv2d_16/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall!^conv2d_5/StatefulPartitionedCall!^conv2d_6/StatefulPartitionedCall!^conv2d_7/StatefulPartitionedCall!^conv2d_8/StatefulPartitionedCall!^conv2d_9/StatefulPartitionedCall)^conv2d_transpose/StatefulPartitionedCall+^conv2d_transpose_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*|
_input_shapesk
i:�����������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2F
!conv2d_10/StatefulPartitionedCall!conv2d_10/StatefulPartitionedCall2F
!conv2d_11/StatefulPartitionedCall!conv2d_11/StatefulPartitionedCall2F
!conv2d_12/StatefulPartitionedCall!conv2d_12/StatefulPartitionedCall2F
!conv2d_13/StatefulPartitionedCall!conv2d_13/StatefulPartitionedCall2F
!conv2d_14/StatefulPartitionedCall!conv2d_14/StatefulPartitionedCall2F
!conv2d_15/StatefulPartitionedCall!conv2d_15/StatefulPartitionedCall2F
!conv2d_16/StatefulPartitionedCall!conv2d_16/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall2D
 conv2d_7/StatefulPartitionedCall conv2d_7/StatefulPartitionedCall2D
 conv2d_8/StatefulPartitionedCall conv2d_8/StatefulPartitionedCall2D
 conv2d_9/StatefulPartitionedCall conv2d_9/StatefulPartitionedCall2T
(conv2d_transpose/StatefulPartitionedCall(conv2d_transpose/StatefulPartitionedCall2X
*conv2d_transpose_1/StatefulPartitionedCall*conv2d_transpose_1/StatefulPartitionedCall:Z V
1
_output_shapes
:�����������
!
_user_specified_name	input_1
��
�!
 __inference__wrapped_model_53011
input_1E
+model_conv2d_conv2d_readvariableop_resource::
,model_conv2d_biasadd_readvariableop_resource:G
-model_conv2d_1_conv2d_readvariableop_resource:<
.model_conv2d_1_biasadd_readvariableop_resource:G
-model_conv2d_2_conv2d_readvariableop_resource:<
.model_conv2d_2_biasadd_readvariableop_resource:G
-model_conv2d_3_conv2d_readvariableop_resource:<
.model_conv2d_3_biasadd_readvariableop_resource:G
-model_conv2d_4_conv2d_readvariableop_resource:<<
.model_conv2d_4_biasadd_readvariableop_resource:<G
-model_conv2d_5_conv2d_readvariableop_resource:<<<
.model_conv2d_5_biasadd_readvariableop_resource:<G
-model_conv2d_6_conv2d_readvariableop_resource:<x<
.model_conv2d_6_biasadd_readvariableop_resource:xG
-model_conv2d_7_conv2d_readvariableop_resource:xx<
.model_conv2d_7_biasadd_readvariableop_resource:xH
-model_conv2d_8_conv2d_readvariableop_resource:x�=
.model_conv2d_8_biasadd_readvariableop_resource:	�I
-model_conv2d_9_conv2d_readvariableop_resource:��=
.model_conv2d_9_biasadd_readvariableop_resource:	�J
.model_conv2d_10_conv2d_readvariableop_resource:��>
/model_conv2d_10_biasadd_readvariableop_resource:	�J
.model_conv2d_11_conv2d_readvariableop_resource:��>
/model_conv2d_11_biasadd_readvariableop_resource:	�Z
?model_conv2d_transpose_conv2d_transpose_readvariableop_resource:x�D
6model_conv2d_transpose_biasadd_readvariableop_resource:xH
.model_conv2d_12_conv2d_readvariableop_resource:xx=
/model_conv2d_12_biasadd_readvariableop_resource:xH
.model_conv2d_13_conv2d_readvariableop_resource:xx=
/model_conv2d_13_biasadd_readvariableop_resource:x[
Amodel_conv2d_transpose_1_conv2d_transpose_readvariableop_resource:<xF
8model_conv2d_transpose_1_biasadd_readvariableop_resource:<H
.model_conv2d_14_conv2d_readvariableop_resource:<<=
/model_conv2d_14_biasadd_readvariableop_resource:<H
.model_conv2d_15_conv2d_readvariableop_resource:<<=
/model_conv2d_15_biasadd_readvariableop_resource:<H
.model_conv2d_16_conv2d_readvariableop_resource:<=
/model_conv2d_16_biasadd_readvariableop_resource:
identity��#model/conv2d/BiasAdd/ReadVariableOp�"model/conv2d/Conv2D/ReadVariableOp�%model/conv2d_1/BiasAdd/ReadVariableOp�$model/conv2d_1/Conv2D/ReadVariableOp�&model/conv2d_10/BiasAdd/ReadVariableOp�%model/conv2d_10/Conv2D/ReadVariableOp�&model/conv2d_11/BiasAdd/ReadVariableOp�%model/conv2d_11/Conv2D/ReadVariableOp�&model/conv2d_12/BiasAdd/ReadVariableOp�%model/conv2d_12/Conv2D/ReadVariableOp�&model/conv2d_13/BiasAdd/ReadVariableOp�%model/conv2d_13/Conv2D/ReadVariableOp�&model/conv2d_14/BiasAdd/ReadVariableOp�%model/conv2d_14/Conv2D/ReadVariableOp�&model/conv2d_15/BiasAdd/ReadVariableOp�%model/conv2d_15/Conv2D/ReadVariableOp�&model/conv2d_16/BiasAdd/ReadVariableOp�%model/conv2d_16/Conv2D/ReadVariableOp�%model/conv2d_2/BiasAdd/ReadVariableOp�$model/conv2d_2/Conv2D/ReadVariableOp�%model/conv2d_3/BiasAdd/ReadVariableOp�$model/conv2d_3/Conv2D/ReadVariableOp�%model/conv2d_4/BiasAdd/ReadVariableOp�$model/conv2d_4/Conv2D/ReadVariableOp�%model/conv2d_5/BiasAdd/ReadVariableOp�$model/conv2d_5/Conv2D/ReadVariableOp�%model/conv2d_6/BiasAdd/ReadVariableOp�$model/conv2d_6/Conv2D/ReadVariableOp�%model/conv2d_7/BiasAdd/ReadVariableOp�$model/conv2d_7/Conv2D/ReadVariableOp�%model/conv2d_8/BiasAdd/ReadVariableOp�$model/conv2d_8/Conv2D/ReadVariableOp�%model/conv2d_9/BiasAdd/ReadVariableOp�$model/conv2d_9/Conv2D/ReadVariableOp�-model/conv2d_transpose/BiasAdd/ReadVariableOp�6model/conv2d_transpose/conv2d_transpose/ReadVariableOp�/model/conv2d_transpose_1/BiasAdd/ReadVariableOp�8model/conv2d_transpose_1/conv2d_transpose/ReadVariableOpm
model/conv2d/CastCastinput_1*

DstT0*

SrcT0*1
_output_shapes
:������������
"model/conv2d/Conv2D/ReadVariableOpReadVariableOp+model_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
model/conv2d/Conv2D/CastCast*model/conv2d/Conv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*&
_output_shapes
:�
model/conv2d/Conv2DConv2Dmodel/conv2d/Cast:y:0model/conv2d/Conv2D/Cast:y:0*
T0*1
_output_shapes
:�����������*
paddingSAME*
strides
�
#model/conv2d/BiasAdd/ReadVariableOpReadVariableOp,model_conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
model/conv2d/BiasAdd/CastCast+model/conv2d/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:�
model/conv2d/BiasAddBiasAddmodel/conv2d/Conv2D:output:0model/conv2d/BiasAdd/Cast:y:0*
T0*1
_output_shapes
:�����������U
model/conv2d/mul/xConst*
_output_shapes
: *
dtype0*
value
B j�p�
model/conv2d/mulMulmodel/conv2d/mul/x:output:0model/conv2d/BiasAdd:output:0*
T0*1
_output_shapes
:�����������V
model/conv2d/Pow/yConst*
_output_shapes
: *
dtype0*
valueB	 j���
model/conv2d/PowPowmodel/conv2d/BiasAdd:output:0model/conv2d/Pow/y:output:0*
T0*1
_output_shapes
:�����������W
model/conv2d/mul_1/xConst*
_output_shapes
: *
dtype0*
value
B j�S�
model/conv2d/mul_1Mulmodel/conv2d/mul_1/x:output:0model/conv2d/Pow:z:0*
T0*1
_output_shapes
:������������
model/conv2d/addAddV2model/conv2d/BiasAdd:output:0model/conv2d/mul_1:z:0*
T0*1
_output_shapes
:�����������W
model/conv2d/mul_2/xConst*
_output_shapes
: *
dtype0*
value
B j�t�
model/conv2d/mul_2Mulmodel/conv2d/mul_2/x:output:0model/conv2d/add:z:0*
T0*1
_output_shapes
:�����������m
model/conv2d/TanhTanhmodel/conv2d/mul_2:z:0*
T0*1
_output_shapes
:�����������W
model/conv2d/add_1/xConst*
_output_shapes
: *
dtype0*
value
B j�x�
model/conv2d/add_1AddV2model/conv2d/add_1/x:output:0model/conv2d/Tanh:y:0*
T0*1
_output_shapes
:������������
model/conv2d/mul_3Mulmodel/conv2d/mul:z:0model/conv2d/add_1:z:0*
T0*1
_output_shapes
:������������
$model/conv2d_1/Conv2D/ReadVariableOpReadVariableOp-model_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
model/conv2d_1/Conv2D/CastCast,model/conv2d_1/Conv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*&
_output_shapes
:�
model/conv2d_1/Conv2DConv2Dmodel/conv2d/mul_3:z:0model/conv2d_1/Conv2D/Cast:y:0*
T0*1
_output_shapes
:�����������*
paddingSAME*
strides
�
%model/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp.model_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
model/conv2d_1/BiasAdd/CastCast-model/conv2d_1/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:�
model/conv2d_1/BiasAddBiasAddmodel/conv2d_1/Conv2D:output:0model/conv2d_1/BiasAdd/Cast:y:0*
T0*1
_output_shapes
:�����������W
model/conv2d_1/mul/xConst*
_output_shapes
: *
dtype0*
value
B j�p�
model/conv2d_1/mulMulmodel/conv2d_1/mul/x:output:0model/conv2d_1/BiasAdd:output:0*
T0*1
_output_shapes
:�����������X
model/conv2d_1/Pow/yConst*
_output_shapes
: *
dtype0*
valueB	 j���
model/conv2d_1/PowPowmodel/conv2d_1/BiasAdd:output:0model/conv2d_1/Pow/y:output:0*
T0*1
_output_shapes
:�����������Y
model/conv2d_1/mul_1/xConst*
_output_shapes
: *
dtype0*
value
B j�S�
model/conv2d_1/mul_1Mulmodel/conv2d_1/mul_1/x:output:0model/conv2d_1/Pow:z:0*
T0*1
_output_shapes
:������������
model/conv2d_1/addAddV2model/conv2d_1/BiasAdd:output:0model/conv2d_1/mul_1:z:0*
T0*1
_output_shapes
:�����������Y
model/conv2d_1/mul_2/xConst*
_output_shapes
: *
dtype0*
value
B j�t�
model/conv2d_1/mul_2Mulmodel/conv2d_1/mul_2/x:output:0model/conv2d_1/add:z:0*
T0*1
_output_shapes
:�����������q
model/conv2d_1/TanhTanhmodel/conv2d_1/mul_2:z:0*
T0*1
_output_shapes
:�����������Y
model/conv2d_1/add_1/xConst*
_output_shapes
: *
dtype0*
value
B j�x�
model/conv2d_1/add_1AddV2model/conv2d_1/add_1/x:output:0model/conv2d_1/Tanh:y:0*
T0*1
_output_shapes
:������������
model/conv2d_1/mul_3Mulmodel/conv2d_1/mul:z:0model/conv2d_1/add_1:z:0*
T0*1
_output_shapes
:������������
$model/conv2d_2/Conv2D/ReadVariableOpReadVariableOp-model_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
model/conv2d_2/Conv2D/CastCast,model/conv2d_2/Conv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*&
_output_shapes
:�
model/conv2d_2/Conv2DConv2Dmodel/conv2d_1/mul_3:z:0model/conv2d_2/Conv2D/Cast:y:0*
T0*1
_output_shapes
:�����������*
paddingSAME*
strides
�
%model/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp.model_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
model/conv2d_2/BiasAdd/CastCast-model/conv2d_2/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:�
model/conv2d_2/BiasAddBiasAddmodel/conv2d_2/Conv2D:output:0model/conv2d_2/BiasAdd/Cast:y:0*
T0*1
_output_shapes
:�����������W
model/conv2d_2/mul/xConst*
_output_shapes
: *
dtype0*
value
B j�p�
model/conv2d_2/mulMulmodel/conv2d_2/mul/x:output:0model/conv2d_2/BiasAdd:output:0*
T0*1
_output_shapes
:�����������X
model/conv2d_2/Pow/yConst*
_output_shapes
: *
dtype0*
valueB	 j���
model/conv2d_2/PowPowmodel/conv2d_2/BiasAdd:output:0model/conv2d_2/Pow/y:output:0*
T0*1
_output_shapes
:�����������Y
model/conv2d_2/mul_1/xConst*
_output_shapes
: *
dtype0*
value
B j�S�
model/conv2d_2/mul_1Mulmodel/conv2d_2/mul_1/x:output:0model/conv2d_2/Pow:z:0*
T0*1
_output_shapes
:������������
model/conv2d_2/addAddV2model/conv2d_2/BiasAdd:output:0model/conv2d_2/mul_1:z:0*
T0*1
_output_shapes
:�����������Y
model/conv2d_2/mul_2/xConst*
_output_shapes
: *
dtype0*
value
B j�t�
model/conv2d_2/mul_2Mulmodel/conv2d_2/mul_2/x:output:0model/conv2d_2/add:z:0*
T0*1
_output_shapes
:�����������q
model/conv2d_2/TanhTanhmodel/conv2d_2/mul_2:z:0*
T0*1
_output_shapes
:�����������Y
model/conv2d_2/add_1/xConst*
_output_shapes
: *
dtype0*
value
B j�x�
model/conv2d_2/add_1AddV2model/conv2d_2/add_1/x:output:0model/conv2d_2/Tanh:y:0*
T0*1
_output_shapes
:������������
model/conv2d_2/mul_3Mulmodel/conv2d_2/mul:z:0model/conv2d_2/add_1:z:0*
T0*1
_output_shapes
:������������
$model/conv2d_3/Conv2D/ReadVariableOpReadVariableOp-model_conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
model/conv2d_3/Conv2D/CastCast,model/conv2d_3/Conv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*&
_output_shapes
:�
model/conv2d_3/Conv2DConv2Dmodel/conv2d_2/mul_3:z:0model/conv2d_3/Conv2D/Cast:y:0*
T0*1
_output_shapes
:�����������*
paddingSAME*
strides
�
%model/conv2d_3/BiasAdd/ReadVariableOpReadVariableOp.model_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
model/conv2d_3/BiasAdd/CastCast-model/conv2d_3/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:�
model/conv2d_3/BiasAddBiasAddmodel/conv2d_3/Conv2D:output:0model/conv2d_3/BiasAdd/Cast:y:0*
T0*1
_output_shapes
:�����������W
model/conv2d_3/mul/xConst*
_output_shapes
: *
dtype0*
value
B j�p�
model/conv2d_3/mulMulmodel/conv2d_3/mul/x:output:0model/conv2d_3/BiasAdd:output:0*
T0*1
_output_shapes
:�����������X
model/conv2d_3/Pow/yConst*
_output_shapes
: *
dtype0*
valueB	 j���
model/conv2d_3/PowPowmodel/conv2d_3/BiasAdd:output:0model/conv2d_3/Pow/y:output:0*
T0*1
_output_shapes
:�����������Y
model/conv2d_3/mul_1/xConst*
_output_shapes
: *
dtype0*
value
B j�S�
model/conv2d_3/mul_1Mulmodel/conv2d_3/mul_1/x:output:0model/conv2d_3/Pow:z:0*
T0*1
_output_shapes
:������������
model/conv2d_3/addAddV2model/conv2d_3/BiasAdd:output:0model/conv2d_3/mul_1:z:0*
T0*1
_output_shapes
:�����������Y
model/conv2d_3/mul_2/xConst*
_output_shapes
: *
dtype0*
value
B j�t�
model/conv2d_3/mul_2Mulmodel/conv2d_3/mul_2/x:output:0model/conv2d_3/add:z:0*
T0*1
_output_shapes
:�����������q
model/conv2d_3/TanhTanhmodel/conv2d_3/mul_2:z:0*
T0*1
_output_shapes
:�����������Y
model/conv2d_3/add_1/xConst*
_output_shapes
: *
dtype0*
value
B j�x�
model/conv2d_3/add_1AddV2model/conv2d_3/add_1/x:output:0model/conv2d_3/Tanh:y:0*
T0*1
_output_shapes
:������������
model/conv2d_3/mul_3Mulmodel/conv2d_3/mul:z:0model/conv2d_3/add_1:z:0*
T0*1
_output_shapes
:�����������r
model/concatenate/CastCastinput_1*

DstT0*

SrcT0*1
_output_shapes
:�����������_
model/concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
model/concatenate/concatConcatV2model/conv2d_3/mul_3:z:0model/concatenate/Cast:y:0&model/concatenate/concat/axis:output:0*
N*
T0*1
_output_shapes
:������������
$model/conv2d_4/Conv2D/ReadVariableOpReadVariableOp-model_conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
:<*
dtype0�
model/conv2d_4/Conv2D/CastCast,model/conv2d_4/Conv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*&
_output_shapes
:<�
model/conv2d_4/Conv2DConv2D!model/concatenate/concat:output:0model/conv2d_4/Conv2D/Cast:y:0*
T0*1
_output_shapes
:�����������<*
paddingSAME*
strides
�
%model/conv2d_4/BiasAdd/ReadVariableOpReadVariableOp.model_conv2d_4_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype0�
model/conv2d_4/BiasAdd/CastCast-model/conv2d_4/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:<�
model/conv2d_4/BiasAddBiasAddmodel/conv2d_4/Conv2D:output:0model/conv2d_4/BiasAdd/Cast:y:0*
T0*1
_output_shapes
:�����������<W
model/conv2d_4/mul/xConst*
_output_shapes
: *
dtype0*
value
B j�p�
model/conv2d_4/mulMulmodel/conv2d_4/mul/x:output:0model/conv2d_4/BiasAdd:output:0*
T0*1
_output_shapes
:�����������<X
model/conv2d_4/Pow/yConst*
_output_shapes
: *
dtype0*
valueB	 j���
model/conv2d_4/PowPowmodel/conv2d_4/BiasAdd:output:0model/conv2d_4/Pow/y:output:0*
T0*1
_output_shapes
:�����������<Y
model/conv2d_4/mul_1/xConst*
_output_shapes
: *
dtype0*
value
B j�S�
model/conv2d_4/mul_1Mulmodel/conv2d_4/mul_1/x:output:0model/conv2d_4/Pow:z:0*
T0*1
_output_shapes
:�����������<�
model/conv2d_4/addAddV2model/conv2d_4/BiasAdd:output:0model/conv2d_4/mul_1:z:0*
T0*1
_output_shapes
:�����������<Y
model/conv2d_4/mul_2/xConst*
_output_shapes
: *
dtype0*
value
B j�t�
model/conv2d_4/mul_2Mulmodel/conv2d_4/mul_2/x:output:0model/conv2d_4/add:z:0*
T0*1
_output_shapes
:�����������<q
model/conv2d_4/TanhTanhmodel/conv2d_4/mul_2:z:0*
T0*1
_output_shapes
:�����������<Y
model/conv2d_4/add_1/xConst*
_output_shapes
: *
dtype0*
value
B j�x�
model/conv2d_4/add_1AddV2model/conv2d_4/add_1/x:output:0model/conv2d_4/Tanh:y:0*
T0*1
_output_shapes
:�����������<�
model/conv2d_4/mul_3Mulmodel/conv2d_4/mul:z:0model/conv2d_4/add_1:z:0*
T0*1
_output_shapes
:�����������<�
$model/conv2d_5/Conv2D/ReadVariableOpReadVariableOp-model_conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
:<<*
dtype0�
model/conv2d_5/Conv2D/CastCast,model/conv2d_5/Conv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*&
_output_shapes
:<<�
model/conv2d_5/Conv2DConv2Dmodel/conv2d_4/mul_3:z:0model/conv2d_5/Conv2D/Cast:y:0*
T0*1
_output_shapes
:�����������<*
paddingSAME*
strides
�
%model/conv2d_5/BiasAdd/ReadVariableOpReadVariableOp.model_conv2d_5_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype0�
model/conv2d_5/BiasAdd/CastCast-model/conv2d_5/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:<�
model/conv2d_5/BiasAddBiasAddmodel/conv2d_5/Conv2D:output:0model/conv2d_5/BiasAdd/Cast:y:0*
T0*1
_output_shapes
:�����������<W
model/conv2d_5/mul/xConst*
_output_shapes
: *
dtype0*
value
B j�p�
model/conv2d_5/mulMulmodel/conv2d_5/mul/x:output:0model/conv2d_5/BiasAdd:output:0*
T0*1
_output_shapes
:�����������<X
model/conv2d_5/Pow/yConst*
_output_shapes
: *
dtype0*
valueB	 j���
model/conv2d_5/PowPowmodel/conv2d_5/BiasAdd:output:0model/conv2d_5/Pow/y:output:0*
T0*1
_output_shapes
:�����������<Y
model/conv2d_5/mul_1/xConst*
_output_shapes
: *
dtype0*
value
B j�S�
model/conv2d_5/mul_1Mulmodel/conv2d_5/mul_1/x:output:0model/conv2d_5/Pow:z:0*
T0*1
_output_shapes
:�����������<�
model/conv2d_5/addAddV2model/conv2d_5/BiasAdd:output:0model/conv2d_5/mul_1:z:0*
T0*1
_output_shapes
:�����������<Y
model/conv2d_5/mul_2/xConst*
_output_shapes
: *
dtype0*
value
B j�t�
model/conv2d_5/mul_2Mulmodel/conv2d_5/mul_2/x:output:0model/conv2d_5/add:z:0*
T0*1
_output_shapes
:�����������<q
model/conv2d_5/TanhTanhmodel/conv2d_5/mul_2:z:0*
T0*1
_output_shapes
:�����������<Y
model/conv2d_5/add_1/xConst*
_output_shapes
: *
dtype0*
value
B j�x�
model/conv2d_5/add_1AddV2model/conv2d_5/add_1/x:output:0model/conv2d_5/Tanh:y:0*
T0*1
_output_shapes
:�����������<�
model/conv2d_5/mul_3Mulmodel/conv2d_5/mul:z:0model/conv2d_5/add_1:z:0*
T0*1
_output_shapes
:�����������<�
model/average_pooling2d/AvgPoolAvgPoolmodel/conv2d_5/mul_3:z:0*
T0*/
_output_shapes
:���������xx<*
ksize
*
paddingSAME*
strides
�
$model/conv2d_6/Conv2D/ReadVariableOpReadVariableOp-model_conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
:<x*
dtype0�
model/conv2d_6/Conv2D/CastCast,model/conv2d_6/Conv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*&
_output_shapes
:<x�
model/conv2d_6/Conv2DConv2D(model/average_pooling2d/AvgPool:output:0model/conv2d_6/Conv2D/Cast:y:0*
T0*/
_output_shapes
:���������xxx*
paddingSAME*
strides
�
%model/conv2d_6/BiasAdd/ReadVariableOpReadVariableOp.model_conv2d_6_biasadd_readvariableop_resource*
_output_shapes
:x*
dtype0�
model/conv2d_6/BiasAdd/CastCast-model/conv2d_6/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:x�
model/conv2d_6/BiasAddBiasAddmodel/conv2d_6/Conv2D:output:0model/conv2d_6/BiasAdd/Cast:y:0*
T0*/
_output_shapes
:���������xxxW
model/conv2d_6/mul/xConst*
_output_shapes
: *
dtype0*
value
B j�p�
model/conv2d_6/mulMulmodel/conv2d_6/mul/x:output:0model/conv2d_6/BiasAdd:output:0*
T0*/
_output_shapes
:���������xxxX
model/conv2d_6/Pow/yConst*
_output_shapes
: *
dtype0*
valueB	 j���
model/conv2d_6/PowPowmodel/conv2d_6/BiasAdd:output:0model/conv2d_6/Pow/y:output:0*
T0*/
_output_shapes
:���������xxxY
model/conv2d_6/mul_1/xConst*
_output_shapes
: *
dtype0*
value
B j�S�
model/conv2d_6/mul_1Mulmodel/conv2d_6/mul_1/x:output:0model/conv2d_6/Pow:z:0*
T0*/
_output_shapes
:���������xxx�
model/conv2d_6/addAddV2model/conv2d_6/BiasAdd:output:0model/conv2d_6/mul_1:z:0*
T0*/
_output_shapes
:���������xxxY
model/conv2d_6/mul_2/xConst*
_output_shapes
: *
dtype0*
value
B j�t�
model/conv2d_6/mul_2Mulmodel/conv2d_6/mul_2/x:output:0model/conv2d_6/add:z:0*
T0*/
_output_shapes
:���������xxxo
model/conv2d_6/TanhTanhmodel/conv2d_6/mul_2:z:0*
T0*/
_output_shapes
:���������xxxY
model/conv2d_6/add_1/xConst*
_output_shapes
: *
dtype0*
value
B j�x�
model/conv2d_6/add_1AddV2model/conv2d_6/add_1/x:output:0model/conv2d_6/Tanh:y:0*
T0*/
_output_shapes
:���������xxx�
model/conv2d_6/mul_3Mulmodel/conv2d_6/mul:z:0model/conv2d_6/add_1:z:0*
T0*/
_output_shapes
:���������xxx�
$model/conv2d_7/Conv2D/ReadVariableOpReadVariableOp-model_conv2d_7_conv2d_readvariableop_resource*&
_output_shapes
:xx*
dtype0�
model/conv2d_7/Conv2D/CastCast,model/conv2d_7/Conv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*&
_output_shapes
:xx�
model/conv2d_7/Conv2DConv2Dmodel/conv2d_6/mul_3:z:0model/conv2d_7/Conv2D/Cast:y:0*
T0*/
_output_shapes
:���������xxx*
paddingSAME*
strides
�
%model/conv2d_7/BiasAdd/ReadVariableOpReadVariableOp.model_conv2d_7_biasadd_readvariableop_resource*
_output_shapes
:x*
dtype0�
model/conv2d_7/BiasAdd/CastCast-model/conv2d_7/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:x�
model/conv2d_7/BiasAddBiasAddmodel/conv2d_7/Conv2D:output:0model/conv2d_7/BiasAdd/Cast:y:0*
T0*/
_output_shapes
:���������xxxW
model/conv2d_7/mul/xConst*
_output_shapes
: *
dtype0*
value
B j�p�
model/conv2d_7/mulMulmodel/conv2d_7/mul/x:output:0model/conv2d_7/BiasAdd:output:0*
T0*/
_output_shapes
:���������xxxX
model/conv2d_7/Pow/yConst*
_output_shapes
: *
dtype0*
valueB	 j���
model/conv2d_7/PowPowmodel/conv2d_7/BiasAdd:output:0model/conv2d_7/Pow/y:output:0*
T0*/
_output_shapes
:���������xxxY
model/conv2d_7/mul_1/xConst*
_output_shapes
: *
dtype0*
value
B j�S�
model/conv2d_7/mul_1Mulmodel/conv2d_7/mul_1/x:output:0model/conv2d_7/Pow:z:0*
T0*/
_output_shapes
:���������xxx�
model/conv2d_7/addAddV2model/conv2d_7/BiasAdd:output:0model/conv2d_7/mul_1:z:0*
T0*/
_output_shapes
:���������xxxY
model/conv2d_7/mul_2/xConst*
_output_shapes
: *
dtype0*
value
B j�t�
model/conv2d_7/mul_2Mulmodel/conv2d_7/mul_2/x:output:0model/conv2d_7/add:z:0*
T0*/
_output_shapes
:���������xxxo
model/conv2d_7/TanhTanhmodel/conv2d_7/mul_2:z:0*
T0*/
_output_shapes
:���������xxxY
model/conv2d_7/add_1/xConst*
_output_shapes
: *
dtype0*
value
B j�x�
model/conv2d_7/add_1AddV2model/conv2d_7/add_1/x:output:0model/conv2d_7/Tanh:y:0*
T0*/
_output_shapes
:���������xxx�
model/conv2d_7/mul_3Mulmodel/conv2d_7/mul:z:0model/conv2d_7/add_1:z:0*
T0*/
_output_shapes
:���������xxx�
!model/average_pooling2d_1/AvgPoolAvgPoolmodel/conv2d_7/mul_3:z:0*
T0*/
_output_shapes
:���������<<x*
ksize
*
paddingSAME*
strides
�
$model/conv2d_8/Conv2D/ReadVariableOpReadVariableOp-model_conv2d_8_conv2d_readvariableop_resource*'
_output_shapes
:x�*
dtype0�
model/conv2d_8/Conv2D/CastCast,model/conv2d_8/Conv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*'
_output_shapes
:x��
model/conv2d_8/Conv2DConv2D*model/average_pooling2d_1/AvgPool:output:0model/conv2d_8/Conv2D/Cast:y:0*
T0*0
_output_shapes
:���������<<�*
paddingSAME*
strides
�
%model/conv2d_8/BiasAdd/ReadVariableOpReadVariableOp.model_conv2d_8_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
model/conv2d_8/BiasAdd/CastCast-model/conv2d_8/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes	
:��
model/conv2d_8/BiasAddBiasAddmodel/conv2d_8/Conv2D:output:0model/conv2d_8/BiasAdd/Cast:y:0*
T0*0
_output_shapes
:���������<<�W
model/conv2d_8/mul/xConst*
_output_shapes
: *
dtype0*
value
B j�p�
model/conv2d_8/mulMulmodel/conv2d_8/mul/x:output:0model/conv2d_8/BiasAdd:output:0*
T0*0
_output_shapes
:���������<<�X
model/conv2d_8/Pow/yConst*
_output_shapes
: *
dtype0*
valueB	 j���
model/conv2d_8/PowPowmodel/conv2d_8/BiasAdd:output:0model/conv2d_8/Pow/y:output:0*
T0*0
_output_shapes
:���������<<�Y
model/conv2d_8/mul_1/xConst*
_output_shapes
: *
dtype0*
value
B j�S�
model/conv2d_8/mul_1Mulmodel/conv2d_8/mul_1/x:output:0model/conv2d_8/Pow:z:0*
T0*0
_output_shapes
:���������<<��
model/conv2d_8/addAddV2model/conv2d_8/BiasAdd:output:0model/conv2d_8/mul_1:z:0*
T0*0
_output_shapes
:���������<<�Y
model/conv2d_8/mul_2/xConst*
_output_shapes
: *
dtype0*
value
B j�t�
model/conv2d_8/mul_2Mulmodel/conv2d_8/mul_2/x:output:0model/conv2d_8/add:z:0*
T0*0
_output_shapes
:���������<<�p
model/conv2d_8/TanhTanhmodel/conv2d_8/mul_2:z:0*
T0*0
_output_shapes
:���������<<�Y
model/conv2d_8/add_1/xConst*
_output_shapes
: *
dtype0*
value
B j�x�
model/conv2d_8/add_1AddV2model/conv2d_8/add_1/x:output:0model/conv2d_8/Tanh:y:0*
T0*0
_output_shapes
:���������<<��
model/conv2d_8/mul_3Mulmodel/conv2d_8/mul:z:0model/conv2d_8/add_1:z:0*
T0*0
_output_shapes
:���������<<��
$model/conv2d_9/Conv2D/ReadVariableOpReadVariableOp-model_conv2d_9_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
model/conv2d_9/Conv2D/CastCast,model/conv2d_9/Conv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*(
_output_shapes
:���
model/conv2d_9/Conv2DConv2Dmodel/conv2d_8/mul_3:z:0model/conv2d_9/Conv2D/Cast:y:0*
T0*0
_output_shapes
:���������<<�*
paddingSAME*
strides
�
%model/conv2d_9/BiasAdd/ReadVariableOpReadVariableOp.model_conv2d_9_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
model/conv2d_9/BiasAdd/CastCast-model/conv2d_9/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes	
:��
model/conv2d_9/BiasAddBiasAddmodel/conv2d_9/Conv2D:output:0model/conv2d_9/BiasAdd/Cast:y:0*
T0*0
_output_shapes
:���������<<�W
model/conv2d_9/mul/xConst*
_output_shapes
: *
dtype0*
value
B j�p�
model/conv2d_9/mulMulmodel/conv2d_9/mul/x:output:0model/conv2d_9/BiasAdd:output:0*
T0*0
_output_shapes
:���������<<�X
model/conv2d_9/Pow/yConst*
_output_shapes
: *
dtype0*
valueB	 j���
model/conv2d_9/PowPowmodel/conv2d_9/BiasAdd:output:0model/conv2d_9/Pow/y:output:0*
T0*0
_output_shapes
:���������<<�Y
model/conv2d_9/mul_1/xConst*
_output_shapes
: *
dtype0*
value
B j�S�
model/conv2d_9/mul_1Mulmodel/conv2d_9/mul_1/x:output:0model/conv2d_9/Pow:z:0*
T0*0
_output_shapes
:���������<<��
model/conv2d_9/addAddV2model/conv2d_9/BiasAdd:output:0model/conv2d_9/mul_1:z:0*
T0*0
_output_shapes
:���������<<�Y
model/conv2d_9/mul_2/xConst*
_output_shapes
: *
dtype0*
value
B j�t�
model/conv2d_9/mul_2Mulmodel/conv2d_9/mul_2/x:output:0model/conv2d_9/add:z:0*
T0*0
_output_shapes
:���������<<�p
model/conv2d_9/TanhTanhmodel/conv2d_9/mul_2:z:0*
T0*0
_output_shapes
:���������<<�Y
model/conv2d_9/add_1/xConst*
_output_shapes
: *
dtype0*
value
B j�x�
model/conv2d_9/add_1AddV2model/conv2d_9/add_1/x:output:0model/conv2d_9/Tanh:y:0*
T0*0
_output_shapes
:���������<<��
model/conv2d_9/mul_3Mulmodel/conv2d_9/mul:z:0model/conv2d_9/add_1:z:0*
T0*0
_output_shapes
:���������<<��
%model/conv2d_10/Conv2D/ReadVariableOpReadVariableOp.model_conv2d_10_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
model/conv2d_10/Conv2D/CastCast-model/conv2d_10/Conv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*(
_output_shapes
:���
model/conv2d_10/Conv2DConv2Dmodel/conv2d_9/mul_3:z:0model/conv2d_10/Conv2D/Cast:y:0*
T0*0
_output_shapes
:���������<<�*
paddingSAME*
strides
�
&model/conv2d_10/BiasAdd/ReadVariableOpReadVariableOp/model_conv2d_10_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
model/conv2d_10/BiasAdd/CastCast.model/conv2d_10/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes	
:��
model/conv2d_10/BiasAddBiasAddmodel/conv2d_10/Conv2D:output:0 model/conv2d_10/BiasAdd/Cast:y:0*
T0*0
_output_shapes
:���������<<�X
model/conv2d_10/mul/xConst*
_output_shapes
: *
dtype0*
value
B j�p�
model/conv2d_10/mulMulmodel/conv2d_10/mul/x:output:0 model/conv2d_10/BiasAdd:output:0*
T0*0
_output_shapes
:���������<<�Y
model/conv2d_10/Pow/yConst*
_output_shapes
: *
dtype0*
valueB	 j���
model/conv2d_10/PowPow model/conv2d_10/BiasAdd:output:0model/conv2d_10/Pow/y:output:0*
T0*0
_output_shapes
:���������<<�Z
model/conv2d_10/mul_1/xConst*
_output_shapes
: *
dtype0*
value
B j�S�
model/conv2d_10/mul_1Mul model/conv2d_10/mul_1/x:output:0model/conv2d_10/Pow:z:0*
T0*0
_output_shapes
:���������<<��
model/conv2d_10/addAddV2 model/conv2d_10/BiasAdd:output:0model/conv2d_10/mul_1:z:0*
T0*0
_output_shapes
:���������<<�Z
model/conv2d_10/mul_2/xConst*
_output_shapes
: *
dtype0*
value
B j�t�
model/conv2d_10/mul_2Mul model/conv2d_10/mul_2/x:output:0model/conv2d_10/add:z:0*
T0*0
_output_shapes
:���������<<�r
model/conv2d_10/TanhTanhmodel/conv2d_10/mul_2:z:0*
T0*0
_output_shapes
:���������<<�Z
model/conv2d_10/add_1/xConst*
_output_shapes
: *
dtype0*
value
B j�x�
model/conv2d_10/add_1AddV2 model/conv2d_10/add_1/x:output:0model/conv2d_10/Tanh:y:0*
T0*0
_output_shapes
:���������<<��
model/conv2d_10/mul_3Mulmodel/conv2d_10/mul:z:0model/conv2d_10/add_1:z:0*
T0*0
_output_shapes
:���������<<��
%model/conv2d_11/Conv2D/ReadVariableOpReadVariableOp.model_conv2d_11_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
model/conv2d_11/Conv2D/CastCast-model/conv2d_11/Conv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*(
_output_shapes
:���
model/conv2d_11/Conv2DConv2Dmodel/conv2d_10/mul_3:z:0model/conv2d_11/Conv2D/Cast:y:0*
T0*0
_output_shapes
:���������<<�*
paddingSAME*
strides
�
&model/conv2d_11/BiasAdd/ReadVariableOpReadVariableOp/model_conv2d_11_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
model/conv2d_11/BiasAdd/CastCast.model/conv2d_11/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes	
:��
model/conv2d_11/BiasAddBiasAddmodel/conv2d_11/Conv2D:output:0 model/conv2d_11/BiasAdd/Cast:y:0*
T0*0
_output_shapes
:���������<<�X
model/conv2d_11/mul/xConst*
_output_shapes
: *
dtype0*
value
B j�p�
model/conv2d_11/mulMulmodel/conv2d_11/mul/x:output:0 model/conv2d_11/BiasAdd:output:0*
T0*0
_output_shapes
:���������<<�Y
model/conv2d_11/Pow/yConst*
_output_shapes
: *
dtype0*
valueB	 j���
model/conv2d_11/PowPow model/conv2d_11/BiasAdd:output:0model/conv2d_11/Pow/y:output:0*
T0*0
_output_shapes
:���������<<�Z
model/conv2d_11/mul_1/xConst*
_output_shapes
: *
dtype0*
value
B j�S�
model/conv2d_11/mul_1Mul model/conv2d_11/mul_1/x:output:0model/conv2d_11/Pow:z:0*
T0*0
_output_shapes
:���������<<��
model/conv2d_11/addAddV2 model/conv2d_11/BiasAdd:output:0model/conv2d_11/mul_1:z:0*
T0*0
_output_shapes
:���������<<�Z
model/conv2d_11/mul_2/xConst*
_output_shapes
: *
dtype0*
value
B j�t�
model/conv2d_11/mul_2Mul model/conv2d_11/mul_2/x:output:0model/conv2d_11/add:z:0*
T0*0
_output_shapes
:���������<<�r
model/conv2d_11/TanhTanhmodel/conv2d_11/mul_2:z:0*
T0*0
_output_shapes
:���������<<�Z
model/conv2d_11/add_1/xConst*
_output_shapes
: *
dtype0*
value
B j�x�
model/conv2d_11/add_1AddV2 model/conv2d_11/add_1/x:output:0model/conv2d_11/Tanh:y:0*
T0*0
_output_shapes
:���������<<��
model/conv2d_11/mul_3Mulmodel/conv2d_11/mul:z:0model/conv2d_11/add_1:z:0*
T0*0
_output_shapes
:���������<<�e
model/conv2d_transpose/ShapeShapemodel/conv2d_11/mul_3:z:0*
T0*
_output_shapes
:t
*model/conv2d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,model/conv2d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:v
,model/conv2d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
$model/conv2d_transpose/strided_sliceStridedSlice%model/conv2d_transpose/Shape:output:03model/conv2d_transpose/strided_slice/stack:output:05model/conv2d_transpose/strided_slice/stack_1:output:05model/conv2d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask`
model/conv2d_transpose/stack/1Const*
_output_shapes
: *
dtype0*
value	B :x`
model/conv2d_transpose/stack/2Const*
_output_shapes
: *
dtype0*
value	B :x`
model/conv2d_transpose/stack/3Const*
_output_shapes
: *
dtype0*
value	B :x�
model/conv2d_transpose/stackPack-model/conv2d_transpose/strided_slice:output:0'model/conv2d_transpose/stack/1:output:0'model/conv2d_transpose/stack/2:output:0'model/conv2d_transpose/stack/3:output:0*
N*
T0*
_output_shapes
:v
,model/conv2d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: x
.model/conv2d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.model/conv2d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
&model/conv2d_transpose/strided_slice_1StridedSlice%model/conv2d_transpose/stack:output:05model/conv2d_transpose/strided_slice_1/stack:output:07model/conv2d_transpose/strided_slice_1/stack_1:output:07model/conv2d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
6model/conv2d_transpose/conv2d_transpose/ReadVariableOpReadVariableOp?model_conv2d_transpose_conv2d_transpose_readvariableop_resource*'
_output_shapes
:x�*
dtype0�
,model/conv2d_transpose/conv2d_transpose/CastCast>model/conv2d_transpose/conv2d_transpose/ReadVariableOp:value:0*

DstT0*

SrcT0*'
_output_shapes
:x��
'model/conv2d_transpose/conv2d_transposeConv2DBackpropInput%model/conv2d_transpose/stack:output:00model/conv2d_transpose/conv2d_transpose/Cast:y:0model/conv2d_11/mul_3:z:0*
T0*/
_output_shapes
:���������xxx*
paddingSAME*
strides
�
-model/conv2d_transpose/BiasAdd/ReadVariableOpReadVariableOp6model_conv2d_transpose_biasadd_readvariableop_resource*
_output_shapes
:x*
dtype0�
#model/conv2d_transpose/BiasAdd/CastCast5model/conv2d_transpose/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:x�
model/conv2d_transpose/BiasAddBiasAdd0model/conv2d_transpose/conv2d_transpose:output:0'model/conv2d_transpose/BiasAdd/Cast:y:0*
T0*/
_output_shapes
:���������xxx_
model/conv2d_transpose/mul/xConst*
_output_shapes
: *
dtype0*
value
B j�p�
model/conv2d_transpose/mulMul%model/conv2d_transpose/mul/x:output:0'model/conv2d_transpose/BiasAdd:output:0*
T0*/
_output_shapes
:���������xxx`
model/conv2d_transpose/Pow/yConst*
_output_shapes
: *
dtype0*
valueB	 j���
model/conv2d_transpose/PowPow'model/conv2d_transpose/BiasAdd:output:0%model/conv2d_transpose/Pow/y:output:0*
T0*/
_output_shapes
:���������xxxa
model/conv2d_transpose/mul_1/xConst*
_output_shapes
: *
dtype0*
value
B j�S�
model/conv2d_transpose/mul_1Mul'model/conv2d_transpose/mul_1/x:output:0model/conv2d_transpose/Pow:z:0*
T0*/
_output_shapes
:���������xxx�
model/conv2d_transpose/addAddV2'model/conv2d_transpose/BiasAdd:output:0 model/conv2d_transpose/mul_1:z:0*
T0*/
_output_shapes
:���������xxxa
model/conv2d_transpose/mul_2/xConst*
_output_shapes
: *
dtype0*
value
B j�t�
model/conv2d_transpose/mul_2Mul'model/conv2d_transpose/mul_2/x:output:0model/conv2d_transpose/add:z:0*
T0*/
_output_shapes
:���������xxx
model/conv2d_transpose/TanhTanh model/conv2d_transpose/mul_2:z:0*
T0*/
_output_shapes
:���������xxxa
model/conv2d_transpose/add_1/xConst*
_output_shapes
: *
dtype0*
value
B j�x�
model/conv2d_transpose/add_1AddV2'model/conv2d_transpose/add_1/x:output:0model/conv2d_transpose/Tanh:y:0*
T0*/
_output_shapes
:���������xxx�
model/conv2d_transpose/mul_3Mulmodel/conv2d_transpose/mul:z:0 model/conv2d_transpose/add_1:z:0*
T0*/
_output_shapes
:���������xxx�
model/add/addAddV2 model/conv2d_transpose/mul_3:z:0model/conv2d_7/mul_3:z:0*
T0*/
_output_shapes
:���������xxx�
%model/conv2d_12/Conv2D/ReadVariableOpReadVariableOp.model_conv2d_12_conv2d_readvariableop_resource*&
_output_shapes
:xx*
dtype0�
model/conv2d_12/Conv2D/CastCast-model/conv2d_12/Conv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*&
_output_shapes
:xx�
model/conv2d_12/Conv2DConv2Dmodel/add/add:z:0model/conv2d_12/Conv2D/Cast:y:0*
T0*/
_output_shapes
:���������xxx*
paddingSAME*
strides
�
&model/conv2d_12/BiasAdd/ReadVariableOpReadVariableOp/model_conv2d_12_biasadd_readvariableop_resource*
_output_shapes
:x*
dtype0�
model/conv2d_12/BiasAdd/CastCast.model/conv2d_12/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:x�
model/conv2d_12/BiasAddBiasAddmodel/conv2d_12/Conv2D:output:0 model/conv2d_12/BiasAdd/Cast:y:0*
T0*/
_output_shapes
:���������xxxX
model/conv2d_12/mul/xConst*
_output_shapes
: *
dtype0*
value
B j�p�
model/conv2d_12/mulMulmodel/conv2d_12/mul/x:output:0 model/conv2d_12/BiasAdd:output:0*
T0*/
_output_shapes
:���������xxxY
model/conv2d_12/Pow/yConst*
_output_shapes
: *
dtype0*
valueB	 j���
model/conv2d_12/PowPow model/conv2d_12/BiasAdd:output:0model/conv2d_12/Pow/y:output:0*
T0*/
_output_shapes
:���������xxxZ
model/conv2d_12/mul_1/xConst*
_output_shapes
: *
dtype0*
value
B j�S�
model/conv2d_12/mul_1Mul model/conv2d_12/mul_1/x:output:0model/conv2d_12/Pow:z:0*
T0*/
_output_shapes
:���������xxx�
model/conv2d_12/addAddV2 model/conv2d_12/BiasAdd:output:0model/conv2d_12/mul_1:z:0*
T0*/
_output_shapes
:���������xxxZ
model/conv2d_12/mul_2/xConst*
_output_shapes
: *
dtype0*
value
B j�t�
model/conv2d_12/mul_2Mul model/conv2d_12/mul_2/x:output:0model/conv2d_12/add:z:0*
T0*/
_output_shapes
:���������xxxq
model/conv2d_12/TanhTanhmodel/conv2d_12/mul_2:z:0*
T0*/
_output_shapes
:���������xxxZ
model/conv2d_12/add_1/xConst*
_output_shapes
: *
dtype0*
value
B j�x�
model/conv2d_12/add_1AddV2 model/conv2d_12/add_1/x:output:0model/conv2d_12/Tanh:y:0*
T0*/
_output_shapes
:���������xxx�
model/conv2d_12/mul_3Mulmodel/conv2d_12/mul:z:0model/conv2d_12/add_1:z:0*
T0*/
_output_shapes
:���������xxx�
%model/conv2d_13/Conv2D/ReadVariableOpReadVariableOp.model_conv2d_13_conv2d_readvariableop_resource*&
_output_shapes
:xx*
dtype0�
model/conv2d_13/Conv2D/CastCast-model/conv2d_13/Conv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*&
_output_shapes
:xx�
model/conv2d_13/Conv2DConv2Dmodel/conv2d_12/mul_3:z:0model/conv2d_13/Conv2D/Cast:y:0*
T0*/
_output_shapes
:���������xxx*
paddingSAME*
strides
�
&model/conv2d_13/BiasAdd/ReadVariableOpReadVariableOp/model_conv2d_13_biasadd_readvariableop_resource*
_output_shapes
:x*
dtype0�
model/conv2d_13/BiasAdd/CastCast.model/conv2d_13/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:x�
model/conv2d_13/BiasAddBiasAddmodel/conv2d_13/Conv2D:output:0 model/conv2d_13/BiasAdd/Cast:y:0*
T0*/
_output_shapes
:���������xxxX
model/conv2d_13/mul/xConst*
_output_shapes
: *
dtype0*
value
B j�p�
model/conv2d_13/mulMulmodel/conv2d_13/mul/x:output:0 model/conv2d_13/BiasAdd:output:0*
T0*/
_output_shapes
:���������xxxY
model/conv2d_13/Pow/yConst*
_output_shapes
: *
dtype0*
valueB	 j���
model/conv2d_13/PowPow model/conv2d_13/BiasAdd:output:0model/conv2d_13/Pow/y:output:0*
T0*/
_output_shapes
:���������xxxZ
model/conv2d_13/mul_1/xConst*
_output_shapes
: *
dtype0*
value
B j�S�
model/conv2d_13/mul_1Mul model/conv2d_13/mul_1/x:output:0model/conv2d_13/Pow:z:0*
T0*/
_output_shapes
:���������xxx�
model/conv2d_13/addAddV2 model/conv2d_13/BiasAdd:output:0model/conv2d_13/mul_1:z:0*
T0*/
_output_shapes
:���������xxxZ
model/conv2d_13/mul_2/xConst*
_output_shapes
: *
dtype0*
value
B j�t�
model/conv2d_13/mul_2Mul model/conv2d_13/mul_2/x:output:0model/conv2d_13/add:z:0*
T0*/
_output_shapes
:���������xxxq
model/conv2d_13/TanhTanhmodel/conv2d_13/mul_2:z:0*
T0*/
_output_shapes
:���������xxxZ
model/conv2d_13/add_1/xConst*
_output_shapes
: *
dtype0*
value
B j�x�
model/conv2d_13/add_1AddV2 model/conv2d_13/add_1/x:output:0model/conv2d_13/Tanh:y:0*
T0*/
_output_shapes
:���������xxx�
model/conv2d_13/mul_3Mulmodel/conv2d_13/mul:z:0model/conv2d_13/add_1:z:0*
T0*/
_output_shapes
:���������xxxg
model/conv2d_transpose_1/ShapeShapemodel/conv2d_13/mul_3:z:0*
T0*
_output_shapes
:v
,model/conv2d_transpose_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: x
.model/conv2d_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.model/conv2d_transpose_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
&model/conv2d_transpose_1/strided_sliceStridedSlice'model/conv2d_transpose_1/Shape:output:05model/conv2d_transpose_1/strided_slice/stack:output:07model/conv2d_transpose_1/strided_slice/stack_1:output:07model/conv2d_transpose_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskc
 model/conv2d_transpose_1/stack/1Const*
_output_shapes
: *
dtype0*
value
B :�c
 model/conv2d_transpose_1/stack/2Const*
_output_shapes
: *
dtype0*
value
B :�b
 model/conv2d_transpose_1/stack/3Const*
_output_shapes
: *
dtype0*
value	B :<�
model/conv2d_transpose_1/stackPack/model/conv2d_transpose_1/strided_slice:output:0)model/conv2d_transpose_1/stack/1:output:0)model/conv2d_transpose_1/stack/2:output:0)model/conv2d_transpose_1/stack/3:output:0*
N*
T0*
_output_shapes
:x
.model/conv2d_transpose_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: z
0model/conv2d_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0model/conv2d_transpose_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
(model/conv2d_transpose_1/strided_slice_1StridedSlice'model/conv2d_transpose_1/stack:output:07model/conv2d_transpose_1/strided_slice_1/stack:output:09model/conv2d_transpose_1/strided_slice_1/stack_1:output:09model/conv2d_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
8model/conv2d_transpose_1/conv2d_transpose/ReadVariableOpReadVariableOpAmodel_conv2d_transpose_1_conv2d_transpose_readvariableop_resource*&
_output_shapes
:<x*
dtype0�
.model/conv2d_transpose_1/conv2d_transpose/CastCast@model/conv2d_transpose_1/conv2d_transpose/ReadVariableOp:value:0*

DstT0*

SrcT0*&
_output_shapes
:<x�
)model/conv2d_transpose_1/conv2d_transposeConv2DBackpropInput'model/conv2d_transpose_1/stack:output:02model/conv2d_transpose_1/conv2d_transpose/Cast:y:0model/conv2d_13/mul_3:z:0*
T0*1
_output_shapes
:�����������<*
paddingSAME*
strides
�
/model/conv2d_transpose_1/BiasAdd/ReadVariableOpReadVariableOp8model_conv2d_transpose_1_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype0�
%model/conv2d_transpose_1/BiasAdd/CastCast7model/conv2d_transpose_1/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:<�
 model/conv2d_transpose_1/BiasAddBiasAdd2model/conv2d_transpose_1/conv2d_transpose:output:0)model/conv2d_transpose_1/BiasAdd/Cast:y:0*
T0*1
_output_shapes
:�����������<a
model/conv2d_transpose_1/mul/xConst*
_output_shapes
: *
dtype0*
value
B j�p�
model/conv2d_transpose_1/mulMul'model/conv2d_transpose_1/mul/x:output:0)model/conv2d_transpose_1/BiasAdd:output:0*
T0*1
_output_shapes
:�����������<b
model/conv2d_transpose_1/Pow/yConst*
_output_shapes
: *
dtype0*
valueB	 j���
model/conv2d_transpose_1/PowPow)model/conv2d_transpose_1/BiasAdd:output:0'model/conv2d_transpose_1/Pow/y:output:0*
T0*1
_output_shapes
:�����������<c
 model/conv2d_transpose_1/mul_1/xConst*
_output_shapes
: *
dtype0*
value
B j�S�
model/conv2d_transpose_1/mul_1Mul)model/conv2d_transpose_1/mul_1/x:output:0 model/conv2d_transpose_1/Pow:z:0*
T0*1
_output_shapes
:�����������<�
model/conv2d_transpose_1/addAddV2)model/conv2d_transpose_1/BiasAdd:output:0"model/conv2d_transpose_1/mul_1:z:0*
T0*1
_output_shapes
:�����������<c
 model/conv2d_transpose_1/mul_2/xConst*
_output_shapes
: *
dtype0*
value
B j�t�
model/conv2d_transpose_1/mul_2Mul)model/conv2d_transpose_1/mul_2/x:output:0 model/conv2d_transpose_1/add:z:0*
T0*1
_output_shapes
:�����������<�
model/conv2d_transpose_1/TanhTanh"model/conv2d_transpose_1/mul_2:z:0*
T0*1
_output_shapes
:�����������<c
 model/conv2d_transpose_1/add_1/xConst*
_output_shapes
: *
dtype0*
value
B j�x�
model/conv2d_transpose_1/add_1AddV2)model/conv2d_transpose_1/add_1/x:output:0!model/conv2d_transpose_1/Tanh:y:0*
T0*1
_output_shapes
:�����������<�
model/conv2d_transpose_1/mul_3Mul model/conv2d_transpose_1/mul:z:0"model/conv2d_transpose_1/add_1:z:0*
T0*1
_output_shapes
:�����������<�
model/add_1/addAddV2"model/conv2d_transpose_1/mul_3:z:0model/conv2d_5/mul_3:z:0*
T0*1
_output_shapes
:�����������<�
%model/conv2d_14/Conv2D/ReadVariableOpReadVariableOp.model_conv2d_14_conv2d_readvariableop_resource*&
_output_shapes
:<<*
dtype0�
model/conv2d_14/Conv2D/CastCast-model/conv2d_14/Conv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*&
_output_shapes
:<<�
model/conv2d_14/Conv2DConv2Dmodel/add_1/add:z:0model/conv2d_14/Conv2D/Cast:y:0*
T0*1
_output_shapes
:�����������<*
paddingSAME*
strides
�
&model/conv2d_14/BiasAdd/ReadVariableOpReadVariableOp/model_conv2d_14_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype0�
model/conv2d_14/BiasAdd/CastCast.model/conv2d_14/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:<�
model/conv2d_14/BiasAddBiasAddmodel/conv2d_14/Conv2D:output:0 model/conv2d_14/BiasAdd/Cast:y:0*
T0*1
_output_shapes
:�����������<X
model/conv2d_14/mul/xConst*
_output_shapes
: *
dtype0*
value
B j�p�
model/conv2d_14/mulMulmodel/conv2d_14/mul/x:output:0 model/conv2d_14/BiasAdd:output:0*
T0*1
_output_shapes
:�����������<Y
model/conv2d_14/Pow/yConst*
_output_shapes
: *
dtype0*
valueB	 j���
model/conv2d_14/PowPow model/conv2d_14/BiasAdd:output:0model/conv2d_14/Pow/y:output:0*
T0*1
_output_shapes
:�����������<Z
model/conv2d_14/mul_1/xConst*
_output_shapes
: *
dtype0*
value
B j�S�
model/conv2d_14/mul_1Mul model/conv2d_14/mul_1/x:output:0model/conv2d_14/Pow:z:0*
T0*1
_output_shapes
:�����������<�
model/conv2d_14/addAddV2 model/conv2d_14/BiasAdd:output:0model/conv2d_14/mul_1:z:0*
T0*1
_output_shapes
:�����������<Z
model/conv2d_14/mul_2/xConst*
_output_shapes
: *
dtype0*
value
B j�t�
model/conv2d_14/mul_2Mul model/conv2d_14/mul_2/x:output:0model/conv2d_14/add:z:0*
T0*1
_output_shapes
:�����������<s
model/conv2d_14/TanhTanhmodel/conv2d_14/mul_2:z:0*
T0*1
_output_shapes
:�����������<Z
model/conv2d_14/add_1/xConst*
_output_shapes
: *
dtype0*
value
B j�x�
model/conv2d_14/add_1AddV2 model/conv2d_14/add_1/x:output:0model/conv2d_14/Tanh:y:0*
T0*1
_output_shapes
:�����������<�
model/conv2d_14/mul_3Mulmodel/conv2d_14/mul:z:0model/conv2d_14/add_1:z:0*
T0*1
_output_shapes
:�����������<�
%model/conv2d_15/Conv2D/ReadVariableOpReadVariableOp.model_conv2d_15_conv2d_readvariableop_resource*&
_output_shapes
:<<*
dtype0�
model/conv2d_15/Conv2D/CastCast-model/conv2d_15/Conv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*&
_output_shapes
:<<�
model/conv2d_15/Conv2DConv2Dmodel/conv2d_14/mul_3:z:0model/conv2d_15/Conv2D/Cast:y:0*
T0*1
_output_shapes
:�����������<*
paddingSAME*
strides
�
&model/conv2d_15/BiasAdd/ReadVariableOpReadVariableOp/model_conv2d_15_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype0�
model/conv2d_15/BiasAdd/CastCast.model/conv2d_15/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:<�
model/conv2d_15/BiasAddBiasAddmodel/conv2d_15/Conv2D:output:0 model/conv2d_15/BiasAdd/Cast:y:0*
T0*1
_output_shapes
:�����������<X
model/conv2d_15/mul/xConst*
_output_shapes
: *
dtype0*
value
B j�p�
model/conv2d_15/mulMulmodel/conv2d_15/mul/x:output:0 model/conv2d_15/BiasAdd:output:0*
T0*1
_output_shapes
:�����������<Y
model/conv2d_15/Pow/yConst*
_output_shapes
: *
dtype0*
valueB	 j���
model/conv2d_15/PowPow model/conv2d_15/BiasAdd:output:0model/conv2d_15/Pow/y:output:0*
T0*1
_output_shapes
:�����������<Z
model/conv2d_15/mul_1/xConst*
_output_shapes
: *
dtype0*
value
B j�S�
model/conv2d_15/mul_1Mul model/conv2d_15/mul_1/x:output:0model/conv2d_15/Pow:z:0*
T0*1
_output_shapes
:�����������<�
model/conv2d_15/addAddV2 model/conv2d_15/BiasAdd:output:0model/conv2d_15/mul_1:z:0*
T0*1
_output_shapes
:�����������<Z
model/conv2d_15/mul_2/xConst*
_output_shapes
: *
dtype0*
value
B j�t�
model/conv2d_15/mul_2Mul model/conv2d_15/mul_2/x:output:0model/conv2d_15/add:z:0*
T0*1
_output_shapes
:�����������<s
model/conv2d_15/TanhTanhmodel/conv2d_15/mul_2:z:0*
T0*1
_output_shapes
:�����������<Z
model/conv2d_15/add_1/xConst*
_output_shapes
: *
dtype0*
value
B j�x�
model/conv2d_15/add_1AddV2 model/conv2d_15/add_1/x:output:0model/conv2d_15/Tanh:y:0*
T0*1
_output_shapes
:�����������<�
model/conv2d_15/mul_3Mulmodel/conv2d_15/mul:z:0model/conv2d_15/add_1:z:0*
T0*1
_output_shapes
:�����������<�
%model/conv2d_16/Conv2D/ReadVariableOpReadVariableOp.model_conv2d_16_conv2d_readvariableop_resource*&
_output_shapes
:<*
dtype0�
model/conv2d_16/Conv2D/CastCast-model/conv2d_16/Conv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*&
_output_shapes
:<�
model/conv2d_16/Conv2DConv2Dmodel/conv2d_15/mul_3:z:0model/conv2d_16/Conv2D/Cast:y:0*
T0*1
_output_shapes
:�����������*
paddingSAME*
strides
�
&model/conv2d_16/BiasAdd/ReadVariableOpReadVariableOp/model_conv2d_16_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
model/conv2d_16/BiasAdd/CastCast.model/conv2d_16/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:�
model/conv2d_16/BiasAddBiasAddmodel/conv2d_16/Conv2D:output:0 model/conv2d_16/BiasAdd/Cast:y:0*
T0*1
_output_shapes
:������������
model/add_2/CastCast model/conv2d_16/BiasAdd:output:0*

DstT0*

SrcT0*1
_output_shapes
:�����������s
model/add_2/addAddV2model/add_2/Cast:y:0input_1*
T0*1
_output_shapes
:�����������l
IdentityIdentitymodel/add_2/add:z:0^NoOp*
T0*1
_output_shapes
:������������
NoOpNoOp$^model/conv2d/BiasAdd/ReadVariableOp#^model/conv2d/Conv2D/ReadVariableOp&^model/conv2d_1/BiasAdd/ReadVariableOp%^model/conv2d_1/Conv2D/ReadVariableOp'^model/conv2d_10/BiasAdd/ReadVariableOp&^model/conv2d_10/Conv2D/ReadVariableOp'^model/conv2d_11/BiasAdd/ReadVariableOp&^model/conv2d_11/Conv2D/ReadVariableOp'^model/conv2d_12/BiasAdd/ReadVariableOp&^model/conv2d_12/Conv2D/ReadVariableOp'^model/conv2d_13/BiasAdd/ReadVariableOp&^model/conv2d_13/Conv2D/ReadVariableOp'^model/conv2d_14/BiasAdd/ReadVariableOp&^model/conv2d_14/Conv2D/ReadVariableOp'^model/conv2d_15/BiasAdd/ReadVariableOp&^model/conv2d_15/Conv2D/ReadVariableOp'^model/conv2d_16/BiasAdd/ReadVariableOp&^model/conv2d_16/Conv2D/ReadVariableOp&^model/conv2d_2/BiasAdd/ReadVariableOp%^model/conv2d_2/Conv2D/ReadVariableOp&^model/conv2d_3/BiasAdd/ReadVariableOp%^model/conv2d_3/Conv2D/ReadVariableOp&^model/conv2d_4/BiasAdd/ReadVariableOp%^model/conv2d_4/Conv2D/ReadVariableOp&^model/conv2d_5/BiasAdd/ReadVariableOp%^model/conv2d_5/Conv2D/ReadVariableOp&^model/conv2d_6/BiasAdd/ReadVariableOp%^model/conv2d_6/Conv2D/ReadVariableOp&^model/conv2d_7/BiasAdd/ReadVariableOp%^model/conv2d_7/Conv2D/ReadVariableOp&^model/conv2d_8/BiasAdd/ReadVariableOp%^model/conv2d_8/Conv2D/ReadVariableOp&^model/conv2d_9/BiasAdd/ReadVariableOp%^model/conv2d_9/Conv2D/ReadVariableOp.^model/conv2d_transpose/BiasAdd/ReadVariableOp7^model/conv2d_transpose/conv2d_transpose/ReadVariableOp0^model/conv2d_transpose_1/BiasAdd/ReadVariableOp9^model/conv2d_transpose_1/conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*|
_input_shapesk
i:�����������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2J
#model/conv2d/BiasAdd/ReadVariableOp#model/conv2d/BiasAdd/ReadVariableOp2H
"model/conv2d/Conv2D/ReadVariableOp"model/conv2d/Conv2D/ReadVariableOp2N
%model/conv2d_1/BiasAdd/ReadVariableOp%model/conv2d_1/BiasAdd/ReadVariableOp2L
$model/conv2d_1/Conv2D/ReadVariableOp$model/conv2d_1/Conv2D/ReadVariableOp2P
&model/conv2d_10/BiasAdd/ReadVariableOp&model/conv2d_10/BiasAdd/ReadVariableOp2N
%model/conv2d_10/Conv2D/ReadVariableOp%model/conv2d_10/Conv2D/ReadVariableOp2P
&model/conv2d_11/BiasAdd/ReadVariableOp&model/conv2d_11/BiasAdd/ReadVariableOp2N
%model/conv2d_11/Conv2D/ReadVariableOp%model/conv2d_11/Conv2D/ReadVariableOp2P
&model/conv2d_12/BiasAdd/ReadVariableOp&model/conv2d_12/BiasAdd/ReadVariableOp2N
%model/conv2d_12/Conv2D/ReadVariableOp%model/conv2d_12/Conv2D/ReadVariableOp2P
&model/conv2d_13/BiasAdd/ReadVariableOp&model/conv2d_13/BiasAdd/ReadVariableOp2N
%model/conv2d_13/Conv2D/ReadVariableOp%model/conv2d_13/Conv2D/ReadVariableOp2P
&model/conv2d_14/BiasAdd/ReadVariableOp&model/conv2d_14/BiasAdd/ReadVariableOp2N
%model/conv2d_14/Conv2D/ReadVariableOp%model/conv2d_14/Conv2D/ReadVariableOp2P
&model/conv2d_15/BiasAdd/ReadVariableOp&model/conv2d_15/BiasAdd/ReadVariableOp2N
%model/conv2d_15/Conv2D/ReadVariableOp%model/conv2d_15/Conv2D/ReadVariableOp2P
&model/conv2d_16/BiasAdd/ReadVariableOp&model/conv2d_16/BiasAdd/ReadVariableOp2N
%model/conv2d_16/Conv2D/ReadVariableOp%model/conv2d_16/Conv2D/ReadVariableOp2N
%model/conv2d_2/BiasAdd/ReadVariableOp%model/conv2d_2/BiasAdd/ReadVariableOp2L
$model/conv2d_2/Conv2D/ReadVariableOp$model/conv2d_2/Conv2D/ReadVariableOp2N
%model/conv2d_3/BiasAdd/ReadVariableOp%model/conv2d_3/BiasAdd/ReadVariableOp2L
$model/conv2d_3/Conv2D/ReadVariableOp$model/conv2d_3/Conv2D/ReadVariableOp2N
%model/conv2d_4/BiasAdd/ReadVariableOp%model/conv2d_4/BiasAdd/ReadVariableOp2L
$model/conv2d_4/Conv2D/ReadVariableOp$model/conv2d_4/Conv2D/ReadVariableOp2N
%model/conv2d_5/BiasAdd/ReadVariableOp%model/conv2d_5/BiasAdd/ReadVariableOp2L
$model/conv2d_5/Conv2D/ReadVariableOp$model/conv2d_5/Conv2D/ReadVariableOp2N
%model/conv2d_6/BiasAdd/ReadVariableOp%model/conv2d_6/BiasAdd/ReadVariableOp2L
$model/conv2d_6/Conv2D/ReadVariableOp$model/conv2d_6/Conv2D/ReadVariableOp2N
%model/conv2d_7/BiasAdd/ReadVariableOp%model/conv2d_7/BiasAdd/ReadVariableOp2L
$model/conv2d_7/Conv2D/ReadVariableOp$model/conv2d_7/Conv2D/ReadVariableOp2N
%model/conv2d_8/BiasAdd/ReadVariableOp%model/conv2d_8/BiasAdd/ReadVariableOp2L
$model/conv2d_8/Conv2D/ReadVariableOp$model/conv2d_8/Conv2D/ReadVariableOp2N
%model/conv2d_9/BiasAdd/ReadVariableOp%model/conv2d_9/BiasAdd/ReadVariableOp2L
$model/conv2d_9/Conv2D/ReadVariableOp$model/conv2d_9/Conv2D/ReadVariableOp2^
-model/conv2d_transpose/BiasAdd/ReadVariableOp-model/conv2d_transpose/BiasAdd/ReadVariableOp2p
6model/conv2d_transpose/conv2d_transpose/ReadVariableOp6model/conv2d_transpose/conv2d_transpose/ReadVariableOp2b
/model/conv2d_transpose_1/BiasAdd/ReadVariableOp/model/conv2d_transpose_1/BiasAdd/ReadVariableOp2t
8model/conv2d_transpose_1/conv2d_transpose/ReadVariableOp8model/conv2d_transpose_1/conv2d_transpose/ReadVariableOp:Z V
1
_output_shapes
:�����������
!
_user_specified_name	input_1
�
�
A__inference_conv2d_layer_call_and_return_conditional_losses_55704

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0r
Conv2D/CastCastConv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*&
_output_shapes
:�
Conv2DConv2DinputsConv2D/Cast:y:0*
T0*1
_output_shapes
:�����������*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0h
BiasAdd/CastCastBiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:q
BiasAddBiasAddConv2D:output:0BiasAdd/Cast:y:0*
T0*1
_output_shapes
:�����������H
mul/xConst*
_output_shapes
: *
dtype0*
value
B j�ph
mulMulmul/x:output:0BiasAdd:output:0*
T0*1
_output_shapes
:�����������I
Pow/yConst*
_output_shapes
: *
dtype0*
valueB	 j��h
PowPowBiasAdd:output:0Pow/y:output:0*
T0*1
_output_shapes
:�����������J
mul_1/xConst*
_output_shapes
: *
dtype0*
value
B j�Sc
mul_1Mulmul_1/x:output:0Pow:z:0*
T0*1
_output_shapes
:�����������e
addAddV2BiasAdd:output:0	mul_1:z:0*
T0*1
_output_shapes
:�����������J
mul_2/xConst*
_output_shapes
: *
dtype0*
value
B j�tc
mul_2Mulmul_2/x:output:0add:z:0*
T0*1
_output_shapes
:�����������S
TanhTanh	mul_2:z:0*
T0*1
_output_shapes
:�����������J
add_1/xConst*
_output_shapes
: *
dtype0*
value
B j�xf
add_1AddV2add_1/x:output:0Tanh:y:0*
T0*1
_output_shapes
:�����������\
mul_3Mulmul:z:0	add_1:z:0*
T0*1
_output_shapes
:�����������b
IdentityIdentity	mul_3:z:0^NoOp*
T0*1
_output_shapes
:�����������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:�����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
�
C__inference_conv2d_1_layer_call_and_return_conditional_losses_53217

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0r
Conv2D/CastCastConv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*&
_output_shapes
:�
Conv2DConv2DinputsConv2D/Cast:y:0*
T0*1
_output_shapes
:�����������*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0h
BiasAdd/CastCastBiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:q
BiasAddBiasAddConv2D:output:0BiasAdd/Cast:y:0*
T0*1
_output_shapes
:�����������H
mul/xConst*
_output_shapes
: *
dtype0*
value
B j�ph
mulMulmul/x:output:0BiasAdd:output:0*
T0*1
_output_shapes
:�����������I
Pow/yConst*
_output_shapes
: *
dtype0*
valueB	 j��h
PowPowBiasAdd:output:0Pow/y:output:0*
T0*1
_output_shapes
:�����������J
mul_1/xConst*
_output_shapes
: *
dtype0*
value
B j�Sc
mul_1Mulmul_1/x:output:0Pow:z:0*
T0*1
_output_shapes
:�����������e
addAddV2BiasAdd:output:0	mul_1:z:0*
T0*1
_output_shapes
:�����������J
mul_2/xConst*
_output_shapes
: *
dtype0*
value
B j�tc
mul_2Mulmul_2/x:output:0add:z:0*
T0*1
_output_shapes
:�����������S
TanhTanh	mul_2:z:0*
T0*1
_output_shapes
:�����������J
add_1/xConst*
_output_shapes
: *
dtype0*
value
B j�xf
add_1AddV2add_1/x:output:0Tanh:y:0*
T0*1
_output_shapes
:�����������\
mul_3Mulmul:z:0	add_1:z:0*
T0*1
_output_shapes
:�����������b
IdentityIdentity	mul_3:z:0^NoOp*
T0*1
_output_shapes
:�����������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:�����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�w
�
@__inference_model_layer_call_and_return_conditional_losses_53723

inputs&
conv2d_53187:
conv2d_53189:(
conv2d_1_53218:
conv2d_1_53220:(
conv2d_2_53249:
conv2d_2_53251:(
conv2d_3_53280:
conv2d_3_53282:(
conv2d_4_53321:<
conv2d_4_53323:<(
conv2d_5_53352:<<
conv2d_5_53354:<(
conv2d_6_53384:<x
conv2d_6_53386:x(
conv2d_7_53415:xx
conv2d_7_53417:x)
conv2d_8_53447:x�
conv2d_8_53449:	�*
conv2d_9_53478:��
conv2d_9_53480:	�+
conv2d_10_53509:��
conv2d_10_53511:	�+
conv2d_11_53540:��
conv2d_11_53542:	�1
conv2d_transpose_53545:x�$
conv2d_transpose_53547:x)
conv2d_12_53584:xx
conv2d_12_53586:x)
conv2d_13_53615:xx
conv2d_13_53617:x2
conv2d_transpose_1_53620:<x&
conv2d_transpose_1_53622:<)
conv2d_14_53659:<<
conv2d_14_53661:<)
conv2d_15_53690:<<
conv2d_15_53692:<)
conv2d_16_53708:<
conv2d_16_53710:
identity��conv2d/StatefulPartitionedCall� conv2d_1/StatefulPartitionedCall�!conv2d_10/StatefulPartitionedCall�!conv2d_11/StatefulPartitionedCall�!conv2d_12/StatefulPartitionedCall�!conv2d_13/StatefulPartitionedCall�!conv2d_14/StatefulPartitionedCall�!conv2d_15/StatefulPartitionedCall�!conv2d_16/StatefulPartitionedCall� conv2d_2/StatefulPartitionedCall� conv2d_3/StatefulPartitionedCall� conv2d_4/StatefulPartitionedCall� conv2d_5/StatefulPartitionedCall� conv2d_6/StatefulPartitionedCall� conv2d_7/StatefulPartitionedCall� conv2d_8/StatefulPartitionedCall� conv2d_9/StatefulPartitionedCall�(conv2d_transpose/StatefulPartitionedCall�*conv2d_transpose_1/StatefulPartitionedCallf
conv2d/CastCastinputs*

DstT0*

SrcT0*1
_output_shapes
:������������
conv2d/StatefulPartitionedCallStatefulPartitionedCallconv2d/Cast:y:0conv2d_53187conv2d_53189*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_53186�
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0conv2d_1_53218conv2d_1_53220*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_53217�
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0conv2d_2_53249conv2d_2_53251*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_conv2d_2_layer_call_and_return_conditional_losses_53248�
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0conv2d_3_53280conv2d_3_53282*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_conv2d_3_layer_call_and_return_conditional_losses_53279k
concatenate/CastCastinputs*

DstT0*

SrcT0*1
_output_shapes
:������������
concatenate/PartitionedCallPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0concatenate/Cast:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_concatenate_layer_call_and_return_conditional_losses_53293�
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0conv2d_4_53321conv2d_4_53323*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������<*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_conv2d_4_layer_call_and_return_conditional_losses_53320�
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0conv2d_5_53352conv2d_5_53354*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������<*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_conv2d_5_layer_call_and_return_conditional_losses_53351�
!average_pooling2d/PartitionedCallPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������xx<* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_average_pooling2d_layer_call_and_return_conditional_losses_53020�
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCall*average_pooling2d/PartitionedCall:output:0conv2d_6_53384conv2d_6_53386*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������xxx*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_conv2d_6_layer_call_and_return_conditional_losses_53383�
 conv2d_7/StatefulPartitionedCallStatefulPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0conv2d_7_53415conv2d_7_53417*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������xxx*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_conv2d_7_layer_call_and_return_conditional_losses_53414�
#average_pooling2d_1/PartitionedCallPartitionedCall)conv2d_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������<<x* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *W
fRRP
N__inference_average_pooling2d_1_layer_call_and_return_conditional_losses_53032�
 conv2d_8/StatefulPartitionedCallStatefulPartitionedCall,average_pooling2d_1/PartitionedCall:output:0conv2d_8_53447conv2d_8_53449*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������<<�*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_conv2d_8_layer_call_and_return_conditional_losses_53446�
 conv2d_9/StatefulPartitionedCallStatefulPartitionedCall)conv2d_8/StatefulPartitionedCall:output:0conv2d_9_53478conv2d_9_53480*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������<<�*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_conv2d_9_layer_call_and_return_conditional_losses_53477�
!conv2d_10/StatefulPartitionedCallStatefulPartitionedCall)conv2d_9/StatefulPartitionedCall:output:0conv2d_10_53509conv2d_10_53511*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������<<�*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv2d_10_layer_call_and_return_conditional_losses_53508�
!conv2d_11/StatefulPartitionedCallStatefulPartitionedCall*conv2d_10/StatefulPartitionedCall:output:0conv2d_11_53540conv2d_11_53542*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������<<�*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv2d_11_layer_call_and_return_conditional_losses_53539�
(conv2d_transpose/StatefulPartitionedCallStatefulPartitionedCall*conv2d_11/StatefulPartitionedCall:output:0conv2d_transpose_53545conv2d_transpose_53547*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������xxx*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *T
fORM
K__inference_conv2d_transpose_layer_call_and_return_conditional_losses_53087�
add/PartitionedCallPartitionedCall1conv2d_transpose/StatefulPartitionedCall:output:0)conv2d_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������xxx* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *G
fBR@
>__inference_add_layer_call_and_return_conditional_losses_53556�
!conv2d_12/StatefulPartitionedCallStatefulPartitionedCalladd/PartitionedCall:output:0conv2d_12_53584conv2d_12_53586*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������xxx*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv2d_12_layer_call_and_return_conditional_losses_53583�
!conv2d_13/StatefulPartitionedCallStatefulPartitionedCall*conv2d_12/StatefulPartitionedCall:output:0conv2d_13_53615conv2d_13_53617*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������xxx*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv2d_13_layer_call_and_return_conditional_losses_53614�
*conv2d_transpose_1/StatefulPartitionedCallStatefulPartitionedCall*conv2d_13/StatefulPartitionedCall:output:0conv2d_transpose_1_53620conv2d_transpose_1_53622*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������<*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *V
fQRO
M__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_53146�
add_1/PartitionedCallPartitionedCall3conv2d_transpose_1/StatefulPartitionedCall:output:0)conv2d_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������<* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *I
fDRB
@__inference_add_1_layer_call_and_return_conditional_losses_53631�
!conv2d_14/StatefulPartitionedCallStatefulPartitionedCalladd_1/PartitionedCall:output:0conv2d_14_53659conv2d_14_53661*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������<*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv2d_14_layer_call_and_return_conditional_losses_53658�
!conv2d_15/StatefulPartitionedCallStatefulPartitionedCall*conv2d_14/StatefulPartitionedCall:output:0conv2d_15_53690conv2d_15_53692*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������<*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv2d_15_layer_call_and_return_conditional_losses_53689�
!conv2d_16/StatefulPartitionedCallStatefulPartitionedCall*conv2d_15/StatefulPartitionedCall:output:0conv2d_16_53708conv2d_16_53710*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv2d_16_layer_call_and_return_conditional_losses_53707�

add_2/CastCast*conv2d_16/StatefulPartitionedCall:output:0*

DstT0*

SrcT0*1
_output_shapes
:������������
add_2/PartitionedCallPartitionedCalladd_2/Cast:y:0inputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *I
fDRB
@__inference_add_2_layer_call_and_return_conditional_losses_53720w
IdentityIdentityadd_2/PartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:������������
NoOpNoOp^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall"^conv2d_10/StatefulPartitionedCall"^conv2d_11/StatefulPartitionedCall"^conv2d_12/StatefulPartitionedCall"^conv2d_13/StatefulPartitionedCall"^conv2d_14/StatefulPartitionedCall"^conv2d_15/StatefulPartitionedCall"^conv2d_16/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall!^conv2d_5/StatefulPartitionedCall!^conv2d_6/StatefulPartitionedCall!^conv2d_7/StatefulPartitionedCall!^conv2d_8/StatefulPartitionedCall!^conv2d_9/StatefulPartitionedCall)^conv2d_transpose/StatefulPartitionedCall+^conv2d_transpose_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*|
_input_shapesk
i:�����������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2F
!conv2d_10/StatefulPartitionedCall!conv2d_10/StatefulPartitionedCall2F
!conv2d_11/StatefulPartitionedCall!conv2d_11/StatefulPartitionedCall2F
!conv2d_12/StatefulPartitionedCall!conv2d_12/StatefulPartitionedCall2F
!conv2d_13/StatefulPartitionedCall!conv2d_13/StatefulPartitionedCall2F
!conv2d_14/StatefulPartitionedCall!conv2d_14/StatefulPartitionedCall2F
!conv2d_15/StatefulPartitionedCall!conv2d_15/StatefulPartitionedCall2F
!conv2d_16/StatefulPartitionedCall!conv2d_16/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall2D
 conv2d_7/StatefulPartitionedCall conv2d_7/StatefulPartitionedCall2D
 conv2d_8/StatefulPartitionedCall conv2d_8/StatefulPartitionedCall2D
 conv2d_9/StatefulPartitionedCall conv2d_9/StatefulPartitionedCall2T
(conv2d_transpose/StatefulPartitionedCall(conv2d_transpose/StatefulPartitionedCall2X
*conv2d_transpose_1/StatefulPartitionedCall*conv2d_transpose_1/StatefulPartitionedCall:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
�
D__inference_conv2d_11_layer_call_and_return_conditional_losses_53539

inputs:
conv2d_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0t
Conv2D/CastCastConv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*(
_output_shapes
:���
Conv2DConv2DinputsConv2D/Cast:y:0*
T0*0
_output_shapes
:���������<<�*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0i
BiasAdd/CastCastBiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes	
:�p
BiasAddBiasAddConv2D:output:0BiasAdd/Cast:y:0*
T0*0
_output_shapes
:���������<<�H
mul/xConst*
_output_shapes
: *
dtype0*
value
B j�pg
mulMulmul/x:output:0BiasAdd:output:0*
T0*0
_output_shapes
:���������<<�I
Pow/yConst*
_output_shapes
: *
dtype0*
valueB	 j��g
PowPowBiasAdd:output:0Pow/y:output:0*
T0*0
_output_shapes
:���������<<�J
mul_1/xConst*
_output_shapes
: *
dtype0*
value
B j�Sb
mul_1Mulmul_1/x:output:0Pow:z:0*
T0*0
_output_shapes
:���������<<�d
addAddV2BiasAdd:output:0	mul_1:z:0*
T0*0
_output_shapes
:���������<<�J
mul_2/xConst*
_output_shapes
: *
dtype0*
value
B j�tb
mul_2Mulmul_2/x:output:0add:z:0*
T0*0
_output_shapes
:���������<<�R
TanhTanh	mul_2:z:0*
T0*0
_output_shapes
:���������<<�J
add_1/xConst*
_output_shapes
: *
dtype0*
value
B j�xe
add_1AddV2add_1/x:output:0Tanh:y:0*
T0*0
_output_shapes
:���������<<�[
mul_3Mulmul:z:0	add_1:z:0*
T0*0
_output_shapes
:���������<<�a
IdentityIdentity	mul_3:z:0^NoOp*
T0*0
_output_shapes
:���������<<�w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������<<�: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:���������<<�
 
_user_specified_nameinputs
�
�
(__inference_conv2d_6_layer_call_fn_55906

inputs!
unknown:<x
	unknown_0:x
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������xxx*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_conv2d_6_layer_call_and_return_conditional_losses_53383w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������xxx`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������xx<: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������xx<
 
_user_specified_nameinputs
�
j
@__inference_add_1_layer_call_and_return_conditional_losses_53631

inputs
inputs_1
identityZ
addAddV2inputsinputs_1*
T0*1
_output_shapes
:�����������<Y
IdentityIdentityadd:z:0*
T0*1
_output_shapes
:�����������<"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*M
_input_shapes<
::�����������<:�����������<:Y U
1
_output_shapes
:�����������<
 
_user_specified_nameinputs:YU
1
_output_shapes
:�����������<
 
_user_specified_nameinputs
��
�
@__inference_model_layer_call_and_return_conditional_losses_55670

inputs?
%conv2d_conv2d_readvariableop_resource:4
&conv2d_biasadd_readvariableop_resource:A
'conv2d_1_conv2d_readvariableop_resource:6
(conv2d_1_biasadd_readvariableop_resource:A
'conv2d_2_conv2d_readvariableop_resource:6
(conv2d_2_biasadd_readvariableop_resource:A
'conv2d_3_conv2d_readvariableop_resource:6
(conv2d_3_biasadd_readvariableop_resource:A
'conv2d_4_conv2d_readvariableop_resource:<6
(conv2d_4_biasadd_readvariableop_resource:<A
'conv2d_5_conv2d_readvariableop_resource:<<6
(conv2d_5_biasadd_readvariableop_resource:<A
'conv2d_6_conv2d_readvariableop_resource:<x6
(conv2d_6_biasadd_readvariableop_resource:xA
'conv2d_7_conv2d_readvariableop_resource:xx6
(conv2d_7_biasadd_readvariableop_resource:xB
'conv2d_8_conv2d_readvariableop_resource:x�7
(conv2d_8_biasadd_readvariableop_resource:	�C
'conv2d_9_conv2d_readvariableop_resource:��7
(conv2d_9_biasadd_readvariableop_resource:	�D
(conv2d_10_conv2d_readvariableop_resource:��8
)conv2d_10_biasadd_readvariableop_resource:	�D
(conv2d_11_conv2d_readvariableop_resource:��8
)conv2d_11_biasadd_readvariableop_resource:	�T
9conv2d_transpose_conv2d_transpose_readvariableop_resource:x�>
0conv2d_transpose_biasadd_readvariableop_resource:xB
(conv2d_12_conv2d_readvariableop_resource:xx7
)conv2d_12_biasadd_readvariableop_resource:xB
(conv2d_13_conv2d_readvariableop_resource:xx7
)conv2d_13_biasadd_readvariableop_resource:xU
;conv2d_transpose_1_conv2d_transpose_readvariableop_resource:<x@
2conv2d_transpose_1_biasadd_readvariableop_resource:<B
(conv2d_14_conv2d_readvariableop_resource:<<7
)conv2d_14_biasadd_readvariableop_resource:<B
(conv2d_15_conv2d_readvariableop_resource:<<7
)conv2d_15_biasadd_readvariableop_resource:<B
(conv2d_16_conv2d_readvariableop_resource:<7
)conv2d_16_biasadd_readvariableop_resource:
identity��conv2d/BiasAdd/ReadVariableOp�conv2d/Conv2D/ReadVariableOp�conv2d_1/BiasAdd/ReadVariableOp�conv2d_1/Conv2D/ReadVariableOp� conv2d_10/BiasAdd/ReadVariableOp�conv2d_10/Conv2D/ReadVariableOp� conv2d_11/BiasAdd/ReadVariableOp�conv2d_11/Conv2D/ReadVariableOp� conv2d_12/BiasAdd/ReadVariableOp�conv2d_12/Conv2D/ReadVariableOp� conv2d_13/BiasAdd/ReadVariableOp�conv2d_13/Conv2D/ReadVariableOp� conv2d_14/BiasAdd/ReadVariableOp�conv2d_14/Conv2D/ReadVariableOp� conv2d_15/BiasAdd/ReadVariableOp�conv2d_15/Conv2D/ReadVariableOp� conv2d_16/BiasAdd/ReadVariableOp�conv2d_16/Conv2D/ReadVariableOp�conv2d_2/BiasAdd/ReadVariableOp�conv2d_2/Conv2D/ReadVariableOp�conv2d_3/BiasAdd/ReadVariableOp�conv2d_3/Conv2D/ReadVariableOp�conv2d_4/BiasAdd/ReadVariableOp�conv2d_4/Conv2D/ReadVariableOp�conv2d_5/BiasAdd/ReadVariableOp�conv2d_5/Conv2D/ReadVariableOp�conv2d_6/BiasAdd/ReadVariableOp�conv2d_6/Conv2D/ReadVariableOp�conv2d_7/BiasAdd/ReadVariableOp�conv2d_7/Conv2D/ReadVariableOp�conv2d_8/BiasAdd/ReadVariableOp�conv2d_8/Conv2D/ReadVariableOp�conv2d_9/BiasAdd/ReadVariableOp�conv2d_9/Conv2D/ReadVariableOp�'conv2d_transpose/BiasAdd/ReadVariableOp�0conv2d_transpose/conv2d_transpose/ReadVariableOp�)conv2d_transpose_1/BiasAdd/ReadVariableOp�2conv2d_transpose_1/conv2d_transpose/ReadVariableOpf
conv2d/CastCastinputs*

DstT0*

SrcT0*1
_output_shapes
:������������
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
conv2d/Conv2D/CastCast$conv2d/Conv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*&
_output_shapes
:�
conv2d/Conv2DConv2Dconv2d/Cast:y:0conv2d/Conv2D/Cast:y:0*
T0*1
_output_shapes
:�����������*
paddingSAME*
strides
�
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
conv2d/BiasAdd/CastCast%conv2d/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:�
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0conv2d/BiasAdd/Cast:y:0*
T0*1
_output_shapes
:�����������O
conv2d/mul/xConst*
_output_shapes
: *
dtype0*
value
B j�p}

conv2d/mulMulconv2d/mul/x:output:0conv2d/BiasAdd:output:0*
T0*1
_output_shapes
:�����������P
conv2d/Pow/yConst*
_output_shapes
: *
dtype0*
valueB	 j��}

conv2d/PowPowconv2d/BiasAdd:output:0conv2d/Pow/y:output:0*
T0*1
_output_shapes
:�����������Q
conv2d/mul_1/xConst*
_output_shapes
: *
dtype0*
value
B j�Sx
conv2d/mul_1Mulconv2d/mul_1/x:output:0conv2d/Pow:z:0*
T0*1
_output_shapes
:�����������z

conv2d/addAddV2conv2d/BiasAdd:output:0conv2d/mul_1:z:0*
T0*1
_output_shapes
:�����������Q
conv2d/mul_2/xConst*
_output_shapes
: *
dtype0*
value
B j�tx
conv2d/mul_2Mulconv2d/mul_2/x:output:0conv2d/add:z:0*
T0*1
_output_shapes
:�����������a
conv2d/TanhTanhconv2d/mul_2:z:0*
T0*1
_output_shapes
:�����������Q
conv2d/add_1/xConst*
_output_shapes
: *
dtype0*
value
B j�x{
conv2d/add_1AddV2conv2d/add_1/x:output:0conv2d/Tanh:y:0*
T0*1
_output_shapes
:�����������q
conv2d/mul_3Mulconv2d/mul:z:0conv2d/add_1:z:0*
T0*1
_output_shapes
:������������
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
conv2d_1/Conv2D/CastCast&conv2d_1/Conv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*&
_output_shapes
:�
conv2d_1/Conv2DConv2Dconv2d/mul_3:z:0conv2d_1/Conv2D/Cast:y:0*
T0*1
_output_shapes
:�����������*
paddingSAME*
strides
�
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0z
conv2d_1/BiasAdd/CastCast'conv2d_1/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:�
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0conv2d_1/BiasAdd/Cast:y:0*
T0*1
_output_shapes
:�����������Q
conv2d_1/mul/xConst*
_output_shapes
: *
dtype0*
value
B j�p�
conv2d_1/mulMulconv2d_1/mul/x:output:0conv2d_1/BiasAdd:output:0*
T0*1
_output_shapes
:�����������R
conv2d_1/Pow/yConst*
_output_shapes
: *
dtype0*
valueB	 j���
conv2d_1/PowPowconv2d_1/BiasAdd:output:0conv2d_1/Pow/y:output:0*
T0*1
_output_shapes
:�����������S
conv2d_1/mul_1/xConst*
_output_shapes
: *
dtype0*
value
B j�S~
conv2d_1/mul_1Mulconv2d_1/mul_1/x:output:0conv2d_1/Pow:z:0*
T0*1
_output_shapes
:������������
conv2d_1/addAddV2conv2d_1/BiasAdd:output:0conv2d_1/mul_1:z:0*
T0*1
_output_shapes
:�����������S
conv2d_1/mul_2/xConst*
_output_shapes
: *
dtype0*
value
B j�t~
conv2d_1/mul_2Mulconv2d_1/mul_2/x:output:0conv2d_1/add:z:0*
T0*1
_output_shapes
:�����������e
conv2d_1/TanhTanhconv2d_1/mul_2:z:0*
T0*1
_output_shapes
:�����������S
conv2d_1/add_1/xConst*
_output_shapes
: *
dtype0*
value
B j�x�
conv2d_1/add_1AddV2conv2d_1/add_1/x:output:0conv2d_1/Tanh:y:0*
T0*1
_output_shapes
:�����������w
conv2d_1/mul_3Mulconv2d_1/mul:z:0conv2d_1/add_1:z:0*
T0*1
_output_shapes
:������������
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
conv2d_2/Conv2D/CastCast&conv2d_2/Conv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*&
_output_shapes
:�
conv2d_2/Conv2DConv2Dconv2d_1/mul_3:z:0conv2d_2/Conv2D/Cast:y:0*
T0*1
_output_shapes
:�����������*
paddingSAME*
strides
�
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0z
conv2d_2/BiasAdd/CastCast'conv2d_2/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:�
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0conv2d_2/BiasAdd/Cast:y:0*
T0*1
_output_shapes
:�����������Q
conv2d_2/mul/xConst*
_output_shapes
: *
dtype0*
value
B j�p�
conv2d_2/mulMulconv2d_2/mul/x:output:0conv2d_2/BiasAdd:output:0*
T0*1
_output_shapes
:�����������R
conv2d_2/Pow/yConst*
_output_shapes
: *
dtype0*
valueB	 j���
conv2d_2/PowPowconv2d_2/BiasAdd:output:0conv2d_2/Pow/y:output:0*
T0*1
_output_shapes
:�����������S
conv2d_2/mul_1/xConst*
_output_shapes
: *
dtype0*
value
B j�S~
conv2d_2/mul_1Mulconv2d_2/mul_1/x:output:0conv2d_2/Pow:z:0*
T0*1
_output_shapes
:������������
conv2d_2/addAddV2conv2d_2/BiasAdd:output:0conv2d_2/mul_1:z:0*
T0*1
_output_shapes
:�����������S
conv2d_2/mul_2/xConst*
_output_shapes
: *
dtype0*
value
B j�t~
conv2d_2/mul_2Mulconv2d_2/mul_2/x:output:0conv2d_2/add:z:0*
T0*1
_output_shapes
:�����������e
conv2d_2/TanhTanhconv2d_2/mul_2:z:0*
T0*1
_output_shapes
:�����������S
conv2d_2/add_1/xConst*
_output_shapes
: *
dtype0*
value
B j�x�
conv2d_2/add_1AddV2conv2d_2/add_1/x:output:0conv2d_2/Tanh:y:0*
T0*1
_output_shapes
:�����������w
conv2d_2/mul_3Mulconv2d_2/mul:z:0conv2d_2/add_1:z:0*
T0*1
_output_shapes
:������������
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
conv2d_3/Conv2D/CastCast&conv2d_3/Conv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*&
_output_shapes
:�
conv2d_3/Conv2DConv2Dconv2d_2/mul_3:z:0conv2d_3/Conv2D/Cast:y:0*
T0*1
_output_shapes
:�����������*
paddingSAME*
strides
�
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0z
conv2d_3/BiasAdd/CastCast'conv2d_3/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:�
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0conv2d_3/BiasAdd/Cast:y:0*
T0*1
_output_shapes
:�����������Q
conv2d_3/mul/xConst*
_output_shapes
: *
dtype0*
value
B j�p�
conv2d_3/mulMulconv2d_3/mul/x:output:0conv2d_3/BiasAdd:output:0*
T0*1
_output_shapes
:�����������R
conv2d_3/Pow/yConst*
_output_shapes
: *
dtype0*
valueB	 j���
conv2d_3/PowPowconv2d_3/BiasAdd:output:0conv2d_3/Pow/y:output:0*
T0*1
_output_shapes
:�����������S
conv2d_3/mul_1/xConst*
_output_shapes
: *
dtype0*
value
B j�S~
conv2d_3/mul_1Mulconv2d_3/mul_1/x:output:0conv2d_3/Pow:z:0*
T0*1
_output_shapes
:������������
conv2d_3/addAddV2conv2d_3/BiasAdd:output:0conv2d_3/mul_1:z:0*
T0*1
_output_shapes
:�����������S
conv2d_3/mul_2/xConst*
_output_shapes
: *
dtype0*
value
B j�t~
conv2d_3/mul_2Mulconv2d_3/mul_2/x:output:0conv2d_3/add:z:0*
T0*1
_output_shapes
:�����������e
conv2d_3/TanhTanhconv2d_3/mul_2:z:0*
T0*1
_output_shapes
:�����������S
conv2d_3/add_1/xConst*
_output_shapes
: *
dtype0*
value
B j�x�
conv2d_3/add_1AddV2conv2d_3/add_1/x:output:0conv2d_3/Tanh:y:0*
T0*1
_output_shapes
:�����������w
conv2d_3/mul_3Mulconv2d_3/mul:z:0conv2d_3/add_1:z:0*
T0*1
_output_shapes
:�����������k
concatenate/CastCastinputs*

DstT0*

SrcT0*1
_output_shapes
:�����������Y
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concatenate/concatConcatV2conv2d_3/mul_3:z:0concatenate/Cast:y:0 concatenate/concat/axis:output:0*
N*
T0*1
_output_shapes
:������������
conv2d_4/Conv2D/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
:<*
dtype0�
conv2d_4/Conv2D/CastCast&conv2d_4/Conv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*&
_output_shapes
:<�
conv2d_4/Conv2DConv2Dconcatenate/concat:output:0conv2d_4/Conv2D/Cast:y:0*
T0*1
_output_shapes
:�����������<*
paddingSAME*
strides
�
conv2d_4/BiasAdd/ReadVariableOpReadVariableOp(conv2d_4_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype0z
conv2d_4/BiasAdd/CastCast'conv2d_4/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:<�
conv2d_4/BiasAddBiasAddconv2d_4/Conv2D:output:0conv2d_4/BiasAdd/Cast:y:0*
T0*1
_output_shapes
:�����������<Q
conv2d_4/mul/xConst*
_output_shapes
: *
dtype0*
value
B j�p�
conv2d_4/mulMulconv2d_4/mul/x:output:0conv2d_4/BiasAdd:output:0*
T0*1
_output_shapes
:�����������<R
conv2d_4/Pow/yConst*
_output_shapes
: *
dtype0*
valueB	 j���
conv2d_4/PowPowconv2d_4/BiasAdd:output:0conv2d_4/Pow/y:output:0*
T0*1
_output_shapes
:�����������<S
conv2d_4/mul_1/xConst*
_output_shapes
: *
dtype0*
value
B j�S~
conv2d_4/mul_1Mulconv2d_4/mul_1/x:output:0conv2d_4/Pow:z:0*
T0*1
_output_shapes
:�����������<�
conv2d_4/addAddV2conv2d_4/BiasAdd:output:0conv2d_4/mul_1:z:0*
T0*1
_output_shapes
:�����������<S
conv2d_4/mul_2/xConst*
_output_shapes
: *
dtype0*
value
B j�t~
conv2d_4/mul_2Mulconv2d_4/mul_2/x:output:0conv2d_4/add:z:0*
T0*1
_output_shapes
:�����������<e
conv2d_4/TanhTanhconv2d_4/mul_2:z:0*
T0*1
_output_shapes
:�����������<S
conv2d_4/add_1/xConst*
_output_shapes
: *
dtype0*
value
B j�x�
conv2d_4/add_1AddV2conv2d_4/add_1/x:output:0conv2d_4/Tanh:y:0*
T0*1
_output_shapes
:�����������<w
conv2d_4/mul_3Mulconv2d_4/mul:z:0conv2d_4/add_1:z:0*
T0*1
_output_shapes
:�����������<�
conv2d_5/Conv2D/ReadVariableOpReadVariableOp'conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
:<<*
dtype0�
conv2d_5/Conv2D/CastCast&conv2d_5/Conv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*&
_output_shapes
:<<�
conv2d_5/Conv2DConv2Dconv2d_4/mul_3:z:0conv2d_5/Conv2D/Cast:y:0*
T0*1
_output_shapes
:�����������<*
paddingSAME*
strides
�
conv2d_5/BiasAdd/ReadVariableOpReadVariableOp(conv2d_5_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype0z
conv2d_5/BiasAdd/CastCast'conv2d_5/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:<�
conv2d_5/BiasAddBiasAddconv2d_5/Conv2D:output:0conv2d_5/BiasAdd/Cast:y:0*
T0*1
_output_shapes
:�����������<Q
conv2d_5/mul/xConst*
_output_shapes
: *
dtype0*
value
B j�p�
conv2d_5/mulMulconv2d_5/mul/x:output:0conv2d_5/BiasAdd:output:0*
T0*1
_output_shapes
:�����������<R
conv2d_5/Pow/yConst*
_output_shapes
: *
dtype0*
valueB	 j���
conv2d_5/PowPowconv2d_5/BiasAdd:output:0conv2d_5/Pow/y:output:0*
T0*1
_output_shapes
:�����������<S
conv2d_5/mul_1/xConst*
_output_shapes
: *
dtype0*
value
B j�S~
conv2d_5/mul_1Mulconv2d_5/mul_1/x:output:0conv2d_5/Pow:z:0*
T0*1
_output_shapes
:�����������<�
conv2d_5/addAddV2conv2d_5/BiasAdd:output:0conv2d_5/mul_1:z:0*
T0*1
_output_shapes
:�����������<S
conv2d_5/mul_2/xConst*
_output_shapes
: *
dtype0*
value
B j�t~
conv2d_5/mul_2Mulconv2d_5/mul_2/x:output:0conv2d_5/add:z:0*
T0*1
_output_shapes
:�����������<e
conv2d_5/TanhTanhconv2d_5/mul_2:z:0*
T0*1
_output_shapes
:�����������<S
conv2d_5/add_1/xConst*
_output_shapes
: *
dtype0*
value
B j�x�
conv2d_5/add_1AddV2conv2d_5/add_1/x:output:0conv2d_5/Tanh:y:0*
T0*1
_output_shapes
:�����������<w
conv2d_5/mul_3Mulconv2d_5/mul:z:0conv2d_5/add_1:z:0*
T0*1
_output_shapes
:�����������<�
average_pooling2d/AvgPoolAvgPoolconv2d_5/mul_3:z:0*
T0*/
_output_shapes
:���������xx<*
ksize
*
paddingSAME*
strides
�
conv2d_6/Conv2D/ReadVariableOpReadVariableOp'conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
:<x*
dtype0�
conv2d_6/Conv2D/CastCast&conv2d_6/Conv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*&
_output_shapes
:<x�
conv2d_6/Conv2DConv2D"average_pooling2d/AvgPool:output:0conv2d_6/Conv2D/Cast:y:0*
T0*/
_output_shapes
:���������xxx*
paddingSAME*
strides
�
conv2d_6/BiasAdd/ReadVariableOpReadVariableOp(conv2d_6_biasadd_readvariableop_resource*
_output_shapes
:x*
dtype0z
conv2d_6/BiasAdd/CastCast'conv2d_6/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:x�
conv2d_6/BiasAddBiasAddconv2d_6/Conv2D:output:0conv2d_6/BiasAdd/Cast:y:0*
T0*/
_output_shapes
:���������xxxQ
conv2d_6/mul/xConst*
_output_shapes
: *
dtype0*
value
B j�p�
conv2d_6/mulMulconv2d_6/mul/x:output:0conv2d_6/BiasAdd:output:0*
T0*/
_output_shapes
:���������xxxR
conv2d_6/Pow/yConst*
_output_shapes
: *
dtype0*
valueB	 j���
conv2d_6/PowPowconv2d_6/BiasAdd:output:0conv2d_6/Pow/y:output:0*
T0*/
_output_shapes
:���������xxxS
conv2d_6/mul_1/xConst*
_output_shapes
: *
dtype0*
value
B j�S|
conv2d_6/mul_1Mulconv2d_6/mul_1/x:output:0conv2d_6/Pow:z:0*
T0*/
_output_shapes
:���������xxx~
conv2d_6/addAddV2conv2d_6/BiasAdd:output:0conv2d_6/mul_1:z:0*
T0*/
_output_shapes
:���������xxxS
conv2d_6/mul_2/xConst*
_output_shapes
: *
dtype0*
value
B j�t|
conv2d_6/mul_2Mulconv2d_6/mul_2/x:output:0conv2d_6/add:z:0*
T0*/
_output_shapes
:���������xxxc
conv2d_6/TanhTanhconv2d_6/mul_2:z:0*
T0*/
_output_shapes
:���������xxxS
conv2d_6/add_1/xConst*
_output_shapes
: *
dtype0*
value
B j�x
conv2d_6/add_1AddV2conv2d_6/add_1/x:output:0conv2d_6/Tanh:y:0*
T0*/
_output_shapes
:���������xxxu
conv2d_6/mul_3Mulconv2d_6/mul:z:0conv2d_6/add_1:z:0*
T0*/
_output_shapes
:���������xxx�
conv2d_7/Conv2D/ReadVariableOpReadVariableOp'conv2d_7_conv2d_readvariableop_resource*&
_output_shapes
:xx*
dtype0�
conv2d_7/Conv2D/CastCast&conv2d_7/Conv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*&
_output_shapes
:xx�
conv2d_7/Conv2DConv2Dconv2d_6/mul_3:z:0conv2d_7/Conv2D/Cast:y:0*
T0*/
_output_shapes
:���������xxx*
paddingSAME*
strides
�
conv2d_7/BiasAdd/ReadVariableOpReadVariableOp(conv2d_7_biasadd_readvariableop_resource*
_output_shapes
:x*
dtype0z
conv2d_7/BiasAdd/CastCast'conv2d_7/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:x�
conv2d_7/BiasAddBiasAddconv2d_7/Conv2D:output:0conv2d_7/BiasAdd/Cast:y:0*
T0*/
_output_shapes
:���������xxxQ
conv2d_7/mul/xConst*
_output_shapes
: *
dtype0*
value
B j�p�
conv2d_7/mulMulconv2d_7/mul/x:output:0conv2d_7/BiasAdd:output:0*
T0*/
_output_shapes
:���������xxxR
conv2d_7/Pow/yConst*
_output_shapes
: *
dtype0*
valueB	 j���
conv2d_7/PowPowconv2d_7/BiasAdd:output:0conv2d_7/Pow/y:output:0*
T0*/
_output_shapes
:���������xxxS
conv2d_7/mul_1/xConst*
_output_shapes
: *
dtype0*
value
B j�S|
conv2d_7/mul_1Mulconv2d_7/mul_1/x:output:0conv2d_7/Pow:z:0*
T0*/
_output_shapes
:���������xxx~
conv2d_7/addAddV2conv2d_7/BiasAdd:output:0conv2d_7/mul_1:z:0*
T0*/
_output_shapes
:���������xxxS
conv2d_7/mul_2/xConst*
_output_shapes
: *
dtype0*
value
B j�t|
conv2d_7/mul_2Mulconv2d_7/mul_2/x:output:0conv2d_7/add:z:0*
T0*/
_output_shapes
:���������xxxc
conv2d_7/TanhTanhconv2d_7/mul_2:z:0*
T0*/
_output_shapes
:���������xxxS
conv2d_7/add_1/xConst*
_output_shapes
: *
dtype0*
value
B j�x
conv2d_7/add_1AddV2conv2d_7/add_1/x:output:0conv2d_7/Tanh:y:0*
T0*/
_output_shapes
:���������xxxu
conv2d_7/mul_3Mulconv2d_7/mul:z:0conv2d_7/add_1:z:0*
T0*/
_output_shapes
:���������xxx�
average_pooling2d_1/AvgPoolAvgPoolconv2d_7/mul_3:z:0*
T0*/
_output_shapes
:���������<<x*
ksize
*
paddingSAME*
strides
�
conv2d_8/Conv2D/ReadVariableOpReadVariableOp'conv2d_8_conv2d_readvariableop_resource*'
_output_shapes
:x�*
dtype0�
conv2d_8/Conv2D/CastCast&conv2d_8/Conv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*'
_output_shapes
:x��
conv2d_8/Conv2DConv2D$average_pooling2d_1/AvgPool:output:0conv2d_8/Conv2D/Cast:y:0*
T0*0
_output_shapes
:���������<<�*
paddingSAME*
strides
�
conv2d_8/BiasAdd/ReadVariableOpReadVariableOp(conv2d_8_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0{
conv2d_8/BiasAdd/CastCast'conv2d_8/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes	
:��
conv2d_8/BiasAddBiasAddconv2d_8/Conv2D:output:0conv2d_8/BiasAdd/Cast:y:0*
T0*0
_output_shapes
:���������<<�Q
conv2d_8/mul/xConst*
_output_shapes
: *
dtype0*
value
B j�p�
conv2d_8/mulMulconv2d_8/mul/x:output:0conv2d_8/BiasAdd:output:0*
T0*0
_output_shapes
:���������<<�R
conv2d_8/Pow/yConst*
_output_shapes
: *
dtype0*
valueB	 j���
conv2d_8/PowPowconv2d_8/BiasAdd:output:0conv2d_8/Pow/y:output:0*
T0*0
_output_shapes
:���������<<�S
conv2d_8/mul_1/xConst*
_output_shapes
: *
dtype0*
value
B j�S}
conv2d_8/mul_1Mulconv2d_8/mul_1/x:output:0conv2d_8/Pow:z:0*
T0*0
_output_shapes
:���������<<�
conv2d_8/addAddV2conv2d_8/BiasAdd:output:0conv2d_8/mul_1:z:0*
T0*0
_output_shapes
:���������<<�S
conv2d_8/mul_2/xConst*
_output_shapes
: *
dtype0*
value
B j�t}
conv2d_8/mul_2Mulconv2d_8/mul_2/x:output:0conv2d_8/add:z:0*
T0*0
_output_shapes
:���������<<�d
conv2d_8/TanhTanhconv2d_8/mul_2:z:0*
T0*0
_output_shapes
:���������<<�S
conv2d_8/add_1/xConst*
_output_shapes
: *
dtype0*
value
B j�x�
conv2d_8/add_1AddV2conv2d_8/add_1/x:output:0conv2d_8/Tanh:y:0*
T0*0
_output_shapes
:���������<<�v
conv2d_8/mul_3Mulconv2d_8/mul:z:0conv2d_8/add_1:z:0*
T0*0
_output_shapes
:���������<<��
conv2d_9/Conv2D/ReadVariableOpReadVariableOp'conv2d_9_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
conv2d_9/Conv2D/CastCast&conv2d_9/Conv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*(
_output_shapes
:���
conv2d_9/Conv2DConv2Dconv2d_8/mul_3:z:0conv2d_9/Conv2D/Cast:y:0*
T0*0
_output_shapes
:���������<<�*
paddingSAME*
strides
�
conv2d_9/BiasAdd/ReadVariableOpReadVariableOp(conv2d_9_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0{
conv2d_9/BiasAdd/CastCast'conv2d_9/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes	
:��
conv2d_9/BiasAddBiasAddconv2d_9/Conv2D:output:0conv2d_9/BiasAdd/Cast:y:0*
T0*0
_output_shapes
:���������<<�Q
conv2d_9/mul/xConst*
_output_shapes
: *
dtype0*
value
B j�p�
conv2d_9/mulMulconv2d_9/mul/x:output:0conv2d_9/BiasAdd:output:0*
T0*0
_output_shapes
:���������<<�R
conv2d_9/Pow/yConst*
_output_shapes
: *
dtype0*
valueB	 j���
conv2d_9/PowPowconv2d_9/BiasAdd:output:0conv2d_9/Pow/y:output:0*
T0*0
_output_shapes
:���������<<�S
conv2d_9/mul_1/xConst*
_output_shapes
: *
dtype0*
value
B j�S}
conv2d_9/mul_1Mulconv2d_9/mul_1/x:output:0conv2d_9/Pow:z:0*
T0*0
_output_shapes
:���������<<�
conv2d_9/addAddV2conv2d_9/BiasAdd:output:0conv2d_9/mul_1:z:0*
T0*0
_output_shapes
:���������<<�S
conv2d_9/mul_2/xConst*
_output_shapes
: *
dtype0*
value
B j�t}
conv2d_9/mul_2Mulconv2d_9/mul_2/x:output:0conv2d_9/add:z:0*
T0*0
_output_shapes
:���������<<�d
conv2d_9/TanhTanhconv2d_9/mul_2:z:0*
T0*0
_output_shapes
:���������<<�S
conv2d_9/add_1/xConst*
_output_shapes
: *
dtype0*
value
B j�x�
conv2d_9/add_1AddV2conv2d_9/add_1/x:output:0conv2d_9/Tanh:y:0*
T0*0
_output_shapes
:���������<<�v
conv2d_9/mul_3Mulconv2d_9/mul:z:0conv2d_9/add_1:z:0*
T0*0
_output_shapes
:���������<<��
conv2d_10/Conv2D/ReadVariableOpReadVariableOp(conv2d_10_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
conv2d_10/Conv2D/CastCast'conv2d_10/Conv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*(
_output_shapes
:���
conv2d_10/Conv2DConv2Dconv2d_9/mul_3:z:0conv2d_10/Conv2D/Cast:y:0*
T0*0
_output_shapes
:���������<<�*
paddingSAME*
strides
�
 conv2d_10/BiasAdd/ReadVariableOpReadVariableOp)conv2d_10_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0}
conv2d_10/BiasAdd/CastCast(conv2d_10/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes	
:��
conv2d_10/BiasAddBiasAddconv2d_10/Conv2D:output:0conv2d_10/BiasAdd/Cast:y:0*
T0*0
_output_shapes
:���������<<�R
conv2d_10/mul/xConst*
_output_shapes
: *
dtype0*
value
B j�p�
conv2d_10/mulMulconv2d_10/mul/x:output:0conv2d_10/BiasAdd:output:0*
T0*0
_output_shapes
:���������<<�S
conv2d_10/Pow/yConst*
_output_shapes
: *
dtype0*
valueB	 j���
conv2d_10/PowPowconv2d_10/BiasAdd:output:0conv2d_10/Pow/y:output:0*
T0*0
_output_shapes
:���������<<�T
conv2d_10/mul_1/xConst*
_output_shapes
: *
dtype0*
value
B j�S�
conv2d_10/mul_1Mulconv2d_10/mul_1/x:output:0conv2d_10/Pow:z:0*
T0*0
_output_shapes
:���������<<��
conv2d_10/addAddV2conv2d_10/BiasAdd:output:0conv2d_10/mul_1:z:0*
T0*0
_output_shapes
:���������<<�T
conv2d_10/mul_2/xConst*
_output_shapes
: *
dtype0*
value
B j�t�
conv2d_10/mul_2Mulconv2d_10/mul_2/x:output:0conv2d_10/add:z:0*
T0*0
_output_shapes
:���������<<�f
conv2d_10/TanhTanhconv2d_10/mul_2:z:0*
T0*0
_output_shapes
:���������<<�T
conv2d_10/add_1/xConst*
_output_shapes
: *
dtype0*
value
B j�x�
conv2d_10/add_1AddV2conv2d_10/add_1/x:output:0conv2d_10/Tanh:y:0*
T0*0
_output_shapes
:���������<<�y
conv2d_10/mul_3Mulconv2d_10/mul:z:0conv2d_10/add_1:z:0*
T0*0
_output_shapes
:���������<<��
conv2d_11/Conv2D/ReadVariableOpReadVariableOp(conv2d_11_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
conv2d_11/Conv2D/CastCast'conv2d_11/Conv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*(
_output_shapes
:���
conv2d_11/Conv2DConv2Dconv2d_10/mul_3:z:0conv2d_11/Conv2D/Cast:y:0*
T0*0
_output_shapes
:���������<<�*
paddingSAME*
strides
�
 conv2d_11/BiasAdd/ReadVariableOpReadVariableOp)conv2d_11_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0}
conv2d_11/BiasAdd/CastCast(conv2d_11/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes	
:��
conv2d_11/BiasAddBiasAddconv2d_11/Conv2D:output:0conv2d_11/BiasAdd/Cast:y:0*
T0*0
_output_shapes
:���������<<�R
conv2d_11/mul/xConst*
_output_shapes
: *
dtype0*
value
B j�p�
conv2d_11/mulMulconv2d_11/mul/x:output:0conv2d_11/BiasAdd:output:0*
T0*0
_output_shapes
:���������<<�S
conv2d_11/Pow/yConst*
_output_shapes
: *
dtype0*
valueB	 j���
conv2d_11/PowPowconv2d_11/BiasAdd:output:0conv2d_11/Pow/y:output:0*
T0*0
_output_shapes
:���������<<�T
conv2d_11/mul_1/xConst*
_output_shapes
: *
dtype0*
value
B j�S�
conv2d_11/mul_1Mulconv2d_11/mul_1/x:output:0conv2d_11/Pow:z:0*
T0*0
_output_shapes
:���������<<��
conv2d_11/addAddV2conv2d_11/BiasAdd:output:0conv2d_11/mul_1:z:0*
T0*0
_output_shapes
:���������<<�T
conv2d_11/mul_2/xConst*
_output_shapes
: *
dtype0*
value
B j�t�
conv2d_11/mul_2Mulconv2d_11/mul_2/x:output:0conv2d_11/add:z:0*
T0*0
_output_shapes
:���������<<�f
conv2d_11/TanhTanhconv2d_11/mul_2:z:0*
T0*0
_output_shapes
:���������<<�T
conv2d_11/add_1/xConst*
_output_shapes
: *
dtype0*
value
B j�x�
conv2d_11/add_1AddV2conv2d_11/add_1/x:output:0conv2d_11/Tanh:y:0*
T0*0
_output_shapes
:���������<<�y
conv2d_11/mul_3Mulconv2d_11/mul:z:0conv2d_11/add_1:z:0*
T0*0
_output_shapes
:���������<<�Y
conv2d_transpose/ShapeShapeconv2d_11/mul_3:z:0*
T0*
_output_shapes
:n
$conv2d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: p
&conv2d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&conv2d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
conv2d_transpose/strided_sliceStridedSliceconv2d_transpose/Shape:output:0-conv2d_transpose/strided_slice/stack:output:0/conv2d_transpose/strided_slice/stack_1:output:0/conv2d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskZ
conv2d_transpose/stack/1Const*
_output_shapes
: *
dtype0*
value	B :xZ
conv2d_transpose/stack/2Const*
_output_shapes
: *
dtype0*
value	B :xZ
conv2d_transpose/stack/3Const*
_output_shapes
: *
dtype0*
value	B :x�
conv2d_transpose/stackPack'conv2d_transpose/strided_slice:output:0!conv2d_transpose/stack/1:output:0!conv2d_transpose/stack/2:output:0!conv2d_transpose/stack/3:output:0*
N*
T0*
_output_shapes
:p
&conv2d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv2d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv2d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
 conv2d_transpose/strided_slice_1StridedSliceconv2d_transpose/stack:output:0/conv2d_transpose/strided_slice_1/stack:output:01conv2d_transpose/strided_slice_1/stack_1:output:01conv2d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
0conv2d_transpose/conv2d_transpose/ReadVariableOpReadVariableOp9conv2d_transpose_conv2d_transpose_readvariableop_resource*'
_output_shapes
:x�*
dtype0�
&conv2d_transpose/conv2d_transpose/CastCast8conv2d_transpose/conv2d_transpose/ReadVariableOp:value:0*

DstT0*

SrcT0*'
_output_shapes
:x��
!conv2d_transpose/conv2d_transposeConv2DBackpropInputconv2d_transpose/stack:output:0*conv2d_transpose/conv2d_transpose/Cast:y:0conv2d_11/mul_3:z:0*
T0*/
_output_shapes
:���������xxx*
paddingSAME*
strides
�
'conv2d_transpose/BiasAdd/ReadVariableOpReadVariableOp0conv2d_transpose_biasadd_readvariableop_resource*
_output_shapes
:x*
dtype0�
conv2d_transpose/BiasAdd/CastCast/conv2d_transpose/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:x�
conv2d_transpose/BiasAddBiasAdd*conv2d_transpose/conv2d_transpose:output:0!conv2d_transpose/BiasAdd/Cast:y:0*
T0*/
_output_shapes
:���������xxxY
conv2d_transpose/mul/xConst*
_output_shapes
: *
dtype0*
value
B j�p�
conv2d_transpose/mulMulconv2d_transpose/mul/x:output:0!conv2d_transpose/BiasAdd:output:0*
T0*/
_output_shapes
:���������xxxZ
conv2d_transpose/Pow/yConst*
_output_shapes
: *
dtype0*
valueB	 j���
conv2d_transpose/PowPow!conv2d_transpose/BiasAdd:output:0conv2d_transpose/Pow/y:output:0*
T0*/
_output_shapes
:���������xxx[
conv2d_transpose/mul_1/xConst*
_output_shapes
: *
dtype0*
value
B j�S�
conv2d_transpose/mul_1Mul!conv2d_transpose/mul_1/x:output:0conv2d_transpose/Pow:z:0*
T0*/
_output_shapes
:���������xxx�
conv2d_transpose/addAddV2!conv2d_transpose/BiasAdd:output:0conv2d_transpose/mul_1:z:0*
T0*/
_output_shapes
:���������xxx[
conv2d_transpose/mul_2/xConst*
_output_shapes
: *
dtype0*
value
B j�t�
conv2d_transpose/mul_2Mul!conv2d_transpose/mul_2/x:output:0conv2d_transpose/add:z:0*
T0*/
_output_shapes
:���������xxxs
conv2d_transpose/TanhTanhconv2d_transpose/mul_2:z:0*
T0*/
_output_shapes
:���������xxx[
conv2d_transpose/add_1/xConst*
_output_shapes
: *
dtype0*
value
B j�x�
conv2d_transpose/add_1AddV2!conv2d_transpose/add_1/x:output:0conv2d_transpose/Tanh:y:0*
T0*/
_output_shapes
:���������xxx�
conv2d_transpose/mul_3Mulconv2d_transpose/mul:z:0conv2d_transpose/add_1:z:0*
T0*/
_output_shapes
:���������xxxz
add/addAddV2conv2d_transpose/mul_3:z:0conv2d_7/mul_3:z:0*
T0*/
_output_shapes
:���������xxx�
conv2d_12/Conv2D/ReadVariableOpReadVariableOp(conv2d_12_conv2d_readvariableop_resource*&
_output_shapes
:xx*
dtype0�
conv2d_12/Conv2D/CastCast'conv2d_12/Conv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*&
_output_shapes
:xx�
conv2d_12/Conv2DConv2Dadd/add:z:0conv2d_12/Conv2D/Cast:y:0*
T0*/
_output_shapes
:���������xxx*
paddingSAME*
strides
�
 conv2d_12/BiasAdd/ReadVariableOpReadVariableOp)conv2d_12_biasadd_readvariableop_resource*
_output_shapes
:x*
dtype0|
conv2d_12/BiasAdd/CastCast(conv2d_12/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:x�
conv2d_12/BiasAddBiasAddconv2d_12/Conv2D:output:0conv2d_12/BiasAdd/Cast:y:0*
T0*/
_output_shapes
:���������xxxR
conv2d_12/mul/xConst*
_output_shapes
: *
dtype0*
value
B j�p�
conv2d_12/mulMulconv2d_12/mul/x:output:0conv2d_12/BiasAdd:output:0*
T0*/
_output_shapes
:���������xxxS
conv2d_12/Pow/yConst*
_output_shapes
: *
dtype0*
valueB	 j���
conv2d_12/PowPowconv2d_12/BiasAdd:output:0conv2d_12/Pow/y:output:0*
T0*/
_output_shapes
:���������xxxT
conv2d_12/mul_1/xConst*
_output_shapes
: *
dtype0*
value
B j�S
conv2d_12/mul_1Mulconv2d_12/mul_1/x:output:0conv2d_12/Pow:z:0*
T0*/
_output_shapes
:���������xxx�
conv2d_12/addAddV2conv2d_12/BiasAdd:output:0conv2d_12/mul_1:z:0*
T0*/
_output_shapes
:���������xxxT
conv2d_12/mul_2/xConst*
_output_shapes
: *
dtype0*
value
B j�t
conv2d_12/mul_2Mulconv2d_12/mul_2/x:output:0conv2d_12/add:z:0*
T0*/
_output_shapes
:���������xxxe
conv2d_12/TanhTanhconv2d_12/mul_2:z:0*
T0*/
_output_shapes
:���������xxxT
conv2d_12/add_1/xConst*
_output_shapes
: *
dtype0*
value
B j�x�
conv2d_12/add_1AddV2conv2d_12/add_1/x:output:0conv2d_12/Tanh:y:0*
T0*/
_output_shapes
:���������xxxx
conv2d_12/mul_3Mulconv2d_12/mul:z:0conv2d_12/add_1:z:0*
T0*/
_output_shapes
:���������xxx�
conv2d_13/Conv2D/ReadVariableOpReadVariableOp(conv2d_13_conv2d_readvariableop_resource*&
_output_shapes
:xx*
dtype0�
conv2d_13/Conv2D/CastCast'conv2d_13/Conv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*&
_output_shapes
:xx�
conv2d_13/Conv2DConv2Dconv2d_12/mul_3:z:0conv2d_13/Conv2D/Cast:y:0*
T0*/
_output_shapes
:���������xxx*
paddingSAME*
strides
�
 conv2d_13/BiasAdd/ReadVariableOpReadVariableOp)conv2d_13_biasadd_readvariableop_resource*
_output_shapes
:x*
dtype0|
conv2d_13/BiasAdd/CastCast(conv2d_13/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:x�
conv2d_13/BiasAddBiasAddconv2d_13/Conv2D:output:0conv2d_13/BiasAdd/Cast:y:0*
T0*/
_output_shapes
:���������xxxR
conv2d_13/mul/xConst*
_output_shapes
: *
dtype0*
value
B j�p�
conv2d_13/mulMulconv2d_13/mul/x:output:0conv2d_13/BiasAdd:output:0*
T0*/
_output_shapes
:���������xxxS
conv2d_13/Pow/yConst*
_output_shapes
: *
dtype0*
valueB	 j���
conv2d_13/PowPowconv2d_13/BiasAdd:output:0conv2d_13/Pow/y:output:0*
T0*/
_output_shapes
:���������xxxT
conv2d_13/mul_1/xConst*
_output_shapes
: *
dtype0*
value
B j�S
conv2d_13/mul_1Mulconv2d_13/mul_1/x:output:0conv2d_13/Pow:z:0*
T0*/
_output_shapes
:���������xxx�
conv2d_13/addAddV2conv2d_13/BiasAdd:output:0conv2d_13/mul_1:z:0*
T0*/
_output_shapes
:���������xxxT
conv2d_13/mul_2/xConst*
_output_shapes
: *
dtype0*
value
B j�t
conv2d_13/mul_2Mulconv2d_13/mul_2/x:output:0conv2d_13/add:z:0*
T0*/
_output_shapes
:���������xxxe
conv2d_13/TanhTanhconv2d_13/mul_2:z:0*
T0*/
_output_shapes
:���������xxxT
conv2d_13/add_1/xConst*
_output_shapes
: *
dtype0*
value
B j�x�
conv2d_13/add_1AddV2conv2d_13/add_1/x:output:0conv2d_13/Tanh:y:0*
T0*/
_output_shapes
:���������xxxx
conv2d_13/mul_3Mulconv2d_13/mul:z:0conv2d_13/add_1:z:0*
T0*/
_output_shapes
:���������xxx[
conv2d_transpose_1/ShapeShapeconv2d_13/mul_3:z:0*
T0*
_output_shapes
:p
&conv2d_transpose_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv2d_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv2d_transpose_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
 conv2d_transpose_1/strided_sliceStridedSlice!conv2d_transpose_1/Shape:output:0/conv2d_transpose_1/strided_slice/stack:output:01conv2d_transpose_1/strided_slice/stack_1:output:01conv2d_transpose_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
conv2d_transpose_1/stack/1Const*
_output_shapes
: *
dtype0*
value
B :�]
conv2d_transpose_1/stack/2Const*
_output_shapes
: *
dtype0*
value
B :�\
conv2d_transpose_1/stack/3Const*
_output_shapes
: *
dtype0*
value	B :<�
conv2d_transpose_1/stackPack)conv2d_transpose_1/strided_slice:output:0#conv2d_transpose_1/stack/1:output:0#conv2d_transpose_1/stack/2:output:0#conv2d_transpose_1/stack/3:output:0*
N*
T0*
_output_shapes
:r
(conv2d_transpose_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*conv2d_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*conv2d_transpose_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
"conv2d_transpose_1/strided_slice_1StridedSlice!conv2d_transpose_1/stack:output:01conv2d_transpose_1/strided_slice_1/stack:output:03conv2d_transpose_1/strided_slice_1/stack_1:output:03conv2d_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
2conv2d_transpose_1/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_1_conv2d_transpose_readvariableop_resource*&
_output_shapes
:<x*
dtype0�
(conv2d_transpose_1/conv2d_transpose/CastCast:conv2d_transpose_1/conv2d_transpose/ReadVariableOp:value:0*

DstT0*

SrcT0*&
_output_shapes
:<x�
#conv2d_transpose_1/conv2d_transposeConv2DBackpropInput!conv2d_transpose_1/stack:output:0,conv2d_transpose_1/conv2d_transpose/Cast:y:0conv2d_13/mul_3:z:0*
T0*1
_output_shapes
:�����������<*
paddingSAME*
strides
�
)conv2d_transpose_1/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_1_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype0�
conv2d_transpose_1/BiasAdd/CastCast1conv2d_transpose_1/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:<�
conv2d_transpose_1/BiasAddBiasAdd,conv2d_transpose_1/conv2d_transpose:output:0#conv2d_transpose_1/BiasAdd/Cast:y:0*
T0*1
_output_shapes
:�����������<[
conv2d_transpose_1/mul/xConst*
_output_shapes
: *
dtype0*
value
B j�p�
conv2d_transpose_1/mulMul!conv2d_transpose_1/mul/x:output:0#conv2d_transpose_1/BiasAdd:output:0*
T0*1
_output_shapes
:�����������<\
conv2d_transpose_1/Pow/yConst*
_output_shapes
: *
dtype0*
valueB	 j���
conv2d_transpose_1/PowPow#conv2d_transpose_1/BiasAdd:output:0!conv2d_transpose_1/Pow/y:output:0*
T0*1
_output_shapes
:�����������<]
conv2d_transpose_1/mul_1/xConst*
_output_shapes
: *
dtype0*
value
B j�S�
conv2d_transpose_1/mul_1Mul#conv2d_transpose_1/mul_1/x:output:0conv2d_transpose_1/Pow:z:0*
T0*1
_output_shapes
:�����������<�
conv2d_transpose_1/addAddV2#conv2d_transpose_1/BiasAdd:output:0conv2d_transpose_1/mul_1:z:0*
T0*1
_output_shapes
:�����������<]
conv2d_transpose_1/mul_2/xConst*
_output_shapes
: *
dtype0*
value
B j�t�
conv2d_transpose_1/mul_2Mul#conv2d_transpose_1/mul_2/x:output:0conv2d_transpose_1/add:z:0*
T0*1
_output_shapes
:�����������<y
conv2d_transpose_1/TanhTanhconv2d_transpose_1/mul_2:z:0*
T0*1
_output_shapes
:�����������<]
conv2d_transpose_1/add_1/xConst*
_output_shapes
: *
dtype0*
value
B j�x�
conv2d_transpose_1/add_1AddV2#conv2d_transpose_1/add_1/x:output:0conv2d_transpose_1/Tanh:y:0*
T0*1
_output_shapes
:�����������<�
conv2d_transpose_1/mul_3Mulconv2d_transpose_1/mul:z:0conv2d_transpose_1/add_1:z:0*
T0*1
_output_shapes
:�����������<�
	add_1/addAddV2conv2d_transpose_1/mul_3:z:0conv2d_5/mul_3:z:0*
T0*1
_output_shapes
:�����������<�
conv2d_14/Conv2D/ReadVariableOpReadVariableOp(conv2d_14_conv2d_readvariableop_resource*&
_output_shapes
:<<*
dtype0�
conv2d_14/Conv2D/CastCast'conv2d_14/Conv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*&
_output_shapes
:<<�
conv2d_14/Conv2DConv2Dadd_1/add:z:0conv2d_14/Conv2D/Cast:y:0*
T0*1
_output_shapes
:�����������<*
paddingSAME*
strides
�
 conv2d_14/BiasAdd/ReadVariableOpReadVariableOp)conv2d_14_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype0|
conv2d_14/BiasAdd/CastCast(conv2d_14/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:<�
conv2d_14/BiasAddBiasAddconv2d_14/Conv2D:output:0conv2d_14/BiasAdd/Cast:y:0*
T0*1
_output_shapes
:�����������<R
conv2d_14/mul/xConst*
_output_shapes
: *
dtype0*
value
B j�p�
conv2d_14/mulMulconv2d_14/mul/x:output:0conv2d_14/BiasAdd:output:0*
T0*1
_output_shapes
:�����������<S
conv2d_14/Pow/yConst*
_output_shapes
: *
dtype0*
valueB	 j���
conv2d_14/PowPowconv2d_14/BiasAdd:output:0conv2d_14/Pow/y:output:0*
T0*1
_output_shapes
:�����������<T
conv2d_14/mul_1/xConst*
_output_shapes
: *
dtype0*
value
B j�S�
conv2d_14/mul_1Mulconv2d_14/mul_1/x:output:0conv2d_14/Pow:z:0*
T0*1
_output_shapes
:�����������<�
conv2d_14/addAddV2conv2d_14/BiasAdd:output:0conv2d_14/mul_1:z:0*
T0*1
_output_shapes
:�����������<T
conv2d_14/mul_2/xConst*
_output_shapes
: *
dtype0*
value
B j�t�
conv2d_14/mul_2Mulconv2d_14/mul_2/x:output:0conv2d_14/add:z:0*
T0*1
_output_shapes
:�����������<g
conv2d_14/TanhTanhconv2d_14/mul_2:z:0*
T0*1
_output_shapes
:�����������<T
conv2d_14/add_1/xConst*
_output_shapes
: *
dtype0*
value
B j�x�
conv2d_14/add_1AddV2conv2d_14/add_1/x:output:0conv2d_14/Tanh:y:0*
T0*1
_output_shapes
:�����������<z
conv2d_14/mul_3Mulconv2d_14/mul:z:0conv2d_14/add_1:z:0*
T0*1
_output_shapes
:�����������<�
conv2d_15/Conv2D/ReadVariableOpReadVariableOp(conv2d_15_conv2d_readvariableop_resource*&
_output_shapes
:<<*
dtype0�
conv2d_15/Conv2D/CastCast'conv2d_15/Conv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*&
_output_shapes
:<<�
conv2d_15/Conv2DConv2Dconv2d_14/mul_3:z:0conv2d_15/Conv2D/Cast:y:0*
T0*1
_output_shapes
:�����������<*
paddingSAME*
strides
�
 conv2d_15/BiasAdd/ReadVariableOpReadVariableOp)conv2d_15_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype0|
conv2d_15/BiasAdd/CastCast(conv2d_15/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:<�
conv2d_15/BiasAddBiasAddconv2d_15/Conv2D:output:0conv2d_15/BiasAdd/Cast:y:0*
T0*1
_output_shapes
:�����������<R
conv2d_15/mul/xConst*
_output_shapes
: *
dtype0*
value
B j�p�
conv2d_15/mulMulconv2d_15/mul/x:output:0conv2d_15/BiasAdd:output:0*
T0*1
_output_shapes
:�����������<S
conv2d_15/Pow/yConst*
_output_shapes
: *
dtype0*
valueB	 j���
conv2d_15/PowPowconv2d_15/BiasAdd:output:0conv2d_15/Pow/y:output:0*
T0*1
_output_shapes
:�����������<T
conv2d_15/mul_1/xConst*
_output_shapes
: *
dtype0*
value
B j�S�
conv2d_15/mul_1Mulconv2d_15/mul_1/x:output:0conv2d_15/Pow:z:0*
T0*1
_output_shapes
:�����������<�
conv2d_15/addAddV2conv2d_15/BiasAdd:output:0conv2d_15/mul_1:z:0*
T0*1
_output_shapes
:�����������<T
conv2d_15/mul_2/xConst*
_output_shapes
: *
dtype0*
value
B j�t�
conv2d_15/mul_2Mulconv2d_15/mul_2/x:output:0conv2d_15/add:z:0*
T0*1
_output_shapes
:�����������<g
conv2d_15/TanhTanhconv2d_15/mul_2:z:0*
T0*1
_output_shapes
:�����������<T
conv2d_15/add_1/xConst*
_output_shapes
: *
dtype0*
value
B j�x�
conv2d_15/add_1AddV2conv2d_15/add_1/x:output:0conv2d_15/Tanh:y:0*
T0*1
_output_shapes
:�����������<z
conv2d_15/mul_3Mulconv2d_15/mul:z:0conv2d_15/add_1:z:0*
T0*1
_output_shapes
:�����������<�
conv2d_16/Conv2D/ReadVariableOpReadVariableOp(conv2d_16_conv2d_readvariableop_resource*&
_output_shapes
:<*
dtype0�
conv2d_16/Conv2D/CastCast'conv2d_16/Conv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*&
_output_shapes
:<�
conv2d_16/Conv2DConv2Dconv2d_15/mul_3:z:0conv2d_16/Conv2D/Cast:y:0*
T0*1
_output_shapes
:�����������*
paddingSAME*
strides
�
 conv2d_16/BiasAdd/ReadVariableOpReadVariableOp)conv2d_16_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0|
conv2d_16/BiasAdd/CastCast(conv2d_16/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:�
conv2d_16/BiasAddBiasAddconv2d_16/Conv2D:output:0conv2d_16/BiasAdd/Cast:y:0*
T0*1
_output_shapes
:�����������y

add_2/CastCastconv2d_16/BiasAdd:output:0*

DstT0*

SrcT0*1
_output_shapes
:�����������f
	add_2/addAddV2add_2/Cast:y:0inputs*
T0*1
_output_shapes
:�����������f
IdentityIdentityadd_2/add:z:0^NoOp*
T0*1
_output_shapes
:������������
NoOpNoOp^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp!^conv2d_10/BiasAdd/ReadVariableOp ^conv2d_10/Conv2D/ReadVariableOp!^conv2d_11/BiasAdd/ReadVariableOp ^conv2d_11/Conv2D/ReadVariableOp!^conv2d_12/BiasAdd/ReadVariableOp ^conv2d_12/Conv2D/ReadVariableOp!^conv2d_13/BiasAdd/ReadVariableOp ^conv2d_13/Conv2D/ReadVariableOp!^conv2d_14/BiasAdd/ReadVariableOp ^conv2d_14/Conv2D/ReadVariableOp!^conv2d_15/BiasAdd/ReadVariableOp ^conv2d_15/Conv2D/ReadVariableOp!^conv2d_16/BiasAdd/ReadVariableOp ^conv2d_16/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp ^conv2d_4/BiasAdd/ReadVariableOp^conv2d_4/Conv2D/ReadVariableOp ^conv2d_5/BiasAdd/ReadVariableOp^conv2d_5/Conv2D/ReadVariableOp ^conv2d_6/BiasAdd/ReadVariableOp^conv2d_6/Conv2D/ReadVariableOp ^conv2d_7/BiasAdd/ReadVariableOp^conv2d_7/Conv2D/ReadVariableOp ^conv2d_8/BiasAdd/ReadVariableOp^conv2d_8/Conv2D/ReadVariableOp ^conv2d_9/BiasAdd/ReadVariableOp^conv2d_9/Conv2D/ReadVariableOp(^conv2d_transpose/BiasAdd/ReadVariableOp1^conv2d_transpose/conv2d_transpose/ReadVariableOp*^conv2d_transpose_1/BiasAdd/ReadVariableOp3^conv2d_transpose_1/conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*|
_input_shapesk
i:�����������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2D
 conv2d_10/BiasAdd/ReadVariableOp conv2d_10/BiasAdd/ReadVariableOp2B
conv2d_10/Conv2D/ReadVariableOpconv2d_10/Conv2D/ReadVariableOp2D
 conv2d_11/BiasAdd/ReadVariableOp conv2d_11/BiasAdd/ReadVariableOp2B
conv2d_11/Conv2D/ReadVariableOpconv2d_11/Conv2D/ReadVariableOp2D
 conv2d_12/BiasAdd/ReadVariableOp conv2d_12/BiasAdd/ReadVariableOp2B
conv2d_12/Conv2D/ReadVariableOpconv2d_12/Conv2D/ReadVariableOp2D
 conv2d_13/BiasAdd/ReadVariableOp conv2d_13/BiasAdd/ReadVariableOp2B
conv2d_13/Conv2D/ReadVariableOpconv2d_13/Conv2D/ReadVariableOp2D
 conv2d_14/BiasAdd/ReadVariableOp conv2d_14/BiasAdd/ReadVariableOp2B
conv2d_14/Conv2D/ReadVariableOpconv2d_14/Conv2D/ReadVariableOp2D
 conv2d_15/BiasAdd/ReadVariableOp conv2d_15/BiasAdd/ReadVariableOp2B
conv2d_15/Conv2D/ReadVariableOpconv2d_15/Conv2D/ReadVariableOp2D
 conv2d_16/BiasAdd/ReadVariableOp conv2d_16/BiasAdd/ReadVariableOp2B
conv2d_16/Conv2D/ReadVariableOpconv2d_16/Conv2D/ReadVariableOp2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2B
conv2d_3/BiasAdd/ReadVariableOpconv2d_3/BiasAdd/ReadVariableOp2@
conv2d_3/Conv2D/ReadVariableOpconv2d_3/Conv2D/ReadVariableOp2B
conv2d_4/BiasAdd/ReadVariableOpconv2d_4/BiasAdd/ReadVariableOp2@
conv2d_4/Conv2D/ReadVariableOpconv2d_4/Conv2D/ReadVariableOp2B
conv2d_5/BiasAdd/ReadVariableOpconv2d_5/BiasAdd/ReadVariableOp2@
conv2d_5/Conv2D/ReadVariableOpconv2d_5/Conv2D/ReadVariableOp2B
conv2d_6/BiasAdd/ReadVariableOpconv2d_6/BiasAdd/ReadVariableOp2@
conv2d_6/Conv2D/ReadVariableOpconv2d_6/Conv2D/ReadVariableOp2B
conv2d_7/BiasAdd/ReadVariableOpconv2d_7/BiasAdd/ReadVariableOp2@
conv2d_7/Conv2D/ReadVariableOpconv2d_7/Conv2D/ReadVariableOp2B
conv2d_8/BiasAdd/ReadVariableOpconv2d_8/BiasAdd/ReadVariableOp2@
conv2d_8/Conv2D/ReadVariableOpconv2d_8/Conv2D/ReadVariableOp2B
conv2d_9/BiasAdd/ReadVariableOpconv2d_9/BiasAdd/ReadVariableOp2@
conv2d_9/Conv2D/ReadVariableOpconv2d_9/Conv2D/ReadVariableOp2R
'conv2d_transpose/BiasAdd/ReadVariableOp'conv2d_transpose/BiasAdd/ReadVariableOp2d
0conv2d_transpose/conv2d_transpose/ReadVariableOp0conv2d_transpose/conv2d_transpose/ReadVariableOp2V
)conv2d_transpose_1/BiasAdd/ReadVariableOp)conv2d_transpose_1/BiasAdd/ReadVariableOp2h
2conv2d_transpose_1/conv2d_transpose/ReadVariableOp2conv2d_transpose_1/conv2d_transpose/ReadVariableOp:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
�
(__inference_conv2d_8_layer_call_fn_55984

inputs"
unknown:x�
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������<<�*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_conv2d_8_layer_call_and_return_conditional_losses_53446x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:���������<<�`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������<<x: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������<<x
 
_user_specified_nameinputs
�
Q
%__inference_add_1_layer_call_fn_56311
inputs_0
inputs_1
identity�
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������<* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *I
fDRB
@__inference_add_1_layer_call_and_return_conditional_losses_53631j
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:�����������<"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*M
_input_shapes<
::�����������<:�����������<:[ W
1
_output_shapes
:�����������<
"
_user_specified_name
inputs/0:[W
1
_output_shapes
:�����������<
"
_user_specified_name
inputs/1
�
O
#__inference_add_layer_call_fn_56174
inputs_0
inputs_1
identity�
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������xxx* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *G
fBR@
>__inference_add_layer_call_and_return_conditional_losses_53556h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������xxx"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:���������xxx:���������xxx:Y U
/
_output_shapes
:���������xxx
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:���������xxx
"
_user_specified_name
inputs/1
�
�
)__inference_conv2d_10_layer_call_fn_56052

inputs#
unknown:��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������<<�*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv2d_10_layer_call_and_return_conditional_losses_53508x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:���������<<�`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������<<�: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:���������<<�
 
_user_specified_nameinputs
�
�
)__inference_conv2d_13_layer_call_fn_56223

inputs!
unknown:xx
	unknown_0:x
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������xxx*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv2d_13_layer_call_and_return_conditional_losses_53614w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������xxx`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������xxx: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������xxx
 
_user_specified_nameinputs
�
�
C__inference_conv2d_6_layer_call_and_return_conditional_losses_53383

inputs8
conv2d_readvariableop_resource:<x-
biasadd_readvariableop_resource:x
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:<x*
dtype0r
Conv2D/CastCastConv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*&
_output_shapes
:<x�
Conv2DConv2DinputsConv2D/Cast:y:0*
T0*/
_output_shapes
:���������xxx*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:x*
dtype0h
BiasAdd/CastCastBiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:xo
BiasAddBiasAddConv2D:output:0BiasAdd/Cast:y:0*
T0*/
_output_shapes
:���������xxxH
mul/xConst*
_output_shapes
: *
dtype0*
value
B j�pf
mulMulmul/x:output:0BiasAdd:output:0*
T0*/
_output_shapes
:���������xxxI
Pow/yConst*
_output_shapes
: *
dtype0*
valueB	 j��f
PowPowBiasAdd:output:0Pow/y:output:0*
T0*/
_output_shapes
:���������xxxJ
mul_1/xConst*
_output_shapes
: *
dtype0*
value
B j�Sa
mul_1Mulmul_1/x:output:0Pow:z:0*
T0*/
_output_shapes
:���������xxxc
addAddV2BiasAdd:output:0	mul_1:z:0*
T0*/
_output_shapes
:���������xxxJ
mul_2/xConst*
_output_shapes
: *
dtype0*
value
B j�ta
mul_2Mulmul_2/x:output:0add:z:0*
T0*/
_output_shapes
:���������xxxQ
TanhTanh	mul_2:z:0*
T0*/
_output_shapes
:���������xxxJ
add_1/xConst*
_output_shapes
: *
dtype0*
value
B j�xd
add_1AddV2add_1/x:output:0Tanh:y:0*
T0*/
_output_shapes
:���������xxxZ
mul_3Mulmul:z:0	add_1:z:0*
T0*/
_output_shapes
:���������xxx`
IdentityIdentity	mul_3:z:0^NoOp*
T0*/
_output_shapes
:���������xxxw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������xx<: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������xx<
 
_user_specified_nameinputs
�
�
(__inference_conv2d_3_layer_call_fn_55781

inputs!
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_conv2d_3_layer_call_and_return_conditional_losses_53279y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:�����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:�����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
�
A__inference_conv2d_layer_call_and_return_conditional_losses_53186

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0r
Conv2D/CastCastConv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*&
_output_shapes
:�
Conv2DConv2DinputsConv2D/Cast:y:0*
T0*1
_output_shapes
:�����������*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0h
BiasAdd/CastCastBiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:q
BiasAddBiasAddConv2D:output:0BiasAdd/Cast:y:0*
T0*1
_output_shapes
:�����������H
mul/xConst*
_output_shapes
: *
dtype0*
value
B j�ph
mulMulmul/x:output:0BiasAdd:output:0*
T0*1
_output_shapes
:�����������I
Pow/yConst*
_output_shapes
: *
dtype0*
valueB	 j��h
PowPowBiasAdd:output:0Pow/y:output:0*
T0*1
_output_shapes
:�����������J
mul_1/xConst*
_output_shapes
: *
dtype0*
value
B j�Sc
mul_1Mulmul_1/x:output:0Pow:z:0*
T0*1
_output_shapes
:�����������e
addAddV2BiasAdd:output:0	mul_1:z:0*
T0*1
_output_shapes
:�����������J
mul_2/xConst*
_output_shapes
: *
dtype0*
value
B j�tc
mul_2Mulmul_2/x:output:0add:z:0*
T0*1
_output_shapes
:�����������S
TanhTanh	mul_2:z:0*
T0*1
_output_shapes
:�����������J
add_1/xConst*
_output_shapes
: *
dtype0*
value
B j�xf
add_1AddV2add_1/x:output:0Tanh:y:0*
T0*1
_output_shapes
:�����������\
mul_3Mulmul:z:0	add_1:z:0*
T0*1
_output_shapes
:�����������b
IdentityIdentity	mul_3:z:0^NoOp*
T0*1
_output_shapes
:�����������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:�����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
p
F__inference_concatenate_layer_call_and_return_conditional_losses_53293

inputs
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*1
_output_shapes
:�����������a
IdentityIdentityconcat:output:0*
T0*1
_output_shapes
:�����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*M
_input_shapes<
::�����������:�����������:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs:YU
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
r
F__inference_concatenate_layer_call_and_return_conditional_losses_55819
inputs_0
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*1
_output_shapes
:�����������a
IdentityIdentityconcat:output:0*
T0*1
_output_shapes
:�����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*M
_input_shapes<
::�����������:�����������:[ W
1
_output_shapes
:�����������
"
_user_specified_name
inputs/0:[W
1
_output_shapes
:�����������
"
_user_specified_name
inputs/1
�
�

%__inference_model_layer_call_fn_54737

inputs!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:#
	unknown_3:
	unknown_4:#
	unknown_5:
	unknown_6:#
	unknown_7:<
	unknown_8:<#
	unknown_9:<<

unknown_10:<$

unknown_11:<x

unknown_12:x$

unknown_13:xx

unknown_14:x%

unknown_15:x�

unknown_16:	�&

unknown_17:��

unknown_18:	�&

unknown_19:��

unknown_20:	�&

unknown_21:��

unknown_22:	�%

unknown_23:x�

unknown_24:x$

unknown_25:xx

unknown_26:x$

unknown_27:xx

unknown_28:x$

unknown_29:<x

unknown_30:<$

unknown_31:<<

unknown_32:<$

unknown_33:<<

unknown_34:<$

unknown_35:<

unknown_36:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36*2
Tin+
)2'*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������*H
_read_only_resource_inputs*
(&	
 !"#$%&*0
config_proto 

CPU

GPU2*0J 8� *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_53723y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:�����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*|
_input_shapesk
i:�����������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
�
C__inference_conv2d_7_layer_call_and_return_conditional_losses_53414

inputs8
conv2d_readvariableop_resource:xx-
biasadd_readvariableop_resource:x
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:xx*
dtype0r
Conv2D/CastCastConv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*&
_output_shapes
:xx�
Conv2DConv2DinputsConv2D/Cast:y:0*
T0*/
_output_shapes
:���������xxx*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:x*
dtype0h
BiasAdd/CastCastBiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:xo
BiasAddBiasAddConv2D:output:0BiasAdd/Cast:y:0*
T0*/
_output_shapes
:���������xxxH
mul/xConst*
_output_shapes
: *
dtype0*
value
B j�pf
mulMulmul/x:output:0BiasAdd:output:0*
T0*/
_output_shapes
:���������xxxI
Pow/yConst*
_output_shapes
: *
dtype0*
valueB	 j��f
PowPowBiasAdd:output:0Pow/y:output:0*
T0*/
_output_shapes
:���������xxxJ
mul_1/xConst*
_output_shapes
: *
dtype0*
value
B j�Sa
mul_1Mulmul_1/x:output:0Pow:z:0*
T0*/
_output_shapes
:���������xxxc
addAddV2BiasAdd:output:0	mul_1:z:0*
T0*/
_output_shapes
:���������xxxJ
mul_2/xConst*
_output_shapes
: *
dtype0*
value
B j�ta
mul_2Mulmul_2/x:output:0add:z:0*
T0*/
_output_shapes
:���������xxxQ
TanhTanh	mul_2:z:0*
T0*/
_output_shapes
:���������xxxJ
add_1/xConst*
_output_shapes
: *
dtype0*
value
B j�xd
add_1AddV2add_1/x:output:0Tanh:y:0*
T0*/
_output_shapes
:���������xxxZ
mul_3Mulmul:z:0	add_1:z:0*
T0*/
_output_shapes
:���������xxx`
IdentityIdentity	mul_3:z:0^NoOp*
T0*/
_output_shapes
:���������xxxw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������xxx: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������xxx
 
_user_specified_nameinputs
�
�
D__inference_conv2d_13_layer_call_and_return_conditional_losses_56248

inputs8
conv2d_readvariableop_resource:xx-
biasadd_readvariableop_resource:x
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:xx*
dtype0r
Conv2D/CastCastConv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*&
_output_shapes
:xx�
Conv2DConv2DinputsConv2D/Cast:y:0*
T0*/
_output_shapes
:���������xxx*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:x*
dtype0h
BiasAdd/CastCastBiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:xo
BiasAddBiasAddConv2D:output:0BiasAdd/Cast:y:0*
T0*/
_output_shapes
:���������xxxH
mul/xConst*
_output_shapes
: *
dtype0*
value
B j�pf
mulMulmul/x:output:0BiasAdd:output:0*
T0*/
_output_shapes
:���������xxxI
Pow/yConst*
_output_shapes
: *
dtype0*
valueB	 j��f
PowPowBiasAdd:output:0Pow/y:output:0*
T0*/
_output_shapes
:���������xxxJ
mul_1/xConst*
_output_shapes
: *
dtype0*
value
B j�Sa
mul_1Mulmul_1/x:output:0Pow:z:0*
T0*/
_output_shapes
:���������xxxc
addAddV2BiasAdd:output:0	mul_1:z:0*
T0*/
_output_shapes
:���������xxxJ
mul_2/xConst*
_output_shapes
: *
dtype0*
value
B j�ta
mul_2Mulmul_2/x:output:0add:z:0*
T0*/
_output_shapes
:���������xxxQ
TanhTanh	mul_2:z:0*
T0*/
_output_shapes
:���������xxxJ
add_1/xConst*
_output_shapes
: *
dtype0*
value
B j�xd
add_1AddV2add_1/x:output:0Tanh:y:0*
T0*/
_output_shapes
:���������xxxZ
mul_3Mulmul:z:0	add_1:z:0*
T0*/
_output_shapes
:���������xxx`
IdentityIdentity	mul_3:z:0^NoOp*
T0*/
_output_shapes
:���������xxxw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������xxx: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������xxx
 
_user_specified_nameinputs
�
�

%__inference_model_layer_call_fn_54351
input_1!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:#
	unknown_3:
	unknown_4:#
	unknown_5:
	unknown_6:#
	unknown_7:<
	unknown_8:<#
	unknown_9:<<

unknown_10:<$

unknown_11:<x

unknown_12:x$

unknown_13:xx

unknown_14:x%

unknown_15:x�

unknown_16:	�&

unknown_17:��

unknown_18:	�&

unknown_19:��

unknown_20:	�&

unknown_21:��

unknown_22:	�%

unknown_23:x�

unknown_24:x$

unknown_25:xx

unknown_26:x$

unknown_27:xx

unknown_28:x$

unknown_29:<x

unknown_30:<$

unknown_31:<<

unknown_32:<$

unknown_33:<<

unknown_34:<$

unknown_35:<

unknown_36:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36*2
Tin+
)2'*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������*H
_read_only_resource_inputs*
(&	
 !"#$%&*0
config_proto 

CPU

GPU2*0J 8� *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_54191y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:�����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*|
_input_shapesk
i:�����������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:�����������
!
_user_specified_name	input_1
�
l
@__inference_add_2_layer_call_and_return_conditional_losses_56418
inputs_0
inputs_1
identity\
addAddV2inputs_0inputs_1*
T0*1
_output_shapes
:�����������Y
IdentityIdentityadd:z:0*
T0*1
_output_shapes
:�����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*M
_input_shapes<
::�����������:�����������:[ W
1
_output_shapes
:�����������
"
_user_specified_name
inputs/0:[W
1
_output_shapes
:�����������
"
_user_specified_name
inputs/1
�
�
D__inference_conv2d_16_layer_call_and_return_conditional_losses_53707

inputs8
conv2d_readvariableop_resource:<-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:<*
dtype0r
Conv2D/CastCastConv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*&
_output_shapes
:<�
Conv2DConv2DinputsConv2D/Cast:y:0*
T0*1
_output_shapes
:�����������*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0h
BiasAdd/CastCastBiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:q
BiasAddBiasAddConv2D:output:0BiasAdd/Cast:y:0*
T0*1
_output_shapes
:�����������i
IdentityIdentityBiasAdd:output:0^NoOp*
T0*1
_output_shapes
:�����������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:�����������<: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:�����������<
 
_user_specified_nameinputs
�
�
)__inference_conv2d_12_layer_call_fn_56189

inputs!
unknown:xx
	unknown_0:x
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������xxx*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv2d_12_layer_call_and_return_conditional_losses_53583w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������xxx`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������xxx: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������xxx
 
_user_specified_nameinputs
�
�
)__inference_conv2d_11_layer_call_fn_56086

inputs#
unknown:��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������<<�*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv2d_11_layer_call_and_return_conditional_losses_53539x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:���������<<�`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������<<�: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:���������<<�
 
_user_specified_nameinputs
�
�
C__inference_conv2d_3_layer_call_and_return_conditional_losses_55806

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0r
Conv2D/CastCastConv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*&
_output_shapes
:�
Conv2DConv2DinputsConv2D/Cast:y:0*
T0*1
_output_shapes
:�����������*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0h
BiasAdd/CastCastBiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:q
BiasAddBiasAddConv2D:output:0BiasAdd/Cast:y:0*
T0*1
_output_shapes
:�����������H
mul/xConst*
_output_shapes
: *
dtype0*
value
B j�ph
mulMulmul/x:output:0BiasAdd:output:0*
T0*1
_output_shapes
:�����������I
Pow/yConst*
_output_shapes
: *
dtype0*
valueB	 j��h
PowPowBiasAdd:output:0Pow/y:output:0*
T0*1
_output_shapes
:�����������J
mul_1/xConst*
_output_shapes
: *
dtype0*
value
B j�Sc
mul_1Mulmul_1/x:output:0Pow:z:0*
T0*1
_output_shapes
:�����������e
addAddV2BiasAdd:output:0	mul_1:z:0*
T0*1
_output_shapes
:�����������J
mul_2/xConst*
_output_shapes
: *
dtype0*
value
B j�tc
mul_2Mulmul_2/x:output:0add:z:0*
T0*1
_output_shapes
:�����������S
TanhTanh	mul_2:z:0*
T0*1
_output_shapes
:�����������J
add_1/xConst*
_output_shapes
: *
dtype0*
value
B j�xf
add_1AddV2add_1/x:output:0Tanh:y:0*
T0*1
_output_shapes
:�����������\
mul_3Mulmul:z:0	add_1:z:0*
T0*1
_output_shapes
:�����������b
IdentityIdentity	mul_3:z:0^NoOp*
T0*1
_output_shapes
:�����������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:�����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
�
C__inference_conv2d_9_layer_call_and_return_conditional_losses_53477

inputs:
conv2d_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0t
Conv2D/CastCastConv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*(
_output_shapes
:���
Conv2DConv2DinputsConv2D/Cast:y:0*
T0*0
_output_shapes
:���������<<�*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0i
BiasAdd/CastCastBiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes	
:�p
BiasAddBiasAddConv2D:output:0BiasAdd/Cast:y:0*
T0*0
_output_shapes
:���������<<�H
mul/xConst*
_output_shapes
: *
dtype0*
value
B j�pg
mulMulmul/x:output:0BiasAdd:output:0*
T0*0
_output_shapes
:���������<<�I
Pow/yConst*
_output_shapes
: *
dtype0*
valueB	 j��g
PowPowBiasAdd:output:0Pow/y:output:0*
T0*0
_output_shapes
:���������<<�J
mul_1/xConst*
_output_shapes
: *
dtype0*
value
B j�Sb
mul_1Mulmul_1/x:output:0Pow:z:0*
T0*0
_output_shapes
:���������<<�d
addAddV2BiasAdd:output:0	mul_1:z:0*
T0*0
_output_shapes
:���������<<�J
mul_2/xConst*
_output_shapes
: *
dtype0*
value
B j�tb
mul_2Mulmul_2/x:output:0add:z:0*
T0*0
_output_shapes
:���������<<�R
TanhTanh	mul_2:z:0*
T0*0
_output_shapes
:���������<<�J
add_1/xConst*
_output_shapes
: *
dtype0*
value
B j�xe
add_1AddV2add_1/x:output:0Tanh:y:0*
T0*0
_output_shapes
:���������<<�[
mul_3Mulmul:z:0	add_1:z:0*
T0*0
_output_shapes
:���������<<�a
IdentityIdentity	mul_3:z:0^NoOp*
T0*0
_output_shapes
:���������<<�w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������<<�: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:���������<<�
 
_user_specified_nameinputs
�
�
C__inference_conv2d_8_layer_call_and_return_conditional_losses_56009

inputs9
conv2d_readvariableop_resource:x�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:x�*
dtype0s
Conv2D/CastCastConv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*'
_output_shapes
:x��
Conv2DConv2DinputsConv2D/Cast:y:0*
T0*0
_output_shapes
:���������<<�*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0i
BiasAdd/CastCastBiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes	
:�p
BiasAddBiasAddConv2D:output:0BiasAdd/Cast:y:0*
T0*0
_output_shapes
:���������<<�H
mul/xConst*
_output_shapes
: *
dtype0*
value
B j�pg
mulMulmul/x:output:0BiasAdd:output:0*
T0*0
_output_shapes
:���������<<�I
Pow/yConst*
_output_shapes
: *
dtype0*
valueB	 j��g
PowPowBiasAdd:output:0Pow/y:output:0*
T0*0
_output_shapes
:���������<<�J
mul_1/xConst*
_output_shapes
: *
dtype0*
value
B j�Sb
mul_1Mulmul_1/x:output:0Pow:z:0*
T0*0
_output_shapes
:���������<<�d
addAddV2BiasAdd:output:0	mul_1:z:0*
T0*0
_output_shapes
:���������<<�J
mul_2/xConst*
_output_shapes
: *
dtype0*
value
B j�tb
mul_2Mulmul_2/x:output:0add:z:0*
T0*0
_output_shapes
:���������<<�R
TanhTanh	mul_2:z:0*
T0*0
_output_shapes
:���������<<�J
add_1/xConst*
_output_shapes
: *
dtype0*
value
B j�xe
add_1AddV2add_1/x:output:0Tanh:y:0*
T0*0
_output_shapes
:���������<<�[
mul_3Mulmul:z:0	add_1:z:0*
T0*0
_output_shapes
:���������<<�a
IdentityIdentity	mul_3:z:0^NoOp*
T0*0
_output_shapes
:���������<<�w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������<<x: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������<<x
 
_user_specified_nameinputs
�
�
0__inference_conv2d_transpose_layer_call_fn_56120

inputs"
unknown:x�
	unknown_0:x
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������x*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *T
fORM
K__inference_conv2d_transpose_layer_call_and_return_conditional_losses_53087�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������x`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,����������������������������: : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
D__inference_conv2d_10_layer_call_and_return_conditional_losses_56077

inputs:
conv2d_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0t
Conv2D/CastCastConv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*(
_output_shapes
:���
Conv2DConv2DinputsConv2D/Cast:y:0*
T0*0
_output_shapes
:���������<<�*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0i
BiasAdd/CastCastBiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes	
:�p
BiasAddBiasAddConv2D:output:0BiasAdd/Cast:y:0*
T0*0
_output_shapes
:���������<<�H
mul/xConst*
_output_shapes
: *
dtype0*
value
B j�pg
mulMulmul/x:output:0BiasAdd:output:0*
T0*0
_output_shapes
:���������<<�I
Pow/yConst*
_output_shapes
: *
dtype0*
valueB	 j��g
PowPowBiasAdd:output:0Pow/y:output:0*
T0*0
_output_shapes
:���������<<�J
mul_1/xConst*
_output_shapes
: *
dtype0*
value
B j�Sb
mul_1Mulmul_1/x:output:0Pow:z:0*
T0*0
_output_shapes
:���������<<�d
addAddV2BiasAdd:output:0	mul_1:z:0*
T0*0
_output_shapes
:���������<<�J
mul_2/xConst*
_output_shapes
: *
dtype0*
value
B j�tb
mul_2Mulmul_2/x:output:0add:z:0*
T0*0
_output_shapes
:���������<<�R
TanhTanh	mul_2:z:0*
T0*0
_output_shapes
:���������<<�J
add_1/xConst*
_output_shapes
: *
dtype0*
value
B j�xe
add_1AddV2add_1/x:output:0Tanh:y:0*
T0*0
_output_shapes
:���������<<�[
mul_3Mulmul:z:0	add_1:z:0*
T0*0
_output_shapes
:���������<<�a
IdentityIdentity	mul_3:z:0^NoOp*
T0*0
_output_shapes
:���������<<�w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������<<�: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:���������<<�
 
_user_specified_nameinputs
�
l
@__inference_add_1_layer_call_and_return_conditional_losses_56317
inputs_0
inputs_1
identity\
addAddV2inputs_0inputs_1*
T0*1
_output_shapes
:�����������<Y
IdentityIdentityadd:z:0*
T0*1
_output_shapes
:�����������<"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*M
_input_shapes<
::�����������<:�����������<:[ W
1
_output_shapes
:�����������<
"
_user_specified_name
inputs/0:[W
1
_output_shapes
:�����������<
"
_user_specified_name
inputs/1
�
�
C__inference_conv2d_9_layer_call_and_return_conditional_losses_56043

inputs:
conv2d_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0t
Conv2D/CastCastConv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*(
_output_shapes
:���
Conv2DConv2DinputsConv2D/Cast:y:0*
T0*0
_output_shapes
:���������<<�*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0i
BiasAdd/CastCastBiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes	
:�p
BiasAddBiasAddConv2D:output:0BiasAdd/Cast:y:0*
T0*0
_output_shapes
:���������<<�H
mul/xConst*
_output_shapes
: *
dtype0*
value
B j�pg
mulMulmul/x:output:0BiasAdd:output:0*
T0*0
_output_shapes
:���������<<�I
Pow/yConst*
_output_shapes
: *
dtype0*
valueB	 j��g
PowPowBiasAdd:output:0Pow/y:output:0*
T0*0
_output_shapes
:���������<<�J
mul_1/xConst*
_output_shapes
: *
dtype0*
value
B j�Sb
mul_1Mulmul_1/x:output:0Pow:z:0*
T0*0
_output_shapes
:���������<<�d
addAddV2BiasAdd:output:0	mul_1:z:0*
T0*0
_output_shapes
:���������<<�J
mul_2/xConst*
_output_shapes
: *
dtype0*
value
B j�tb
mul_2Mulmul_2/x:output:0add:z:0*
T0*0
_output_shapes
:���������<<�R
TanhTanh	mul_2:z:0*
T0*0
_output_shapes
:���������<<�J
add_1/xConst*
_output_shapes
: *
dtype0*
value
B j�xe
add_1AddV2add_1/x:output:0Tanh:y:0*
T0*0
_output_shapes
:���������<<�[
mul_3Mulmul:z:0	add_1:z:0*
T0*0
_output_shapes
:���������<<�a
IdentityIdentity	mul_3:z:0^NoOp*
T0*0
_output_shapes
:���������<<�w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������<<�: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:���������<<�
 
_user_specified_nameinputs
�
�
C__inference_conv2d_5_layer_call_and_return_conditional_losses_55887

inputs8
conv2d_readvariableop_resource:<<-
biasadd_readvariableop_resource:<
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:<<*
dtype0r
Conv2D/CastCastConv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*&
_output_shapes
:<<�
Conv2DConv2DinputsConv2D/Cast:y:0*
T0*1
_output_shapes
:�����������<*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:<*
dtype0h
BiasAdd/CastCastBiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:<q
BiasAddBiasAddConv2D:output:0BiasAdd/Cast:y:0*
T0*1
_output_shapes
:�����������<H
mul/xConst*
_output_shapes
: *
dtype0*
value
B j�ph
mulMulmul/x:output:0BiasAdd:output:0*
T0*1
_output_shapes
:�����������<I
Pow/yConst*
_output_shapes
: *
dtype0*
valueB	 j��h
PowPowBiasAdd:output:0Pow/y:output:0*
T0*1
_output_shapes
:�����������<J
mul_1/xConst*
_output_shapes
: *
dtype0*
value
B j�Sc
mul_1Mulmul_1/x:output:0Pow:z:0*
T0*1
_output_shapes
:�����������<e
addAddV2BiasAdd:output:0	mul_1:z:0*
T0*1
_output_shapes
:�����������<J
mul_2/xConst*
_output_shapes
: *
dtype0*
value
B j�tc
mul_2Mulmul_2/x:output:0add:z:0*
T0*1
_output_shapes
:�����������<S
TanhTanh	mul_2:z:0*
T0*1
_output_shapes
:�����������<J
add_1/xConst*
_output_shapes
: *
dtype0*
value
B j�xf
add_1AddV2add_1/x:output:0Tanh:y:0*
T0*1
_output_shapes
:�����������<\
mul_3Mulmul:z:0	add_1:z:0*
T0*1
_output_shapes
:�����������<b
IdentityIdentity	mul_3:z:0^NoOp*
T0*1
_output_shapes
:�����������<w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:�����������<: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:�����������<
 
_user_specified_nameinputs
�
�

%__inference_model_layer_call_fn_54818

inputs!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:#
	unknown_3:
	unknown_4:#
	unknown_5:
	unknown_6:#
	unknown_7:<
	unknown_8:<#
	unknown_9:<<

unknown_10:<$

unknown_11:<x

unknown_12:x$

unknown_13:xx

unknown_14:x%

unknown_15:x�

unknown_16:	�&

unknown_17:��

unknown_18:	�&

unknown_19:��

unknown_20:	�&

unknown_21:��

unknown_22:	�%

unknown_23:x�

unknown_24:x$

unknown_25:xx

unknown_26:x$

unknown_27:xx

unknown_28:x$

unknown_29:<x

unknown_30:<$

unknown_31:<<

unknown_32:<$

unknown_33:<<

unknown_34:<$

unknown_35:<

unknown_36:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36*2
Tin+
)2'*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������*H
_read_only_resource_inputs*
(&	
 !"#$%&*0
config_proto 

CPU

GPU2*0J 8� *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_54191y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:�����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*|
_input_shapesk
i:�����������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
�
D__inference_conv2d_13_layer_call_and_return_conditional_losses_53614

inputs8
conv2d_readvariableop_resource:xx-
biasadd_readvariableop_resource:x
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:xx*
dtype0r
Conv2D/CastCastConv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*&
_output_shapes
:xx�
Conv2DConv2DinputsConv2D/Cast:y:0*
T0*/
_output_shapes
:���������xxx*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:x*
dtype0h
BiasAdd/CastCastBiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:xo
BiasAddBiasAddConv2D:output:0BiasAdd/Cast:y:0*
T0*/
_output_shapes
:���������xxxH
mul/xConst*
_output_shapes
: *
dtype0*
value
B j�pf
mulMulmul/x:output:0BiasAdd:output:0*
T0*/
_output_shapes
:���������xxxI
Pow/yConst*
_output_shapes
: *
dtype0*
valueB	 j��f
PowPowBiasAdd:output:0Pow/y:output:0*
T0*/
_output_shapes
:���������xxxJ
mul_1/xConst*
_output_shapes
: *
dtype0*
value
B j�Sa
mul_1Mulmul_1/x:output:0Pow:z:0*
T0*/
_output_shapes
:���������xxxc
addAddV2BiasAdd:output:0	mul_1:z:0*
T0*/
_output_shapes
:���������xxxJ
mul_2/xConst*
_output_shapes
: *
dtype0*
value
B j�ta
mul_2Mulmul_2/x:output:0add:z:0*
T0*/
_output_shapes
:���������xxxQ
TanhTanh	mul_2:z:0*
T0*/
_output_shapes
:���������xxxJ
add_1/xConst*
_output_shapes
: *
dtype0*
value
B j�xd
add_1AddV2add_1/x:output:0Tanh:y:0*
T0*/
_output_shapes
:���������xxxZ
mul_3Mulmul:z:0	add_1:z:0*
T0*/
_output_shapes
:���������xxx`
IdentityIdentity	mul_3:z:0^NoOp*
T0*/
_output_shapes
:���������xxxw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������xxx: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������xxx
 
_user_specified_nameinputs
�
�
D__inference_conv2d_12_layer_call_and_return_conditional_losses_53583

inputs8
conv2d_readvariableop_resource:xx-
biasadd_readvariableop_resource:x
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:xx*
dtype0r
Conv2D/CastCastConv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*&
_output_shapes
:xx�
Conv2DConv2DinputsConv2D/Cast:y:0*
T0*/
_output_shapes
:���������xxx*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:x*
dtype0h
BiasAdd/CastCastBiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:xo
BiasAddBiasAddConv2D:output:0BiasAdd/Cast:y:0*
T0*/
_output_shapes
:���������xxxH
mul/xConst*
_output_shapes
: *
dtype0*
value
B j�pf
mulMulmul/x:output:0BiasAdd:output:0*
T0*/
_output_shapes
:���������xxxI
Pow/yConst*
_output_shapes
: *
dtype0*
valueB	 j��f
PowPowBiasAdd:output:0Pow/y:output:0*
T0*/
_output_shapes
:���������xxxJ
mul_1/xConst*
_output_shapes
: *
dtype0*
value
B j�Sa
mul_1Mulmul_1/x:output:0Pow:z:0*
T0*/
_output_shapes
:���������xxxc
addAddV2BiasAdd:output:0	mul_1:z:0*
T0*/
_output_shapes
:���������xxxJ
mul_2/xConst*
_output_shapes
: *
dtype0*
value
B j�ta
mul_2Mulmul_2/x:output:0add:z:0*
T0*/
_output_shapes
:���������xxxQ
TanhTanh	mul_2:z:0*
T0*/
_output_shapes
:���������xxxJ
add_1/xConst*
_output_shapes
: *
dtype0*
value
B j�xd
add_1AddV2add_1/x:output:0Tanh:y:0*
T0*/
_output_shapes
:���������xxxZ
mul_3Mulmul:z:0	add_1:z:0*
T0*/
_output_shapes
:���������xxx`
IdentityIdentity	mul_3:z:0^NoOp*
T0*/
_output_shapes
:���������xxxw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������xxx: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������xxx
 
_user_specified_nameinputs
�
�
(__inference_conv2d_7_layer_call_fn_55940

inputs!
unknown:xx
	unknown_0:x
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������xxx*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_conv2d_7_layer_call_and_return_conditional_losses_53414w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������xxx`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������xxx: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������xxx
 
_user_specified_nameinputs
�
h
L__inference_average_pooling2d_layer_call_and_return_conditional_losses_53020

inputs
identity�
AvgPoolAvgPoolinputs*
T0*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingSAME*
strides
{
IdentityIdentityAvgPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
�

#__inference_signature_wrapper_54656
input_1!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:#
	unknown_3:
	unknown_4:#
	unknown_5:
	unknown_6:#
	unknown_7:<
	unknown_8:<#
	unknown_9:<<

unknown_10:<$

unknown_11:<x

unknown_12:x$

unknown_13:xx

unknown_14:x%

unknown_15:x�

unknown_16:	�&

unknown_17:��

unknown_18:	�&

unknown_19:��

unknown_20:	�&

unknown_21:��

unknown_22:	�%

unknown_23:x�

unknown_24:x$

unknown_25:xx

unknown_26:x$

unknown_27:xx

unknown_28:x$

unknown_29:<x

unknown_30:<$

unknown_31:<<

unknown_32:<$

unknown_33:<<

unknown_34:<$

unknown_35:<

unknown_36:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36*2
Tin+
)2'*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������*H
_read_only_resource_inputs*
(&	
 !"#$%&*0
config_proto 

CPU

GPU2*0J 8� *)
f$R"
 __inference__wrapped_model_53011y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:�����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*|
_input_shapesk
i:�����������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:�����������
!
_user_specified_name	input_1
�
�
(__inference_conv2d_4_layer_call_fn_55828

inputs!
unknown:<
	unknown_0:<
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������<*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_conv2d_4_layer_call_and_return_conditional_losses_53320y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:�����������<`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:�����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
�
&__inference_conv2d_layer_call_fn_55679

inputs!
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_53186y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:�����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:�����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
j
@__inference_add_2_layer_call_and_return_conditional_losses_53720

inputs
inputs_1
identityZ
addAddV2inputsinputs_1*
T0*1
_output_shapes
:�����������Y
IdentityIdentityadd:z:0*
T0*1
_output_shapes
:�����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*M
_input_shapes<
::�����������:�����������:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs:YU
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
�
C__inference_conv2d_2_layer_call_and_return_conditional_losses_53248

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0r
Conv2D/CastCastConv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*&
_output_shapes
:�
Conv2DConv2DinputsConv2D/Cast:y:0*
T0*1
_output_shapes
:�����������*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0h
BiasAdd/CastCastBiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:q
BiasAddBiasAddConv2D:output:0BiasAdd/Cast:y:0*
T0*1
_output_shapes
:�����������H
mul/xConst*
_output_shapes
: *
dtype0*
value
B j�ph
mulMulmul/x:output:0BiasAdd:output:0*
T0*1
_output_shapes
:�����������I
Pow/yConst*
_output_shapes
: *
dtype0*
valueB	 j��h
PowPowBiasAdd:output:0Pow/y:output:0*
T0*1
_output_shapes
:�����������J
mul_1/xConst*
_output_shapes
: *
dtype0*
value
B j�Sc
mul_1Mulmul_1/x:output:0Pow:z:0*
T0*1
_output_shapes
:�����������e
addAddV2BiasAdd:output:0	mul_1:z:0*
T0*1
_output_shapes
:�����������J
mul_2/xConst*
_output_shapes
: *
dtype0*
value
B j�tc
mul_2Mulmul_2/x:output:0add:z:0*
T0*1
_output_shapes
:�����������S
TanhTanh	mul_2:z:0*
T0*1
_output_shapes
:�����������J
add_1/xConst*
_output_shapes
: *
dtype0*
value
B j�xf
add_1AddV2add_1/x:output:0Tanh:y:0*
T0*1
_output_shapes
:�����������\
mul_3Mulmul:z:0	add_1:z:0*
T0*1
_output_shapes
:�����������b
IdentityIdentity	mul_3:z:0^NoOp*
T0*1
_output_shapes
:�����������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:�����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
Q
%__inference_add_2_layer_call_fn_56412
inputs_0
inputs_1
identity�
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *I
fDRB
@__inference_add_2_layer_call_and_return_conditional_losses_53720j
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:�����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*M
_input_shapes<
::�����������:�����������:[ W
1
_output_shapes
:�����������
"
_user_specified_name
inputs/0:[W
1
_output_shapes
:�����������
"
_user_specified_name
inputs/1
�w
�
@__inference_model_layer_call_and_return_conditional_losses_54567
input_1&
conv2d_54463:
conv2d_54465:(
conv2d_1_54468:
conv2d_1_54470:(
conv2d_2_54473:
conv2d_2_54475:(
conv2d_3_54478:
conv2d_3_54480:(
conv2d_4_54485:<
conv2d_4_54487:<(
conv2d_5_54490:<<
conv2d_5_54492:<(
conv2d_6_54496:<x
conv2d_6_54498:x(
conv2d_7_54501:xx
conv2d_7_54503:x)
conv2d_8_54507:x�
conv2d_8_54509:	�*
conv2d_9_54512:��
conv2d_9_54514:	�+
conv2d_10_54517:��
conv2d_10_54519:	�+
conv2d_11_54522:��
conv2d_11_54524:	�1
conv2d_transpose_54527:x�$
conv2d_transpose_54529:x)
conv2d_12_54533:xx
conv2d_12_54535:x)
conv2d_13_54538:xx
conv2d_13_54540:x2
conv2d_transpose_1_54543:<x&
conv2d_transpose_1_54545:<)
conv2d_14_54549:<<
conv2d_14_54551:<)
conv2d_15_54554:<<
conv2d_15_54556:<)
conv2d_16_54559:<
conv2d_16_54561:
identity��conv2d/StatefulPartitionedCall� conv2d_1/StatefulPartitionedCall�!conv2d_10/StatefulPartitionedCall�!conv2d_11/StatefulPartitionedCall�!conv2d_12/StatefulPartitionedCall�!conv2d_13/StatefulPartitionedCall�!conv2d_14/StatefulPartitionedCall�!conv2d_15/StatefulPartitionedCall�!conv2d_16/StatefulPartitionedCall� conv2d_2/StatefulPartitionedCall� conv2d_3/StatefulPartitionedCall� conv2d_4/StatefulPartitionedCall� conv2d_5/StatefulPartitionedCall� conv2d_6/StatefulPartitionedCall� conv2d_7/StatefulPartitionedCall� conv2d_8/StatefulPartitionedCall� conv2d_9/StatefulPartitionedCall�(conv2d_transpose/StatefulPartitionedCall�*conv2d_transpose_1/StatefulPartitionedCallg
conv2d/CastCastinput_1*

DstT0*

SrcT0*1
_output_shapes
:������������
conv2d/StatefulPartitionedCallStatefulPartitionedCallconv2d/Cast:y:0conv2d_54463conv2d_54465*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_53186�
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0conv2d_1_54468conv2d_1_54470*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_53217�
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0conv2d_2_54473conv2d_2_54475*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_conv2d_2_layer_call_and_return_conditional_losses_53248�
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0conv2d_3_54478conv2d_3_54480*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_conv2d_3_layer_call_and_return_conditional_losses_53279l
concatenate/CastCastinput_1*

DstT0*

SrcT0*1
_output_shapes
:������������
concatenate/PartitionedCallPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0concatenate/Cast:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_concatenate_layer_call_and_return_conditional_losses_53293�
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0conv2d_4_54485conv2d_4_54487*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������<*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_conv2d_4_layer_call_and_return_conditional_losses_53320�
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0conv2d_5_54490conv2d_5_54492*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������<*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_conv2d_5_layer_call_and_return_conditional_losses_53351�
!average_pooling2d/PartitionedCallPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������xx<* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_average_pooling2d_layer_call_and_return_conditional_losses_53020�
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCall*average_pooling2d/PartitionedCall:output:0conv2d_6_54496conv2d_6_54498*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������xxx*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_conv2d_6_layer_call_and_return_conditional_losses_53383�
 conv2d_7/StatefulPartitionedCallStatefulPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0conv2d_7_54501conv2d_7_54503*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������xxx*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_conv2d_7_layer_call_and_return_conditional_losses_53414�
#average_pooling2d_1/PartitionedCallPartitionedCall)conv2d_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������<<x* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *W
fRRP
N__inference_average_pooling2d_1_layer_call_and_return_conditional_losses_53032�
 conv2d_8/StatefulPartitionedCallStatefulPartitionedCall,average_pooling2d_1/PartitionedCall:output:0conv2d_8_54507conv2d_8_54509*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������<<�*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_conv2d_8_layer_call_and_return_conditional_losses_53446�
 conv2d_9/StatefulPartitionedCallStatefulPartitionedCall)conv2d_8/StatefulPartitionedCall:output:0conv2d_9_54512conv2d_9_54514*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������<<�*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_conv2d_9_layer_call_and_return_conditional_losses_53477�
!conv2d_10/StatefulPartitionedCallStatefulPartitionedCall)conv2d_9/StatefulPartitionedCall:output:0conv2d_10_54517conv2d_10_54519*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������<<�*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv2d_10_layer_call_and_return_conditional_losses_53508�
!conv2d_11/StatefulPartitionedCallStatefulPartitionedCall*conv2d_10/StatefulPartitionedCall:output:0conv2d_11_54522conv2d_11_54524*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������<<�*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv2d_11_layer_call_and_return_conditional_losses_53539�
(conv2d_transpose/StatefulPartitionedCallStatefulPartitionedCall*conv2d_11/StatefulPartitionedCall:output:0conv2d_transpose_54527conv2d_transpose_54529*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������xxx*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *T
fORM
K__inference_conv2d_transpose_layer_call_and_return_conditional_losses_53087�
add/PartitionedCallPartitionedCall1conv2d_transpose/StatefulPartitionedCall:output:0)conv2d_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������xxx* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *G
fBR@
>__inference_add_layer_call_and_return_conditional_losses_53556�
!conv2d_12/StatefulPartitionedCallStatefulPartitionedCalladd/PartitionedCall:output:0conv2d_12_54533conv2d_12_54535*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������xxx*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv2d_12_layer_call_and_return_conditional_losses_53583�
!conv2d_13/StatefulPartitionedCallStatefulPartitionedCall*conv2d_12/StatefulPartitionedCall:output:0conv2d_13_54538conv2d_13_54540*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������xxx*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv2d_13_layer_call_and_return_conditional_losses_53614�
*conv2d_transpose_1/StatefulPartitionedCallStatefulPartitionedCall*conv2d_13/StatefulPartitionedCall:output:0conv2d_transpose_1_54543conv2d_transpose_1_54545*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������<*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *V
fQRO
M__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_53146�
add_1/PartitionedCallPartitionedCall3conv2d_transpose_1/StatefulPartitionedCall:output:0)conv2d_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������<* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *I
fDRB
@__inference_add_1_layer_call_and_return_conditional_losses_53631�
!conv2d_14/StatefulPartitionedCallStatefulPartitionedCalladd_1/PartitionedCall:output:0conv2d_14_54549conv2d_14_54551*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������<*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv2d_14_layer_call_and_return_conditional_losses_53658�
!conv2d_15/StatefulPartitionedCallStatefulPartitionedCall*conv2d_14/StatefulPartitionedCall:output:0conv2d_15_54554conv2d_15_54556*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������<*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv2d_15_layer_call_and_return_conditional_losses_53689�
!conv2d_16/StatefulPartitionedCallStatefulPartitionedCall*conv2d_15/StatefulPartitionedCall:output:0conv2d_16_54559conv2d_16_54561*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv2d_16_layer_call_and_return_conditional_losses_53707�

add_2/CastCast*conv2d_16/StatefulPartitionedCall:output:0*

DstT0*

SrcT0*1
_output_shapes
:������������
add_2/PartitionedCallPartitionedCalladd_2/Cast:y:0input_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *I
fDRB
@__inference_add_2_layer_call_and_return_conditional_losses_53720w
IdentityIdentityadd_2/PartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:������������
NoOpNoOp^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall"^conv2d_10/StatefulPartitionedCall"^conv2d_11/StatefulPartitionedCall"^conv2d_12/StatefulPartitionedCall"^conv2d_13/StatefulPartitionedCall"^conv2d_14/StatefulPartitionedCall"^conv2d_15/StatefulPartitionedCall"^conv2d_16/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall!^conv2d_5/StatefulPartitionedCall!^conv2d_6/StatefulPartitionedCall!^conv2d_7/StatefulPartitionedCall!^conv2d_8/StatefulPartitionedCall!^conv2d_9/StatefulPartitionedCall)^conv2d_transpose/StatefulPartitionedCall+^conv2d_transpose_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*|
_input_shapesk
i:�����������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2F
!conv2d_10/StatefulPartitionedCall!conv2d_10/StatefulPartitionedCall2F
!conv2d_11/StatefulPartitionedCall!conv2d_11/StatefulPartitionedCall2F
!conv2d_12/StatefulPartitionedCall!conv2d_12/StatefulPartitionedCall2F
!conv2d_13/StatefulPartitionedCall!conv2d_13/StatefulPartitionedCall2F
!conv2d_14/StatefulPartitionedCall!conv2d_14/StatefulPartitionedCall2F
!conv2d_15/StatefulPartitionedCall!conv2d_15/StatefulPartitionedCall2F
!conv2d_16/StatefulPartitionedCall!conv2d_16/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall2D
 conv2d_7/StatefulPartitionedCall conv2d_7/StatefulPartitionedCall2D
 conv2d_8/StatefulPartitionedCall conv2d_8/StatefulPartitionedCall2D
 conv2d_9/StatefulPartitionedCall conv2d_9/StatefulPartitionedCall2T
(conv2d_transpose/StatefulPartitionedCall(conv2d_transpose/StatefulPartitionedCall2X
*conv2d_transpose_1/StatefulPartitionedCall*conv2d_transpose_1/StatefulPartitionedCall:Z V
1
_output_shapes
:�����������
!
_user_specified_name	input_1
�
�
)__inference_conv2d_14_layer_call_fn_56326

inputs!
unknown:<<
	unknown_0:<
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������<*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv2d_14_layer_call_and_return_conditional_losses_53658y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:�����������<`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:�����������<: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:�����������<
 
_user_specified_nameinputs
�
�
)__inference_conv2d_16_layer_call_fn_56394

inputs!
unknown:<
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv2d_16_layer_call_and_return_conditional_losses_53707y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:�����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:�����������<: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:�����������<
 
_user_specified_nameinputs
�
�
(__inference_conv2d_9_layer_call_fn_56018

inputs#
unknown:��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������<<�*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_conv2d_9_layer_call_and_return_conditional_losses_53477x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:���������<<�`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������<<�: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:���������<<�
 
_user_specified_nameinputs
�
j
>__inference_add_layer_call_and_return_conditional_losses_56180
inputs_0
inputs_1
identityZ
addAddV2inputs_0inputs_1*
T0*/
_output_shapes
:���������xxxW
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:���������xxx"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:���������xxx:���������xxx:Y U
/
_output_shapes
:���������xxx
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:���������xxx
"
_user_specified_name
inputs/1
۩
�s
!__inference__traced_restore_57441
file_prefix8
assignvariableop_conv2d_kernel:,
assignvariableop_1_conv2d_bias:<
"assignvariableop_2_conv2d_1_kernel:.
 assignvariableop_3_conv2d_1_bias:<
"assignvariableop_4_conv2d_2_kernel:.
 assignvariableop_5_conv2d_2_bias:<
"assignvariableop_6_conv2d_3_kernel:.
 assignvariableop_7_conv2d_3_bias:<
"assignvariableop_8_conv2d_4_kernel:<.
 assignvariableop_9_conv2d_4_bias:<=
#assignvariableop_10_conv2d_5_kernel:<</
!assignvariableop_11_conv2d_5_bias:<=
#assignvariableop_12_conv2d_6_kernel:<x/
!assignvariableop_13_conv2d_6_bias:x=
#assignvariableop_14_conv2d_7_kernel:xx/
!assignvariableop_15_conv2d_7_bias:x>
#assignvariableop_16_conv2d_8_kernel:x�0
!assignvariableop_17_conv2d_8_bias:	�?
#assignvariableop_18_conv2d_9_kernel:��0
!assignvariableop_19_conv2d_9_bias:	�@
$assignvariableop_20_conv2d_10_kernel:��1
"assignvariableop_21_conv2d_10_bias:	�@
$assignvariableop_22_conv2d_11_kernel:��1
"assignvariableop_23_conv2d_11_bias:	�F
+assignvariableop_24_conv2d_transpose_kernel:x�7
)assignvariableop_25_conv2d_transpose_bias:x>
$assignvariableop_26_conv2d_12_kernel:xx0
"assignvariableop_27_conv2d_12_bias:x>
$assignvariableop_28_conv2d_13_kernel:xx0
"assignvariableop_29_conv2d_13_bias:xG
-assignvariableop_30_conv2d_transpose_1_kernel:<x9
+assignvariableop_31_conv2d_transpose_1_bias:<>
$assignvariableop_32_conv2d_14_kernel:<<0
"assignvariableop_33_conv2d_14_bias:<>
$assignvariableop_34_conv2d_15_kernel:<<0
"assignvariableop_35_conv2d_15_bias:<>
$assignvariableop_36_conv2d_16_kernel:<0
"assignvariableop_37_conv2d_16_bias:.
$assignvariableop_38_cond_1_adam_iter:	 0
&assignvariableop_39_cond_1_adam_beta_1: 0
&assignvariableop_40_cond_1_adam_beta_2: /
%assignvariableop_41_cond_1_adam_decay: 7
-assignvariableop_42_cond_1_adam_learning_rate: 0
&assignvariableop_43_current_loss_scale: (
assignvariableop_44_good_steps:	 %
assignvariableop_45_total_2: %
assignvariableop_46_count_2: %
assignvariableop_47_total_1: %
assignvariableop_48_count_1: #
assignvariableop_49_total: #
assignvariableop_50_count: I
/assignvariableop_51_cond_1_adam_conv2d_kernel_m:;
-assignvariableop_52_cond_1_adam_conv2d_bias_m:K
1assignvariableop_53_cond_1_adam_conv2d_1_kernel_m:=
/assignvariableop_54_cond_1_adam_conv2d_1_bias_m:K
1assignvariableop_55_cond_1_adam_conv2d_2_kernel_m:=
/assignvariableop_56_cond_1_adam_conv2d_2_bias_m:K
1assignvariableop_57_cond_1_adam_conv2d_3_kernel_m:=
/assignvariableop_58_cond_1_adam_conv2d_3_bias_m:K
1assignvariableop_59_cond_1_adam_conv2d_4_kernel_m:<=
/assignvariableop_60_cond_1_adam_conv2d_4_bias_m:<K
1assignvariableop_61_cond_1_adam_conv2d_5_kernel_m:<<=
/assignvariableop_62_cond_1_adam_conv2d_5_bias_m:<K
1assignvariableop_63_cond_1_adam_conv2d_6_kernel_m:<x=
/assignvariableop_64_cond_1_adam_conv2d_6_bias_m:xK
1assignvariableop_65_cond_1_adam_conv2d_7_kernel_m:xx=
/assignvariableop_66_cond_1_adam_conv2d_7_bias_m:xL
1assignvariableop_67_cond_1_adam_conv2d_8_kernel_m:x�>
/assignvariableop_68_cond_1_adam_conv2d_8_bias_m:	�M
1assignvariableop_69_cond_1_adam_conv2d_9_kernel_m:��>
/assignvariableop_70_cond_1_adam_conv2d_9_bias_m:	�N
2assignvariableop_71_cond_1_adam_conv2d_10_kernel_m:��?
0assignvariableop_72_cond_1_adam_conv2d_10_bias_m:	�N
2assignvariableop_73_cond_1_adam_conv2d_11_kernel_m:��?
0assignvariableop_74_cond_1_adam_conv2d_11_bias_m:	�T
9assignvariableop_75_cond_1_adam_conv2d_transpose_kernel_m:x�E
7assignvariableop_76_cond_1_adam_conv2d_transpose_bias_m:xL
2assignvariableop_77_cond_1_adam_conv2d_12_kernel_m:xx>
0assignvariableop_78_cond_1_adam_conv2d_12_bias_m:xL
2assignvariableop_79_cond_1_adam_conv2d_13_kernel_m:xx>
0assignvariableop_80_cond_1_adam_conv2d_13_bias_m:xU
;assignvariableop_81_cond_1_adam_conv2d_transpose_1_kernel_m:<xG
9assignvariableop_82_cond_1_adam_conv2d_transpose_1_bias_m:<L
2assignvariableop_83_cond_1_adam_conv2d_14_kernel_m:<<>
0assignvariableop_84_cond_1_adam_conv2d_14_bias_m:<L
2assignvariableop_85_cond_1_adam_conv2d_15_kernel_m:<<>
0assignvariableop_86_cond_1_adam_conv2d_15_bias_m:<L
2assignvariableop_87_cond_1_adam_conv2d_16_kernel_m:<>
0assignvariableop_88_cond_1_adam_conv2d_16_bias_m:I
/assignvariableop_89_cond_1_adam_conv2d_kernel_v:;
-assignvariableop_90_cond_1_adam_conv2d_bias_v:K
1assignvariableop_91_cond_1_adam_conv2d_1_kernel_v:=
/assignvariableop_92_cond_1_adam_conv2d_1_bias_v:K
1assignvariableop_93_cond_1_adam_conv2d_2_kernel_v:=
/assignvariableop_94_cond_1_adam_conv2d_2_bias_v:K
1assignvariableop_95_cond_1_adam_conv2d_3_kernel_v:=
/assignvariableop_96_cond_1_adam_conv2d_3_bias_v:K
1assignvariableop_97_cond_1_adam_conv2d_4_kernel_v:<=
/assignvariableop_98_cond_1_adam_conv2d_4_bias_v:<K
1assignvariableop_99_cond_1_adam_conv2d_5_kernel_v:<<>
0assignvariableop_100_cond_1_adam_conv2d_5_bias_v:<L
2assignvariableop_101_cond_1_adam_conv2d_6_kernel_v:<x>
0assignvariableop_102_cond_1_adam_conv2d_6_bias_v:xL
2assignvariableop_103_cond_1_adam_conv2d_7_kernel_v:xx>
0assignvariableop_104_cond_1_adam_conv2d_7_bias_v:xM
2assignvariableop_105_cond_1_adam_conv2d_8_kernel_v:x�?
0assignvariableop_106_cond_1_adam_conv2d_8_bias_v:	�N
2assignvariableop_107_cond_1_adam_conv2d_9_kernel_v:��?
0assignvariableop_108_cond_1_adam_conv2d_9_bias_v:	�O
3assignvariableop_109_cond_1_adam_conv2d_10_kernel_v:��@
1assignvariableop_110_cond_1_adam_conv2d_10_bias_v:	�O
3assignvariableop_111_cond_1_adam_conv2d_11_kernel_v:��@
1assignvariableop_112_cond_1_adam_conv2d_11_bias_v:	�U
:assignvariableop_113_cond_1_adam_conv2d_transpose_kernel_v:x�F
8assignvariableop_114_cond_1_adam_conv2d_transpose_bias_v:xM
3assignvariableop_115_cond_1_adam_conv2d_12_kernel_v:xx?
1assignvariableop_116_cond_1_adam_conv2d_12_bias_v:xM
3assignvariableop_117_cond_1_adam_conv2d_13_kernel_v:xx?
1assignvariableop_118_cond_1_adam_conv2d_13_bias_v:xV
<assignvariableop_119_cond_1_adam_conv2d_transpose_1_kernel_v:<xH
:assignvariableop_120_cond_1_adam_conv2d_transpose_1_bias_v:<M
3assignvariableop_121_cond_1_adam_conv2d_14_kernel_v:<<?
1assignvariableop_122_cond_1_adam_conv2d_14_bias_v:<M
3assignvariableop_123_cond_1_adam_conv2d_15_kernel_v:<<?
1assignvariableop_124_cond_1_adam_conv2d_15_bias_v:<M
3assignvariableop_125_cond_1_adam_conv2d_16_kernel_v:<?
1assignvariableop_126_cond_1_adam_conv2d_16_bias_v:M
3assignvariableop_127_cond_1_adam_conv2d_kernel_vhat:?
1assignvariableop_128_cond_1_adam_conv2d_bias_vhat:O
5assignvariableop_129_cond_1_adam_conv2d_1_kernel_vhat:A
3assignvariableop_130_cond_1_adam_conv2d_1_bias_vhat:O
5assignvariableop_131_cond_1_adam_conv2d_2_kernel_vhat:A
3assignvariableop_132_cond_1_adam_conv2d_2_bias_vhat:O
5assignvariableop_133_cond_1_adam_conv2d_3_kernel_vhat:A
3assignvariableop_134_cond_1_adam_conv2d_3_bias_vhat:O
5assignvariableop_135_cond_1_adam_conv2d_4_kernel_vhat:<A
3assignvariableop_136_cond_1_adam_conv2d_4_bias_vhat:<O
5assignvariableop_137_cond_1_adam_conv2d_5_kernel_vhat:<<A
3assignvariableop_138_cond_1_adam_conv2d_5_bias_vhat:<O
5assignvariableop_139_cond_1_adam_conv2d_6_kernel_vhat:<xA
3assignvariableop_140_cond_1_adam_conv2d_6_bias_vhat:xO
5assignvariableop_141_cond_1_adam_conv2d_7_kernel_vhat:xxA
3assignvariableop_142_cond_1_adam_conv2d_7_bias_vhat:xP
5assignvariableop_143_cond_1_adam_conv2d_8_kernel_vhat:x�B
3assignvariableop_144_cond_1_adam_conv2d_8_bias_vhat:	�Q
5assignvariableop_145_cond_1_adam_conv2d_9_kernel_vhat:��B
3assignvariableop_146_cond_1_adam_conv2d_9_bias_vhat:	�R
6assignvariableop_147_cond_1_adam_conv2d_10_kernel_vhat:��C
4assignvariableop_148_cond_1_adam_conv2d_10_bias_vhat:	�R
6assignvariableop_149_cond_1_adam_conv2d_11_kernel_vhat:��C
4assignvariableop_150_cond_1_adam_conv2d_11_bias_vhat:	�X
=assignvariableop_151_cond_1_adam_conv2d_transpose_kernel_vhat:x�I
;assignvariableop_152_cond_1_adam_conv2d_transpose_bias_vhat:xP
6assignvariableop_153_cond_1_adam_conv2d_12_kernel_vhat:xxB
4assignvariableop_154_cond_1_adam_conv2d_12_bias_vhat:xP
6assignvariableop_155_cond_1_adam_conv2d_13_kernel_vhat:xxB
4assignvariableop_156_cond_1_adam_conv2d_13_bias_vhat:xY
?assignvariableop_157_cond_1_adam_conv2d_transpose_1_kernel_vhat:<xK
=assignvariableop_158_cond_1_adam_conv2d_transpose_1_bias_vhat:<P
6assignvariableop_159_cond_1_adam_conv2d_14_kernel_vhat:<<B
4assignvariableop_160_cond_1_adam_conv2d_14_bias_vhat:<P
6assignvariableop_161_cond_1_adam_conv2d_15_kernel_vhat:<<B
4assignvariableop_162_cond_1_adam_conv2d_15_bias_vhat:<P
6assignvariableop_163_cond_1_adam_conv2d_16_kernel_vhat:<B
4assignvariableop_164_cond_1_adam_conv2d_16_bias_vhat:
identity_166��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_100�AssignVariableOp_101�AssignVariableOp_102�AssignVariableOp_103�AssignVariableOp_104�AssignVariableOp_105�AssignVariableOp_106�AssignVariableOp_107�AssignVariableOp_108�AssignVariableOp_109�AssignVariableOp_11�AssignVariableOp_110�AssignVariableOp_111�AssignVariableOp_112�AssignVariableOp_113�AssignVariableOp_114�AssignVariableOp_115�AssignVariableOp_116�AssignVariableOp_117�AssignVariableOp_118�AssignVariableOp_119�AssignVariableOp_12�AssignVariableOp_120�AssignVariableOp_121�AssignVariableOp_122�AssignVariableOp_123�AssignVariableOp_124�AssignVariableOp_125�AssignVariableOp_126�AssignVariableOp_127�AssignVariableOp_128�AssignVariableOp_129�AssignVariableOp_13�AssignVariableOp_130�AssignVariableOp_131�AssignVariableOp_132�AssignVariableOp_133�AssignVariableOp_134�AssignVariableOp_135�AssignVariableOp_136�AssignVariableOp_137�AssignVariableOp_138�AssignVariableOp_139�AssignVariableOp_14�AssignVariableOp_140�AssignVariableOp_141�AssignVariableOp_142�AssignVariableOp_143�AssignVariableOp_144�AssignVariableOp_145�AssignVariableOp_146�AssignVariableOp_147�AssignVariableOp_148�AssignVariableOp_149�AssignVariableOp_15�AssignVariableOp_150�AssignVariableOp_151�AssignVariableOp_152�AssignVariableOp_153�AssignVariableOp_154�AssignVariableOp_155�AssignVariableOp_156�AssignVariableOp_157�AssignVariableOp_158�AssignVariableOp_159�AssignVariableOp_16�AssignVariableOp_160�AssignVariableOp_161�AssignVariableOp_162�AssignVariableOp_163�AssignVariableOp_164�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_32�AssignVariableOp_33�AssignVariableOp_34�AssignVariableOp_35�AssignVariableOp_36�AssignVariableOp_37�AssignVariableOp_38�AssignVariableOp_39�AssignVariableOp_4�AssignVariableOp_40�AssignVariableOp_41�AssignVariableOp_42�AssignVariableOp_43�AssignVariableOp_44�AssignVariableOp_45�AssignVariableOp_46�AssignVariableOp_47�AssignVariableOp_48�AssignVariableOp_49�AssignVariableOp_5�AssignVariableOp_50�AssignVariableOp_51�AssignVariableOp_52�AssignVariableOp_53�AssignVariableOp_54�AssignVariableOp_55�AssignVariableOp_56�AssignVariableOp_57�AssignVariableOp_58�AssignVariableOp_59�AssignVariableOp_6�AssignVariableOp_60�AssignVariableOp_61�AssignVariableOp_62�AssignVariableOp_63�AssignVariableOp_64�AssignVariableOp_65�AssignVariableOp_66�AssignVariableOp_67�AssignVariableOp_68�AssignVariableOp_69�AssignVariableOp_7�AssignVariableOp_70�AssignVariableOp_71�AssignVariableOp_72�AssignVariableOp_73�AssignVariableOp_74�AssignVariableOp_75�AssignVariableOp_76�AssignVariableOp_77�AssignVariableOp_78�AssignVariableOp_79�AssignVariableOp_8�AssignVariableOp_80�AssignVariableOp_81�AssignVariableOp_82�AssignVariableOp_83�AssignVariableOp_84�AssignVariableOp_85�AssignVariableOp_86�AssignVariableOp_87�AssignVariableOp_88�AssignVariableOp_89�AssignVariableOp_9�AssignVariableOp_90�AssignVariableOp_91�AssignVariableOp_92�AssignVariableOp_93�AssignVariableOp_94�AssignVariableOp_95�AssignVariableOp_96�AssignVariableOp_97�AssignVariableOp_98�AssignVariableOp_99�b
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes	
:�*
dtype0*�a
value�aB�a�B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-17/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-17/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-18/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-18/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEBBoptimizer/loss_scale/current_loss_scale/.ATTRIBUTES/VARIABLE_VALUEB:optimizer/loss_scale/good_steps/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-16/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-16/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-18/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-18/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-16/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-16/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-18/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-18/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-16/kernel/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-16/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-18/kernel/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-18/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes	
:�*
dtype0*�
value�B��B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*�
dtypes�
�2�		[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOpassignvariableop_conv2d_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOpassignvariableop_1_conv2d_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp"assignvariableop_2_conv2d_1_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOp assignvariableop_3_conv2d_1_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp"assignvariableop_4_conv2d_2_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOp assignvariableop_5_conv2d_2_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOp"assignvariableop_6_conv2d_3_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOp assignvariableop_7_conv2d_3_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOp"assignvariableop_8_conv2d_4_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOp assignvariableop_9_conv2d_4_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOp#assignvariableop_10_conv2d_5_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOp!assignvariableop_11_conv2d_5_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOp#assignvariableop_12_conv2d_6_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOp!assignvariableop_13_conv2d_6_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOp#assignvariableop_14_conv2d_7_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOp!assignvariableop_15_conv2d_7_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOp#assignvariableop_16_conv2d_8_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOp!assignvariableop_17_conv2d_8_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOp#assignvariableop_18_conv2d_9_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOp!assignvariableop_19_conv2d_9_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOp$assignvariableop_20_conv2d_10_kernelIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOp"assignvariableop_21_conv2d_10_biasIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_22AssignVariableOp$assignvariableop_22_conv2d_11_kernelIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_23AssignVariableOp"assignvariableop_23_conv2d_11_biasIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_24AssignVariableOp+assignvariableop_24_conv2d_transpose_kernelIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_25AssignVariableOp)assignvariableop_25_conv2d_transpose_biasIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_26AssignVariableOp$assignvariableop_26_conv2d_12_kernelIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_27AssignVariableOp"assignvariableop_27_conv2d_12_biasIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_28AssignVariableOp$assignvariableop_28_conv2d_13_kernelIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_29AssignVariableOp"assignvariableop_29_conv2d_13_biasIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_30AssignVariableOp-assignvariableop_30_conv2d_transpose_1_kernelIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_31AssignVariableOp+assignvariableop_31_conv2d_transpose_1_biasIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_32AssignVariableOp$assignvariableop_32_conv2d_14_kernelIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_33AssignVariableOp"assignvariableop_33_conv2d_14_biasIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_34AssignVariableOp$assignvariableop_34_conv2d_15_kernelIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_35AssignVariableOp"assignvariableop_35_conv2d_15_biasIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_36AssignVariableOp$assignvariableop_36_conv2d_16_kernelIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_37AssignVariableOp"assignvariableop_37_conv2d_16_biasIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_38AssignVariableOp$assignvariableop_38_cond_1_adam_iterIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_39AssignVariableOp&assignvariableop_39_cond_1_adam_beta_1Identity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_40AssignVariableOp&assignvariableop_40_cond_1_adam_beta_2Identity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_41AssignVariableOp%assignvariableop_41_cond_1_adam_decayIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_42AssignVariableOp-assignvariableop_42_cond_1_adam_learning_rateIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_43AssignVariableOp&assignvariableop_43_current_loss_scaleIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_44AssignVariableOpassignvariableop_44_good_stepsIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_45AssignVariableOpassignvariableop_45_total_2Identity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_46AssignVariableOpassignvariableop_46_count_2Identity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_47AssignVariableOpassignvariableop_47_total_1Identity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_48AssignVariableOpassignvariableop_48_count_1Identity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_49AssignVariableOpassignvariableop_49_totalIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_50AssignVariableOpassignvariableop_50_countIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_51AssignVariableOp/assignvariableop_51_cond_1_adam_conv2d_kernel_mIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_52AssignVariableOp-assignvariableop_52_cond_1_adam_conv2d_bias_mIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_53AssignVariableOp1assignvariableop_53_cond_1_adam_conv2d_1_kernel_mIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_54AssignVariableOp/assignvariableop_54_cond_1_adam_conv2d_1_bias_mIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_55AssignVariableOp1assignvariableop_55_cond_1_adam_conv2d_2_kernel_mIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_56AssignVariableOp/assignvariableop_56_cond_1_adam_conv2d_2_bias_mIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_57AssignVariableOp1assignvariableop_57_cond_1_adam_conv2d_3_kernel_mIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_58AssignVariableOp/assignvariableop_58_cond_1_adam_conv2d_3_bias_mIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_59AssignVariableOp1assignvariableop_59_cond_1_adam_conv2d_4_kernel_mIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_60AssignVariableOp/assignvariableop_60_cond_1_adam_conv2d_4_bias_mIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_61AssignVariableOp1assignvariableop_61_cond_1_adam_conv2d_5_kernel_mIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_62AssignVariableOp/assignvariableop_62_cond_1_adam_conv2d_5_bias_mIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_63AssignVariableOp1assignvariableop_63_cond_1_adam_conv2d_6_kernel_mIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_64AssignVariableOp/assignvariableop_64_cond_1_adam_conv2d_6_bias_mIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_65AssignVariableOp1assignvariableop_65_cond_1_adam_conv2d_7_kernel_mIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_66AssignVariableOp/assignvariableop_66_cond_1_adam_conv2d_7_bias_mIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_67AssignVariableOp1assignvariableop_67_cond_1_adam_conv2d_8_kernel_mIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_68AssignVariableOp/assignvariableop_68_cond_1_adam_conv2d_8_bias_mIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_69AssignVariableOp1assignvariableop_69_cond_1_adam_conv2d_9_kernel_mIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_70AssignVariableOp/assignvariableop_70_cond_1_adam_conv2d_9_bias_mIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_71AssignVariableOp2assignvariableop_71_cond_1_adam_conv2d_10_kernel_mIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_72AssignVariableOp0assignvariableop_72_cond_1_adam_conv2d_10_bias_mIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_73AssignVariableOp2assignvariableop_73_cond_1_adam_conv2d_11_kernel_mIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_74AssignVariableOp0assignvariableop_74_cond_1_adam_conv2d_11_bias_mIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_75AssignVariableOp9assignvariableop_75_cond_1_adam_conv2d_transpose_kernel_mIdentity_75:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_76AssignVariableOp7assignvariableop_76_cond_1_adam_conv2d_transpose_bias_mIdentity_76:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_77AssignVariableOp2assignvariableop_77_cond_1_adam_conv2d_12_kernel_mIdentity_77:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_78AssignVariableOp0assignvariableop_78_cond_1_adam_conv2d_12_bias_mIdentity_78:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_79IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_79AssignVariableOp2assignvariableop_79_cond_1_adam_conv2d_13_kernel_mIdentity_79:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_80IdentityRestoreV2:tensors:80"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_80AssignVariableOp0assignvariableop_80_cond_1_adam_conv2d_13_bias_mIdentity_80:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_81IdentityRestoreV2:tensors:81"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_81AssignVariableOp;assignvariableop_81_cond_1_adam_conv2d_transpose_1_kernel_mIdentity_81:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_82IdentityRestoreV2:tensors:82"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_82AssignVariableOp9assignvariableop_82_cond_1_adam_conv2d_transpose_1_bias_mIdentity_82:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_83IdentityRestoreV2:tensors:83"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_83AssignVariableOp2assignvariableop_83_cond_1_adam_conv2d_14_kernel_mIdentity_83:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_84IdentityRestoreV2:tensors:84"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_84AssignVariableOp0assignvariableop_84_cond_1_adam_conv2d_14_bias_mIdentity_84:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_85IdentityRestoreV2:tensors:85"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_85AssignVariableOp2assignvariableop_85_cond_1_adam_conv2d_15_kernel_mIdentity_85:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_86IdentityRestoreV2:tensors:86"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_86AssignVariableOp0assignvariableop_86_cond_1_adam_conv2d_15_bias_mIdentity_86:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_87IdentityRestoreV2:tensors:87"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_87AssignVariableOp2assignvariableop_87_cond_1_adam_conv2d_16_kernel_mIdentity_87:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_88IdentityRestoreV2:tensors:88"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_88AssignVariableOp0assignvariableop_88_cond_1_adam_conv2d_16_bias_mIdentity_88:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_89IdentityRestoreV2:tensors:89"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_89AssignVariableOp/assignvariableop_89_cond_1_adam_conv2d_kernel_vIdentity_89:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_90IdentityRestoreV2:tensors:90"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_90AssignVariableOp-assignvariableop_90_cond_1_adam_conv2d_bias_vIdentity_90:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_91IdentityRestoreV2:tensors:91"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_91AssignVariableOp1assignvariableop_91_cond_1_adam_conv2d_1_kernel_vIdentity_91:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_92IdentityRestoreV2:tensors:92"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_92AssignVariableOp/assignvariableop_92_cond_1_adam_conv2d_1_bias_vIdentity_92:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_93IdentityRestoreV2:tensors:93"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_93AssignVariableOp1assignvariableop_93_cond_1_adam_conv2d_2_kernel_vIdentity_93:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_94IdentityRestoreV2:tensors:94"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_94AssignVariableOp/assignvariableop_94_cond_1_adam_conv2d_2_bias_vIdentity_94:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_95IdentityRestoreV2:tensors:95"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_95AssignVariableOp1assignvariableop_95_cond_1_adam_conv2d_3_kernel_vIdentity_95:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_96IdentityRestoreV2:tensors:96"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_96AssignVariableOp/assignvariableop_96_cond_1_adam_conv2d_3_bias_vIdentity_96:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_97IdentityRestoreV2:tensors:97"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_97AssignVariableOp1assignvariableop_97_cond_1_adam_conv2d_4_kernel_vIdentity_97:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_98IdentityRestoreV2:tensors:98"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_98AssignVariableOp/assignvariableop_98_cond_1_adam_conv2d_4_bias_vIdentity_98:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_99IdentityRestoreV2:tensors:99"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_99AssignVariableOp1assignvariableop_99_cond_1_adam_conv2d_5_kernel_vIdentity_99:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_100IdentityRestoreV2:tensors:100"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_100AssignVariableOp0assignvariableop_100_cond_1_adam_conv2d_5_bias_vIdentity_100:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_101IdentityRestoreV2:tensors:101"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_101AssignVariableOp2assignvariableop_101_cond_1_adam_conv2d_6_kernel_vIdentity_101:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_102IdentityRestoreV2:tensors:102"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_102AssignVariableOp0assignvariableop_102_cond_1_adam_conv2d_6_bias_vIdentity_102:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_103IdentityRestoreV2:tensors:103"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_103AssignVariableOp2assignvariableop_103_cond_1_adam_conv2d_7_kernel_vIdentity_103:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_104IdentityRestoreV2:tensors:104"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_104AssignVariableOp0assignvariableop_104_cond_1_adam_conv2d_7_bias_vIdentity_104:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_105IdentityRestoreV2:tensors:105"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_105AssignVariableOp2assignvariableop_105_cond_1_adam_conv2d_8_kernel_vIdentity_105:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_106IdentityRestoreV2:tensors:106"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_106AssignVariableOp0assignvariableop_106_cond_1_adam_conv2d_8_bias_vIdentity_106:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_107IdentityRestoreV2:tensors:107"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_107AssignVariableOp2assignvariableop_107_cond_1_adam_conv2d_9_kernel_vIdentity_107:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_108IdentityRestoreV2:tensors:108"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_108AssignVariableOp0assignvariableop_108_cond_1_adam_conv2d_9_bias_vIdentity_108:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_109IdentityRestoreV2:tensors:109"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_109AssignVariableOp3assignvariableop_109_cond_1_adam_conv2d_10_kernel_vIdentity_109:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_110IdentityRestoreV2:tensors:110"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_110AssignVariableOp1assignvariableop_110_cond_1_adam_conv2d_10_bias_vIdentity_110:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_111IdentityRestoreV2:tensors:111"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_111AssignVariableOp3assignvariableop_111_cond_1_adam_conv2d_11_kernel_vIdentity_111:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_112IdentityRestoreV2:tensors:112"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_112AssignVariableOp1assignvariableop_112_cond_1_adam_conv2d_11_bias_vIdentity_112:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_113IdentityRestoreV2:tensors:113"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_113AssignVariableOp:assignvariableop_113_cond_1_adam_conv2d_transpose_kernel_vIdentity_113:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_114IdentityRestoreV2:tensors:114"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_114AssignVariableOp8assignvariableop_114_cond_1_adam_conv2d_transpose_bias_vIdentity_114:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_115IdentityRestoreV2:tensors:115"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_115AssignVariableOp3assignvariableop_115_cond_1_adam_conv2d_12_kernel_vIdentity_115:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_116IdentityRestoreV2:tensors:116"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_116AssignVariableOp1assignvariableop_116_cond_1_adam_conv2d_12_bias_vIdentity_116:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_117IdentityRestoreV2:tensors:117"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_117AssignVariableOp3assignvariableop_117_cond_1_adam_conv2d_13_kernel_vIdentity_117:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_118IdentityRestoreV2:tensors:118"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_118AssignVariableOp1assignvariableop_118_cond_1_adam_conv2d_13_bias_vIdentity_118:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_119IdentityRestoreV2:tensors:119"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_119AssignVariableOp<assignvariableop_119_cond_1_adam_conv2d_transpose_1_kernel_vIdentity_119:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_120IdentityRestoreV2:tensors:120"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_120AssignVariableOp:assignvariableop_120_cond_1_adam_conv2d_transpose_1_bias_vIdentity_120:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_121IdentityRestoreV2:tensors:121"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_121AssignVariableOp3assignvariableop_121_cond_1_adam_conv2d_14_kernel_vIdentity_121:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_122IdentityRestoreV2:tensors:122"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_122AssignVariableOp1assignvariableop_122_cond_1_adam_conv2d_14_bias_vIdentity_122:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_123IdentityRestoreV2:tensors:123"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_123AssignVariableOp3assignvariableop_123_cond_1_adam_conv2d_15_kernel_vIdentity_123:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_124IdentityRestoreV2:tensors:124"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_124AssignVariableOp1assignvariableop_124_cond_1_adam_conv2d_15_bias_vIdentity_124:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_125IdentityRestoreV2:tensors:125"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_125AssignVariableOp3assignvariableop_125_cond_1_adam_conv2d_16_kernel_vIdentity_125:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_126IdentityRestoreV2:tensors:126"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_126AssignVariableOp1assignvariableop_126_cond_1_adam_conv2d_16_bias_vIdentity_126:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_127IdentityRestoreV2:tensors:127"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_127AssignVariableOp3assignvariableop_127_cond_1_adam_conv2d_kernel_vhatIdentity_127:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_128IdentityRestoreV2:tensors:128"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_128AssignVariableOp1assignvariableop_128_cond_1_adam_conv2d_bias_vhatIdentity_128:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_129IdentityRestoreV2:tensors:129"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_129AssignVariableOp5assignvariableop_129_cond_1_adam_conv2d_1_kernel_vhatIdentity_129:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_130IdentityRestoreV2:tensors:130"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_130AssignVariableOp3assignvariableop_130_cond_1_adam_conv2d_1_bias_vhatIdentity_130:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_131IdentityRestoreV2:tensors:131"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_131AssignVariableOp5assignvariableop_131_cond_1_adam_conv2d_2_kernel_vhatIdentity_131:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_132IdentityRestoreV2:tensors:132"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_132AssignVariableOp3assignvariableop_132_cond_1_adam_conv2d_2_bias_vhatIdentity_132:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_133IdentityRestoreV2:tensors:133"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_133AssignVariableOp5assignvariableop_133_cond_1_adam_conv2d_3_kernel_vhatIdentity_133:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_134IdentityRestoreV2:tensors:134"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_134AssignVariableOp3assignvariableop_134_cond_1_adam_conv2d_3_bias_vhatIdentity_134:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_135IdentityRestoreV2:tensors:135"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_135AssignVariableOp5assignvariableop_135_cond_1_adam_conv2d_4_kernel_vhatIdentity_135:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_136IdentityRestoreV2:tensors:136"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_136AssignVariableOp3assignvariableop_136_cond_1_adam_conv2d_4_bias_vhatIdentity_136:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_137IdentityRestoreV2:tensors:137"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_137AssignVariableOp5assignvariableop_137_cond_1_adam_conv2d_5_kernel_vhatIdentity_137:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_138IdentityRestoreV2:tensors:138"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_138AssignVariableOp3assignvariableop_138_cond_1_adam_conv2d_5_bias_vhatIdentity_138:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_139IdentityRestoreV2:tensors:139"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_139AssignVariableOp5assignvariableop_139_cond_1_adam_conv2d_6_kernel_vhatIdentity_139:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_140IdentityRestoreV2:tensors:140"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_140AssignVariableOp3assignvariableop_140_cond_1_adam_conv2d_6_bias_vhatIdentity_140:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_141IdentityRestoreV2:tensors:141"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_141AssignVariableOp5assignvariableop_141_cond_1_adam_conv2d_7_kernel_vhatIdentity_141:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_142IdentityRestoreV2:tensors:142"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_142AssignVariableOp3assignvariableop_142_cond_1_adam_conv2d_7_bias_vhatIdentity_142:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_143IdentityRestoreV2:tensors:143"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_143AssignVariableOp5assignvariableop_143_cond_1_adam_conv2d_8_kernel_vhatIdentity_143:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_144IdentityRestoreV2:tensors:144"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_144AssignVariableOp3assignvariableop_144_cond_1_adam_conv2d_8_bias_vhatIdentity_144:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_145IdentityRestoreV2:tensors:145"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_145AssignVariableOp5assignvariableop_145_cond_1_adam_conv2d_9_kernel_vhatIdentity_145:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_146IdentityRestoreV2:tensors:146"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_146AssignVariableOp3assignvariableop_146_cond_1_adam_conv2d_9_bias_vhatIdentity_146:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_147IdentityRestoreV2:tensors:147"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_147AssignVariableOp6assignvariableop_147_cond_1_adam_conv2d_10_kernel_vhatIdentity_147:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_148IdentityRestoreV2:tensors:148"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_148AssignVariableOp4assignvariableop_148_cond_1_adam_conv2d_10_bias_vhatIdentity_148:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_149IdentityRestoreV2:tensors:149"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_149AssignVariableOp6assignvariableop_149_cond_1_adam_conv2d_11_kernel_vhatIdentity_149:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_150IdentityRestoreV2:tensors:150"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_150AssignVariableOp4assignvariableop_150_cond_1_adam_conv2d_11_bias_vhatIdentity_150:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_151IdentityRestoreV2:tensors:151"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_151AssignVariableOp=assignvariableop_151_cond_1_adam_conv2d_transpose_kernel_vhatIdentity_151:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_152IdentityRestoreV2:tensors:152"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_152AssignVariableOp;assignvariableop_152_cond_1_adam_conv2d_transpose_bias_vhatIdentity_152:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_153IdentityRestoreV2:tensors:153"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_153AssignVariableOp6assignvariableop_153_cond_1_adam_conv2d_12_kernel_vhatIdentity_153:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_154IdentityRestoreV2:tensors:154"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_154AssignVariableOp4assignvariableop_154_cond_1_adam_conv2d_12_bias_vhatIdentity_154:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_155IdentityRestoreV2:tensors:155"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_155AssignVariableOp6assignvariableop_155_cond_1_adam_conv2d_13_kernel_vhatIdentity_155:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_156IdentityRestoreV2:tensors:156"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_156AssignVariableOp4assignvariableop_156_cond_1_adam_conv2d_13_bias_vhatIdentity_156:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_157IdentityRestoreV2:tensors:157"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_157AssignVariableOp?assignvariableop_157_cond_1_adam_conv2d_transpose_1_kernel_vhatIdentity_157:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_158IdentityRestoreV2:tensors:158"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_158AssignVariableOp=assignvariableop_158_cond_1_adam_conv2d_transpose_1_bias_vhatIdentity_158:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_159IdentityRestoreV2:tensors:159"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_159AssignVariableOp6assignvariableop_159_cond_1_adam_conv2d_14_kernel_vhatIdentity_159:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_160IdentityRestoreV2:tensors:160"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_160AssignVariableOp4assignvariableop_160_cond_1_adam_conv2d_14_bias_vhatIdentity_160:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_161IdentityRestoreV2:tensors:161"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_161AssignVariableOp6assignvariableop_161_cond_1_adam_conv2d_15_kernel_vhatIdentity_161:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_162IdentityRestoreV2:tensors:162"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_162AssignVariableOp4assignvariableop_162_cond_1_adam_conv2d_15_bias_vhatIdentity_162:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_163IdentityRestoreV2:tensors:163"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_163AssignVariableOp6assignvariableop_163_cond_1_adam_conv2d_16_kernel_vhatIdentity_163:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_164IdentityRestoreV2:tensors:164"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_164AssignVariableOp4assignvariableop_164_cond_1_adam_conv2d_16_bias_vhatIdentity_164:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 �
Identity_165Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_103^AssignVariableOp_104^AssignVariableOp_105^AssignVariableOp_106^AssignVariableOp_107^AssignVariableOp_108^AssignVariableOp_109^AssignVariableOp_11^AssignVariableOp_110^AssignVariableOp_111^AssignVariableOp_112^AssignVariableOp_113^AssignVariableOp_114^AssignVariableOp_115^AssignVariableOp_116^AssignVariableOp_117^AssignVariableOp_118^AssignVariableOp_119^AssignVariableOp_12^AssignVariableOp_120^AssignVariableOp_121^AssignVariableOp_122^AssignVariableOp_123^AssignVariableOp_124^AssignVariableOp_125^AssignVariableOp_126^AssignVariableOp_127^AssignVariableOp_128^AssignVariableOp_129^AssignVariableOp_13^AssignVariableOp_130^AssignVariableOp_131^AssignVariableOp_132^AssignVariableOp_133^AssignVariableOp_134^AssignVariableOp_135^AssignVariableOp_136^AssignVariableOp_137^AssignVariableOp_138^AssignVariableOp_139^AssignVariableOp_14^AssignVariableOp_140^AssignVariableOp_141^AssignVariableOp_142^AssignVariableOp_143^AssignVariableOp_144^AssignVariableOp_145^AssignVariableOp_146^AssignVariableOp_147^AssignVariableOp_148^AssignVariableOp_149^AssignVariableOp_15^AssignVariableOp_150^AssignVariableOp_151^AssignVariableOp_152^AssignVariableOp_153^AssignVariableOp_154^AssignVariableOp_155^AssignVariableOp_156^AssignVariableOp_157^AssignVariableOp_158^AssignVariableOp_159^AssignVariableOp_16^AssignVariableOp_160^AssignVariableOp_161^AssignVariableOp_162^AssignVariableOp_163^AssignVariableOp_164^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99^NoOp"/device:CPU:0*
T0*
_output_shapes
: Y
Identity_166IdentityIdentity_165:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_103^AssignVariableOp_104^AssignVariableOp_105^AssignVariableOp_106^AssignVariableOp_107^AssignVariableOp_108^AssignVariableOp_109^AssignVariableOp_11^AssignVariableOp_110^AssignVariableOp_111^AssignVariableOp_112^AssignVariableOp_113^AssignVariableOp_114^AssignVariableOp_115^AssignVariableOp_116^AssignVariableOp_117^AssignVariableOp_118^AssignVariableOp_119^AssignVariableOp_12^AssignVariableOp_120^AssignVariableOp_121^AssignVariableOp_122^AssignVariableOp_123^AssignVariableOp_124^AssignVariableOp_125^AssignVariableOp_126^AssignVariableOp_127^AssignVariableOp_128^AssignVariableOp_129^AssignVariableOp_13^AssignVariableOp_130^AssignVariableOp_131^AssignVariableOp_132^AssignVariableOp_133^AssignVariableOp_134^AssignVariableOp_135^AssignVariableOp_136^AssignVariableOp_137^AssignVariableOp_138^AssignVariableOp_139^AssignVariableOp_14^AssignVariableOp_140^AssignVariableOp_141^AssignVariableOp_142^AssignVariableOp_143^AssignVariableOp_144^AssignVariableOp_145^AssignVariableOp_146^AssignVariableOp_147^AssignVariableOp_148^AssignVariableOp_149^AssignVariableOp_15^AssignVariableOp_150^AssignVariableOp_151^AssignVariableOp_152^AssignVariableOp_153^AssignVariableOp_154^AssignVariableOp_155^AssignVariableOp_156^AssignVariableOp_157^AssignVariableOp_158^AssignVariableOp_159^AssignVariableOp_16^AssignVariableOp_160^AssignVariableOp_161^AssignVariableOp_162^AssignVariableOp_163^AssignVariableOp_164^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99*"
_acd_function_control_output(*
_output_shapes
 "%
identity_166Identity_166:output:0*�
_input_shapes�
�: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102,
AssignVariableOp_100AssignVariableOp_1002,
AssignVariableOp_101AssignVariableOp_1012,
AssignVariableOp_102AssignVariableOp_1022,
AssignVariableOp_103AssignVariableOp_1032,
AssignVariableOp_104AssignVariableOp_1042,
AssignVariableOp_105AssignVariableOp_1052,
AssignVariableOp_106AssignVariableOp_1062,
AssignVariableOp_107AssignVariableOp_1072,
AssignVariableOp_108AssignVariableOp_1082,
AssignVariableOp_109AssignVariableOp_1092*
AssignVariableOp_11AssignVariableOp_112,
AssignVariableOp_110AssignVariableOp_1102,
AssignVariableOp_111AssignVariableOp_1112,
AssignVariableOp_112AssignVariableOp_1122,
AssignVariableOp_113AssignVariableOp_1132,
AssignVariableOp_114AssignVariableOp_1142,
AssignVariableOp_115AssignVariableOp_1152,
AssignVariableOp_116AssignVariableOp_1162,
AssignVariableOp_117AssignVariableOp_1172,
AssignVariableOp_118AssignVariableOp_1182,
AssignVariableOp_119AssignVariableOp_1192*
AssignVariableOp_12AssignVariableOp_122,
AssignVariableOp_120AssignVariableOp_1202,
AssignVariableOp_121AssignVariableOp_1212,
AssignVariableOp_122AssignVariableOp_1222,
AssignVariableOp_123AssignVariableOp_1232,
AssignVariableOp_124AssignVariableOp_1242,
AssignVariableOp_125AssignVariableOp_1252,
AssignVariableOp_126AssignVariableOp_1262,
AssignVariableOp_127AssignVariableOp_1272,
AssignVariableOp_128AssignVariableOp_1282,
AssignVariableOp_129AssignVariableOp_1292*
AssignVariableOp_13AssignVariableOp_132,
AssignVariableOp_130AssignVariableOp_1302,
AssignVariableOp_131AssignVariableOp_1312,
AssignVariableOp_132AssignVariableOp_1322,
AssignVariableOp_133AssignVariableOp_1332,
AssignVariableOp_134AssignVariableOp_1342,
AssignVariableOp_135AssignVariableOp_1352,
AssignVariableOp_136AssignVariableOp_1362,
AssignVariableOp_137AssignVariableOp_1372,
AssignVariableOp_138AssignVariableOp_1382,
AssignVariableOp_139AssignVariableOp_1392*
AssignVariableOp_14AssignVariableOp_142,
AssignVariableOp_140AssignVariableOp_1402,
AssignVariableOp_141AssignVariableOp_1412,
AssignVariableOp_142AssignVariableOp_1422,
AssignVariableOp_143AssignVariableOp_1432,
AssignVariableOp_144AssignVariableOp_1442,
AssignVariableOp_145AssignVariableOp_1452,
AssignVariableOp_146AssignVariableOp_1462,
AssignVariableOp_147AssignVariableOp_1472,
AssignVariableOp_148AssignVariableOp_1482,
AssignVariableOp_149AssignVariableOp_1492*
AssignVariableOp_15AssignVariableOp_152,
AssignVariableOp_150AssignVariableOp_1502,
AssignVariableOp_151AssignVariableOp_1512,
AssignVariableOp_152AssignVariableOp_1522,
AssignVariableOp_153AssignVariableOp_1532,
AssignVariableOp_154AssignVariableOp_1542,
AssignVariableOp_155AssignVariableOp_1552,
AssignVariableOp_156AssignVariableOp_1562,
AssignVariableOp_157AssignVariableOp_1572,
AssignVariableOp_158AssignVariableOp_1582,
AssignVariableOp_159AssignVariableOp_1592*
AssignVariableOp_16AssignVariableOp_162,
AssignVariableOp_160AssignVariableOp_1602,
AssignVariableOp_161AssignVariableOp_1612,
AssignVariableOp_162AssignVariableOp_1622,
AssignVariableOp_163AssignVariableOp_1632,
AssignVariableOp_164AssignVariableOp_1642*
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
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_7AssignVariableOp_72*
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_71AssignVariableOp_712*
AssignVariableOp_72AssignVariableOp_722*
AssignVariableOp_73AssignVariableOp_732*
AssignVariableOp_74AssignVariableOp_742*
AssignVariableOp_75AssignVariableOp_752*
AssignVariableOp_76AssignVariableOp_762*
AssignVariableOp_77AssignVariableOp_772*
AssignVariableOp_78AssignVariableOp_782*
AssignVariableOp_79AssignVariableOp_792(
AssignVariableOp_8AssignVariableOp_82*
AssignVariableOp_80AssignVariableOp_802*
AssignVariableOp_81AssignVariableOp_812*
AssignVariableOp_82AssignVariableOp_822*
AssignVariableOp_83AssignVariableOp_832*
AssignVariableOp_84AssignVariableOp_842*
AssignVariableOp_85AssignVariableOp_852*
AssignVariableOp_86AssignVariableOp_862*
AssignVariableOp_87AssignVariableOp_872*
AssignVariableOp_88AssignVariableOp_882*
AssignVariableOp_89AssignVariableOp_892(
AssignVariableOp_9AssignVariableOp_92*
AssignVariableOp_90AssignVariableOp_902*
AssignVariableOp_91AssignVariableOp_912*
AssignVariableOp_92AssignVariableOp_922*
AssignVariableOp_93AssignVariableOp_932*
AssignVariableOp_94AssignVariableOp_942*
AssignVariableOp_95AssignVariableOp_952*
AssignVariableOp_96AssignVariableOp_962*
AssignVariableOp_97AssignVariableOp_972*
AssignVariableOp_98AssignVariableOp_982*
AssignVariableOp_99AssignVariableOp_99:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
j
N__inference_average_pooling2d_1_layer_call_and_return_conditional_losses_53032

inputs
identity�
AvgPoolAvgPoolinputs*
T0*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingSAME*
strides
{
IdentityIdentityAvgPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
�
C__inference_conv2d_4_layer_call_and_return_conditional_losses_53320

inputs8
conv2d_readvariableop_resource:<-
biasadd_readvariableop_resource:<
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:<*
dtype0r
Conv2D/CastCastConv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*&
_output_shapes
:<�
Conv2DConv2DinputsConv2D/Cast:y:0*
T0*1
_output_shapes
:�����������<*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:<*
dtype0h
BiasAdd/CastCastBiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:<q
BiasAddBiasAddConv2D:output:0BiasAdd/Cast:y:0*
T0*1
_output_shapes
:�����������<H
mul/xConst*
_output_shapes
: *
dtype0*
value
B j�ph
mulMulmul/x:output:0BiasAdd:output:0*
T0*1
_output_shapes
:�����������<I
Pow/yConst*
_output_shapes
: *
dtype0*
valueB	 j��h
PowPowBiasAdd:output:0Pow/y:output:0*
T0*1
_output_shapes
:�����������<J
mul_1/xConst*
_output_shapes
: *
dtype0*
value
B j�Sc
mul_1Mulmul_1/x:output:0Pow:z:0*
T0*1
_output_shapes
:�����������<e
addAddV2BiasAdd:output:0	mul_1:z:0*
T0*1
_output_shapes
:�����������<J
mul_2/xConst*
_output_shapes
: *
dtype0*
value
B j�tc
mul_2Mulmul_2/x:output:0add:z:0*
T0*1
_output_shapes
:�����������<S
TanhTanh	mul_2:z:0*
T0*1
_output_shapes
:�����������<J
add_1/xConst*
_output_shapes
: *
dtype0*
value
B j�xf
add_1AddV2add_1/x:output:0Tanh:y:0*
T0*1
_output_shapes
:�����������<\
mul_3Mulmul:z:0	add_1:z:0*
T0*1
_output_shapes
:�����������<b
IdentityIdentity	mul_3:z:0^NoOp*
T0*1
_output_shapes
:�����������<w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:�����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
��
�L
__inference__traced_save_56936
file_prefix,
(savev2_conv2d_kernel_read_readvariableop*
&savev2_conv2d_bias_read_readvariableop.
*savev2_conv2d_1_kernel_read_readvariableop,
(savev2_conv2d_1_bias_read_readvariableop.
*savev2_conv2d_2_kernel_read_readvariableop,
(savev2_conv2d_2_bias_read_readvariableop.
*savev2_conv2d_3_kernel_read_readvariableop,
(savev2_conv2d_3_bias_read_readvariableop.
*savev2_conv2d_4_kernel_read_readvariableop,
(savev2_conv2d_4_bias_read_readvariableop.
*savev2_conv2d_5_kernel_read_readvariableop,
(savev2_conv2d_5_bias_read_readvariableop.
*savev2_conv2d_6_kernel_read_readvariableop,
(savev2_conv2d_6_bias_read_readvariableop.
*savev2_conv2d_7_kernel_read_readvariableop,
(savev2_conv2d_7_bias_read_readvariableop.
*savev2_conv2d_8_kernel_read_readvariableop,
(savev2_conv2d_8_bias_read_readvariableop.
*savev2_conv2d_9_kernel_read_readvariableop,
(savev2_conv2d_9_bias_read_readvariableop/
+savev2_conv2d_10_kernel_read_readvariableop-
)savev2_conv2d_10_bias_read_readvariableop/
+savev2_conv2d_11_kernel_read_readvariableop-
)savev2_conv2d_11_bias_read_readvariableop6
2savev2_conv2d_transpose_kernel_read_readvariableop4
0savev2_conv2d_transpose_bias_read_readvariableop/
+savev2_conv2d_12_kernel_read_readvariableop-
)savev2_conv2d_12_bias_read_readvariableop/
+savev2_conv2d_13_kernel_read_readvariableop-
)savev2_conv2d_13_bias_read_readvariableop8
4savev2_conv2d_transpose_1_kernel_read_readvariableop6
2savev2_conv2d_transpose_1_bias_read_readvariableop/
+savev2_conv2d_14_kernel_read_readvariableop-
)savev2_conv2d_14_bias_read_readvariableop/
+savev2_conv2d_15_kernel_read_readvariableop-
)savev2_conv2d_15_bias_read_readvariableop/
+savev2_conv2d_16_kernel_read_readvariableop-
)savev2_conv2d_16_bias_read_readvariableop/
+savev2_cond_1_adam_iter_read_readvariableop	1
-savev2_cond_1_adam_beta_1_read_readvariableop1
-savev2_cond_1_adam_beta_2_read_readvariableop0
,savev2_cond_1_adam_decay_read_readvariableop8
4savev2_cond_1_adam_learning_rate_read_readvariableop1
-savev2_current_loss_scale_read_readvariableop)
%savev2_good_steps_read_readvariableop	&
"savev2_total_2_read_readvariableop&
"savev2_count_2_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop:
6savev2_cond_1_adam_conv2d_kernel_m_read_readvariableop8
4savev2_cond_1_adam_conv2d_bias_m_read_readvariableop<
8savev2_cond_1_adam_conv2d_1_kernel_m_read_readvariableop:
6savev2_cond_1_adam_conv2d_1_bias_m_read_readvariableop<
8savev2_cond_1_adam_conv2d_2_kernel_m_read_readvariableop:
6savev2_cond_1_adam_conv2d_2_bias_m_read_readvariableop<
8savev2_cond_1_adam_conv2d_3_kernel_m_read_readvariableop:
6savev2_cond_1_adam_conv2d_3_bias_m_read_readvariableop<
8savev2_cond_1_adam_conv2d_4_kernel_m_read_readvariableop:
6savev2_cond_1_adam_conv2d_4_bias_m_read_readvariableop<
8savev2_cond_1_adam_conv2d_5_kernel_m_read_readvariableop:
6savev2_cond_1_adam_conv2d_5_bias_m_read_readvariableop<
8savev2_cond_1_adam_conv2d_6_kernel_m_read_readvariableop:
6savev2_cond_1_adam_conv2d_6_bias_m_read_readvariableop<
8savev2_cond_1_adam_conv2d_7_kernel_m_read_readvariableop:
6savev2_cond_1_adam_conv2d_7_bias_m_read_readvariableop<
8savev2_cond_1_adam_conv2d_8_kernel_m_read_readvariableop:
6savev2_cond_1_adam_conv2d_8_bias_m_read_readvariableop<
8savev2_cond_1_adam_conv2d_9_kernel_m_read_readvariableop:
6savev2_cond_1_adam_conv2d_9_bias_m_read_readvariableop=
9savev2_cond_1_adam_conv2d_10_kernel_m_read_readvariableop;
7savev2_cond_1_adam_conv2d_10_bias_m_read_readvariableop=
9savev2_cond_1_adam_conv2d_11_kernel_m_read_readvariableop;
7savev2_cond_1_adam_conv2d_11_bias_m_read_readvariableopD
@savev2_cond_1_adam_conv2d_transpose_kernel_m_read_readvariableopB
>savev2_cond_1_adam_conv2d_transpose_bias_m_read_readvariableop=
9savev2_cond_1_adam_conv2d_12_kernel_m_read_readvariableop;
7savev2_cond_1_adam_conv2d_12_bias_m_read_readvariableop=
9savev2_cond_1_adam_conv2d_13_kernel_m_read_readvariableop;
7savev2_cond_1_adam_conv2d_13_bias_m_read_readvariableopF
Bsavev2_cond_1_adam_conv2d_transpose_1_kernel_m_read_readvariableopD
@savev2_cond_1_adam_conv2d_transpose_1_bias_m_read_readvariableop=
9savev2_cond_1_adam_conv2d_14_kernel_m_read_readvariableop;
7savev2_cond_1_adam_conv2d_14_bias_m_read_readvariableop=
9savev2_cond_1_adam_conv2d_15_kernel_m_read_readvariableop;
7savev2_cond_1_adam_conv2d_15_bias_m_read_readvariableop=
9savev2_cond_1_adam_conv2d_16_kernel_m_read_readvariableop;
7savev2_cond_1_adam_conv2d_16_bias_m_read_readvariableop:
6savev2_cond_1_adam_conv2d_kernel_v_read_readvariableop8
4savev2_cond_1_adam_conv2d_bias_v_read_readvariableop<
8savev2_cond_1_adam_conv2d_1_kernel_v_read_readvariableop:
6savev2_cond_1_adam_conv2d_1_bias_v_read_readvariableop<
8savev2_cond_1_adam_conv2d_2_kernel_v_read_readvariableop:
6savev2_cond_1_adam_conv2d_2_bias_v_read_readvariableop<
8savev2_cond_1_adam_conv2d_3_kernel_v_read_readvariableop:
6savev2_cond_1_adam_conv2d_3_bias_v_read_readvariableop<
8savev2_cond_1_adam_conv2d_4_kernel_v_read_readvariableop:
6savev2_cond_1_adam_conv2d_4_bias_v_read_readvariableop<
8savev2_cond_1_adam_conv2d_5_kernel_v_read_readvariableop:
6savev2_cond_1_adam_conv2d_5_bias_v_read_readvariableop<
8savev2_cond_1_adam_conv2d_6_kernel_v_read_readvariableop:
6savev2_cond_1_adam_conv2d_6_bias_v_read_readvariableop<
8savev2_cond_1_adam_conv2d_7_kernel_v_read_readvariableop:
6savev2_cond_1_adam_conv2d_7_bias_v_read_readvariableop<
8savev2_cond_1_adam_conv2d_8_kernel_v_read_readvariableop:
6savev2_cond_1_adam_conv2d_8_bias_v_read_readvariableop<
8savev2_cond_1_adam_conv2d_9_kernel_v_read_readvariableop:
6savev2_cond_1_adam_conv2d_9_bias_v_read_readvariableop=
9savev2_cond_1_adam_conv2d_10_kernel_v_read_readvariableop;
7savev2_cond_1_adam_conv2d_10_bias_v_read_readvariableop=
9savev2_cond_1_adam_conv2d_11_kernel_v_read_readvariableop;
7savev2_cond_1_adam_conv2d_11_bias_v_read_readvariableopD
@savev2_cond_1_adam_conv2d_transpose_kernel_v_read_readvariableopB
>savev2_cond_1_adam_conv2d_transpose_bias_v_read_readvariableop=
9savev2_cond_1_adam_conv2d_12_kernel_v_read_readvariableop;
7savev2_cond_1_adam_conv2d_12_bias_v_read_readvariableop=
9savev2_cond_1_adam_conv2d_13_kernel_v_read_readvariableop;
7savev2_cond_1_adam_conv2d_13_bias_v_read_readvariableopF
Bsavev2_cond_1_adam_conv2d_transpose_1_kernel_v_read_readvariableopD
@savev2_cond_1_adam_conv2d_transpose_1_bias_v_read_readvariableop=
9savev2_cond_1_adam_conv2d_14_kernel_v_read_readvariableop;
7savev2_cond_1_adam_conv2d_14_bias_v_read_readvariableop=
9savev2_cond_1_adam_conv2d_15_kernel_v_read_readvariableop;
7savev2_cond_1_adam_conv2d_15_bias_v_read_readvariableop=
9savev2_cond_1_adam_conv2d_16_kernel_v_read_readvariableop;
7savev2_cond_1_adam_conv2d_16_bias_v_read_readvariableop=
9savev2_cond_1_adam_conv2d_kernel_vhat_read_readvariableop;
7savev2_cond_1_adam_conv2d_bias_vhat_read_readvariableop?
;savev2_cond_1_adam_conv2d_1_kernel_vhat_read_readvariableop=
9savev2_cond_1_adam_conv2d_1_bias_vhat_read_readvariableop?
;savev2_cond_1_adam_conv2d_2_kernel_vhat_read_readvariableop=
9savev2_cond_1_adam_conv2d_2_bias_vhat_read_readvariableop?
;savev2_cond_1_adam_conv2d_3_kernel_vhat_read_readvariableop=
9savev2_cond_1_adam_conv2d_3_bias_vhat_read_readvariableop?
;savev2_cond_1_adam_conv2d_4_kernel_vhat_read_readvariableop=
9savev2_cond_1_adam_conv2d_4_bias_vhat_read_readvariableop?
;savev2_cond_1_adam_conv2d_5_kernel_vhat_read_readvariableop=
9savev2_cond_1_adam_conv2d_5_bias_vhat_read_readvariableop?
;savev2_cond_1_adam_conv2d_6_kernel_vhat_read_readvariableop=
9savev2_cond_1_adam_conv2d_6_bias_vhat_read_readvariableop?
;savev2_cond_1_adam_conv2d_7_kernel_vhat_read_readvariableop=
9savev2_cond_1_adam_conv2d_7_bias_vhat_read_readvariableop?
;savev2_cond_1_adam_conv2d_8_kernel_vhat_read_readvariableop=
9savev2_cond_1_adam_conv2d_8_bias_vhat_read_readvariableop?
;savev2_cond_1_adam_conv2d_9_kernel_vhat_read_readvariableop=
9savev2_cond_1_adam_conv2d_9_bias_vhat_read_readvariableop@
<savev2_cond_1_adam_conv2d_10_kernel_vhat_read_readvariableop>
:savev2_cond_1_adam_conv2d_10_bias_vhat_read_readvariableop@
<savev2_cond_1_adam_conv2d_11_kernel_vhat_read_readvariableop>
:savev2_cond_1_adam_conv2d_11_bias_vhat_read_readvariableopG
Csavev2_cond_1_adam_conv2d_transpose_kernel_vhat_read_readvariableopE
Asavev2_cond_1_adam_conv2d_transpose_bias_vhat_read_readvariableop@
<savev2_cond_1_adam_conv2d_12_kernel_vhat_read_readvariableop>
:savev2_cond_1_adam_conv2d_12_bias_vhat_read_readvariableop@
<savev2_cond_1_adam_conv2d_13_kernel_vhat_read_readvariableop>
:savev2_cond_1_adam_conv2d_13_bias_vhat_read_readvariableopI
Esavev2_cond_1_adam_conv2d_transpose_1_kernel_vhat_read_readvariableopG
Csavev2_cond_1_adam_conv2d_transpose_1_bias_vhat_read_readvariableop@
<savev2_cond_1_adam_conv2d_14_kernel_vhat_read_readvariableop>
:savev2_cond_1_adam_conv2d_14_bias_vhat_read_readvariableop@
<savev2_cond_1_adam_conv2d_15_kernel_vhat_read_readvariableop>
:savev2_cond_1_adam_conv2d_15_bias_vhat_read_readvariableop@
<savev2_cond_1_adam_conv2d_16_kernel_vhat_read_readvariableop>
:savev2_cond_1_adam_conv2d_16_bias_vhat_read_readvariableop
savev2_const

identity_1��MergeV2Checkpointsw
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
_temp/part�
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
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: �b
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes	
:�*
dtype0*�a
value�aB�a�B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-17/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-17/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-18/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-18/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEBBoptimizer/loss_scale/current_loss_scale/.ATTRIBUTES/VARIABLE_VALUEB:optimizer/loss_scale/good_steps/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-16/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-16/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-18/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-18/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-16/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-16/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-18/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-18/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-16/kernel/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-16/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-18/kernel/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-18/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes	
:�*
dtype0*�
value�B��B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �H
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0(savev2_conv2d_kernel_read_readvariableop&savev2_conv2d_bias_read_readvariableop*savev2_conv2d_1_kernel_read_readvariableop(savev2_conv2d_1_bias_read_readvariableop*savev2_conv2d_2_kernel_read_readvariableop(savev2_conv2d_2_bias_read_readvariableop*savev2_conv2d_3_kernel_read_readvariableop(savev2_conv2d_3_bias_read_readvariableop*savev2_conv2d_4_kernel_read_readvariableop(savev2_conv2d_4_bias_read_readvariableop*savev2_conv2d_5_kernel_read_readvariableop(savev2_conv2d_5_bias_read_readvariableop*savev2_conv2d_6_kernel_read_readvariableop(savev2_conv2d_6_bias_read_readvariableop*savev2_conv2d_7_kernel_read_readvariableop(savev2_conv2d_7_bias_read_readvariableop*savev2_conv2d_8_kernel_read_readvariableop(savev2_conv2d_8_bias_read_readvariableop*savev2_conv2d_9_kernel_read_readvariableop(savev2_conv2d_9_bias_read_readvariableop+savev2_conv2d_10_kernel_read_readvariableop)savev2_conv2d_10_bias_read_readvariableop+savev2_conv2d_11_kernel_read_readvariableop)savev2_conv2d_11_bias_read_readvariableop2savev2_conv2d_transpose_kernel_read_readvariableop0savev2_conv2d_transpose_bias_read_readvariableop+savev2_conv2d_12_kernel_read_readvariableop)savev2_conv2d_12_bias_read_readvariableop+savev2_conv2d_13_kernel_read_readvariableop)savev2_conv2d_13_bias_read_readvariableop4savev2_conv2d_transpose_1_kernel_read_readvariableop2savev2_conv2d_transpose_1_bias_read_readvariableop+savev2_conv2d_14_kernel_read_readvariableop)savev2_conv2d_14_bias_read_readvariableop+savev2_conv2d_15_kernel_read_readvariableop)savev2_conv2d_15_bias_read_readvariableop+savev2_conv2d_16_kernel_read_readvariableop)savev2_conv2d_16_bias_read_readvariableop+savev2_cond_1_adam_iter_read_readvariableop-savev2_cond_1_adam_beta_1_read_readvariableop-savev2_cond_1_adam_beta_2_read_readvariableop,savev2_cond_1_adam_decay_read_readvariableop4savev2_cond_1_adam_learning_rate_read_readvariableop-savev2_current_loss_scale_read_readvariableop%savev2_good_steps_read_readvariableop"savev2_total_2_read_readvariableop"savev2_count_2_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop6savev2_cond_1_adam_conv2d_kernel_m_read_readvariableop4savev2_cond_1_adam_conv2d_bias_m_read_readvariableop8savev2_cond_1_adam_conv2d_1_kernel_m_read_readvariableop6savev2_cond_1_adam_conv2d_1_bias_m_read_readvariableop8savev2_cond_1_adam_conv2d_2_kernel_m_read_readvariableop6savev2_cond_1_adam_conv2d_2_bias_m_read_readvariableop8savev2_cond_1_adam_conv2d_3_kernel_m_read_readvariableop6savev2_cond_1_adam_conv2d_3_bias_m_read_readvariableop8savev2_cond_1_adam_conv2d_4_kernel_m_read_readvariableop6savev2_cond_1_adam_conv2d_4_bias_m_read_readvariableop8savev2_cond_1_adam_conv2d_5_kernel_m_read_readvariableop6savev2_cond_1_adam_conv2d_5_bias_m_read_readvariableop8savev2_cond_1_adam_conv2d_6_kernel_m_read_readvariableop6savev2_cond_1_adam_conv2d_6_bias_m_read_readvariableop8savev2_cond_1_adam_conv2d_7_kernel_m_read_readvariableop6savev2_cond_1_adam_conv2d_7_bias_m_read_readvariableop8savev2_cond_1_adam_conv2d_8_kernel_m_read_readvariableop6savev2_cond_1_adam_conv2d_8_bias_m_read_readvariableop8savev2_cond_1_adam_conv2d_9_kernel_m_read_readvariableop6savev2_cond_1_adam_conv2d_9_bias_m_read_readvariableop9savev2_cond_1_adam_conv2d_10_kernel_m_read_readvariableop7savev2_cond_1_adam_conv2d_10_bias_m_read_readvariableop9savev2_cond_1_adam_conv2d_11_kernel_m_read_readvariableop7savev2_cond_1_adam_conv2d_11_bias_m_read_readvariableop@savev2_cond_1_adam_conv2d_transpose_kernel_m_read_readvariableop>savev2_cond_1_adam_conv2d_transpose_bias_m_read_readvariableop9savev2_cond_1_adam_conv2d_12_kernel_m_read_readvariableop7savev2_cond_1_adam_conv2d_12_bias_m_read_readvariableop9savev2_cond_1_adam_conv2d_13_kernel_m_read_readvariableop7savev2_cond_1_adam_conv2d_13_bias_m_read_readvariableopBsavev2_cond_1_adam_conv2d_transpose_1_kernel_m_read_readvariableop@savev2_cond_1_adam_conv2d_transpose_1_bias_m_read_readvariableop9savev2_cond_1_adam_conv2d_14_kernel_m_read_readvariableop7savev2_cond_1_adam_conv2d_14_bias_m_read_readvariableop9savev2_cond_1_adam_conv2d_15_kernel_m_read_readvariableop7savev2_cond_1_adam_conv2d_15_bias_m_read_readvariableop9savev2_cond_1_adam_conv2d_16_kernel_m_read_readvariableop7savev2_cond_1_adam_conv2d_16_bias_m_read_readvariableop6savev2_cond_1_adam_conv2d_kernel_v_read_readvariableop4savev2_cond_1_adam_conv2d_bias_v_read_readvariableop8savev2_cond_1_adam_conv2d_1_kernel_v_read_readvariableop6savev2_cond_1_adam_conv2d_1_bias_v_read_readvariableop8savev2_cond_1_adam_conv2d_2_kernel_v_read_readvariableop6savev2_cond_1_adam_conv2d_2_bias_v_read_readvariableop8savev2_cond_1_adam_conv2d_3_kernel_v_read_readvariableop6savev2_cond_1_adam_conv2d_3_bias_v_read_readvariableop8savev2_cond_1_adam_conv2d_4_kernel_v_read_readvariableop6savev2_cond_1_adam_conv2d_4_bias_v_read_readvariableop8savev2_cond_1_adam_conv2d_5_kernel_v_read_readvariableop6savev2_cond_1_adam_conv2d_5_bias_v_read_readvariableop8savev2_cond_1_adam_conv2d_6_kernel_v_read_readvariableop6savev2_cond_1_adam_conv2d_6_bias_v_read_readvariableop8savev2_cond_1_adam_conv2d_7_kernel_v_read_readvariableop6savev2_cond_1_adam_conv2d_7_bias_v_read_readvariableop8savev2_cond_1_adam_conv2d_8_kernel_v_read_readvariableop6savev2_cond_1_adam_conv2d_8_bias_v_read_readvariableop8savev2_cond_1_adam_conv2d_9_kernel_v_read_readvariableop6savev2_cond_1_adam_conv2d_9_bias_v_read_readvariableop9savev2_cond_1_adam_conv2d_10_kernel_v_read_readvariableop7savev2_cond_1_adam_conv2d_10_bias_v_read_readvariableop9savev2_cond_1_adam_conv2d_11_kernel_v_read_readvariableop7savev2_cond_1_adam_conv2d_11_bias_v_read_readvariableop@savev2_cond_1_adam_conv2d_transpose_kernel_v_read_readvariableop>savev2_cond_1_adam_conv2d_transpose_bias_v_read_readvariableop9savev2_cond_1_adam_conv2d_12_kernel_v_read_readvariableop7savev2_cond_1_adam_conv2d_12_bias_v_read_readvariableop9savev2_cond_1_adam_conv2d_13_kernel_v_read_readvariableop7savev2_cond_1_adam_conv2d_13_bias_v_read_readvariableopBsavev2_cond_1_adam_conv2d_transpose_1_kernel_v_read_readvariableop@savev2_cond_1_adam_conv2d_transpose_1_bias_v_read_readvariableop9savev2_cond_1_adam_conv2d_14_kernel_v_read_readvariableop7savev2_cond_1_adam_conv2d_14_bias_v_read_readvariableop9savev2_cond_1_adam_conv2d_15_kernel_v_read_readvariableop7savev2_cond_1_adam_conv2d_15_bias_v_read_readvariableop9savev2_cond_1_adam_conv2d_16_kernel_v_read_readvariableop7savev2_cond_1_adam_conv2d_16_bias_v_read_readvariableop9savev2_cond_1_adam_conv2d_kernel_vhat_read_readvariableop7savev2_cond_1_adam_conv2d_bias_vhat_read_readvariableop;savev2_cond_1_adam_conv2d_1_kernel_vhat_read_readvariableop9savev2_cond_1_adam_conv2d_1_bias_vhat_read_readvariableop;savev2_cond_1_adam_conv2d_2_kernel_vhat_read_readvariableop9savev2_cond_1_adam_conv2d_2_bias_vhat_read_readvariableop;savev2_cond_1_adam_conv2d_3_kernel_vhat_read_readvariableop9savev2_cond_1_adam_conv2d_3_bias_vhat_read_readvariableop;savev2_cond_1_adam_conv2d_4_kernel_vhat_read_readvariableop9savev2_cond_1_adam_conv2d_4_bias_vhat_read_readvariableop;savev2_cond_1_adam_conv2d_5_kernel_vhat_read_readvariableop9savev2_cond_1_adam_conv2d_5_bias_vhat_read_readvariableop;savev2_cond_1_adam_conv2d_6_kernel_vhat_read_readvariableop9savev2_cond_1_adam_conv2d_6_bias_vhat_read_readvariableop;savev2_cond_1_adam_conv2d_7_kernel_vhat_read_readvariableop9savev2_cond_1_adam_conv2d_7_bias_vhat_read_readvariableop;savev2_cond_1_adam_conv2d_8_kernel_vhat_read_readvariableop9savev2_cond_1_adam_conv2d_8_bias_vhat_read_readvariableop;savev2_cond_1_adam_conv2d_9_kernel_vhat_read_readvariableop9savev2_cond_1_adam_conv2d_9_bias_vhat_read_readvariableop<savev2_cond_1_adam_conv2d_10_kernel_vhat_read_readvariableop:savev2_cond_1_adam_conv2d_10_bias_vhat_read_readvariableop<savev2_cond_1_adam_conv2d_11_kernel_vhat_read_readvariableop:savev2_cond_1_adam_conv2d_11_bias_vhat_read_readvariableopCsavev2_cond_1_adam_conv2d_transpose_kernel_vhat_read_readvariableopAsavev2_cond_1_adam_conv2d_transpose_bias_vhat_read_readvariableop<savev2_cond_1_adam_conv2d_12_kernel_vhat_read_readvariableop:savev2_cond_1_adam_conv2d_12_bias_vhat_read_readvariableop<savev2_cond_1_adam_conv2d_13_kernel_vhat_read_readvariableop:savev2_cond_1_adam_conv2d_13_bias_vhat_read_readvariableopEsavev2_cond_1_adam_conv2d_transpose_1_kernel_vhat_read_readvariableopCsavev2_cond_1_adam_conv2d_transpose_1_bias_vhat_read_readvariableop<savev2_cond_1_adam_conv2d_14_kernel_vhat_read_readvariableop:savev2_cond_1_adam_conv2d_14_bias_vhat_read_readvariableop<savev2_cond_1_adam_conv2d_15_kernel_vhat_read_readvariableop:savev2_cond_1_adam_conv2d_15_bias_vhat_read_readvariableop<savev2_cond_1_adam_conv2d_16_kernel_vhat_read_readvariableop:savev2_cond_1_adam_conv2d_16_bias_vhat_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *�
dtypes�
�2�		�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
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

identity_1Identity_1:output:0*�
_input_shapes�
�: :::::::::<:<:<<:<:<x:x:xx:x:x�:�:��:�:��:�:��:�:x�:x:xx:x:xx:x:<x:<:<<:<:<<:<:<:: : : : : : : : : : : : : :::::::::<:<:<<:<:<x:x:xx:x:x�:�:��:�:��:�:��:�:x�:x:xx:x:xx:x:<x:<:<<:<:<<:<:<::::::::::<:<:<<:<:<x:x:xx:x:x�:�:��:�:��:�:��:�:x�:x:xx:x:xx:x:<x:<:<<:<:<<:<:<::::::::::<:<:<<:<:<x:x:xx:x:x�:�:��:�:��:�:��:�:x�:x:xx:x:xx:x:<x:<:<<:<:<<:<:<:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::,	(
&
_output_shapes
:<: 


_output_shapes
:<:,(
&
_output_shapes
:<<: 

_output_shapes
:<:,(
&
_output_shapes
:<x: 

_output_shapes
:x:,(
&
_output_shapes
:xx: 

_output_shapes
:x:-)
'
_output_shapes
:x�:!

_output_shapes	
:�:.*
(
_output_shapes
:��:!

_output_shapes	
:�:.*
(
_output_shapes
:��:!

_output_shapes	
:�:.*
(
_output_shapes
:��:!

_output_shapes	
:�:-)
'
_output_shapes
:x�: 

_output_shapes
:x:,(
&
_output_shapes
:xx: 

_output_shapes
:x:,(
&
_output_shapes
:xx: 

_output_shapes
:x:,(
&
_output_shapes
:<x:  

_output_shapes
:<:,!(
&
_output_shapes
:<<: "

_output_shapes
:<:,#(
&
_output_shapes
:<<: $

_output_shapes
:<:,%(
&
_output_shapes
:<: &

_output_shapes
::'

_output_shapes
: :(

_output_shapes
: :)

_output_shapes
: :*

_output_shapes
: :+

_output_shapes
: :,

_output_shapes
: :-

_output_shapes
: :.

_output_shapes
: :/

_output_shapes
: :0

_output_shapes
: :1

_output_shapes
: :2

_output_shapes
: :3

_output_shapes
: :,4(
&
_output_shapes
:: 5

_output_shapes
::,6(
&
_output_shapes
:: 7

_output_shapes
::,8(
&
_output_shapes
:: 9

_output_shapes
::,:(
&
_output_shapes
:: ;

_output_shapes
::,<(
&
_output_shapes
:<: =

_output_shapes
:<:,>(
&
_output_shapes
:<<: ?

_output_shapes
:<:,@(
&
_output_shapes
:<x: A

_output_shapes
:x:,B(
&
_output_shapes
:xx: C

_output_shapes
:x:-D)
'
_output_shapes
:x�:!E

_output_shapes	
:�:.F*
(
_output_shapes
:��:!G

_output_shapes	
:�:.H*
(
_output_shapes
:��:!I

_output_shapes	
:�:.J*
(
_output_shapes
:��:!K

_output_shapes	
:�:-L)
'
_output_shapes
:x�: M

_output_shapes
:x:,N(
&
_output_shapes
:xx: O

_output_shapes
:x:,P(
&
_output_shapes
:xx: Q

_output_shapes
:x:,R(
&
_output_shapes
:<x: S

_output_shapes
:<:,T(
&
_output_shapes
:<<: U

_output_shapes
:<:,V(
&
_output_shapes
:<<: W

_output_shapes
:<:,X(
&
_output_shapes
:<: Y

_output_shapes
::,Z(
&
_output_shapes
:: [

_output_shapes
::,\(
&
_output_shapes
:: ]

_output_shapes
::,^(
&
_output_shapes
:: _

_output_shapes
::,`(
&
_output_shapes
:: a

_output_shapes
::,b(
&
_output_shapes
:<: c

_output_shapes
:<:,d(
&
_output_shapes
:<<: e

_output_shapes
:<:,f(
&
_output_shapes
:<x: g

_output_shapes
:x:,h(
&
_output_shapes
:xx: i

_output_shapes
:x:-j)
'
_output_shapes
:x�:!k

_output_shapes	
:�:.l*
(
_output_shapes
:��:!m

_output_shapes	
:�:.n*
(
_output_shapes
:��:!o

_output_shapes	
:�:.p*
(
_output_shapes
:��:!q

_output_shapes	
:�:-r)
'
_output_shapes
:x�: s

_output_shapes
:x:,t(
&
_output_shapes
:xx: u

_output_shapes
:x:,v(
&
_output_shapes
:xx: w

_output_shapes
:x:,x(
&
_output_shapes
:<x: y

_output_shapes
:<:,z(
&
_output_shapes
:<<: {

_output_shapes
:<:,|(
&
_output_shapes
:<<: }

_output_shapes
:<:,~(
&
_output_shapes
:<: 

_output_shapes
::-�(
&
_output_shapes
::!�

_output_shapes
::-�(
&
_output_shapes
::!�

_output_shapes
::-�(
&
_output_shapes
::!�

_output_shapes
::-�(
&
_output_shapes
::!�

_output_shapes
::-�(
&
_output_shapes
:<:!�

_output_shapes
:<:-�(
&
_output_shapes
:<<:!�

_output_shapes
:<:-�(
&
_output_shapes
:<x:!�

_output_shapes
:x:-�(
&
_output_shapes
:xx:!�

_output_shapes
:x:.�)
'
_output_shapes
:x�:"�

_output_shapes	
:�:/�*
(
_output_shapes
:��:"�

_output_shapes	
:�:/�*
(
_output_shapes
:��:"�

_output_shapes	
:�:/�*
(
_output_shapes
:��:"�

_output_shapes	
:�:.�)
'
_output_shapes
:x�:!�

_output_shapes
:x:-�(
&
_output_shapes
:xx:!�

_output_shapes
:x:-�(
&
_output_shapes
:xx:!�

_output_shapes
:x:-�(
&
_output_shapes
:<x:!�

_output_shapes
:<:-�(
&
_output_shapes
:<<:!�

_output_shapes
:<:-�(
&
_output_shapes
:<<:!�

_output_shapes
:<:-�(
&
_output_shapes
:<:!�

_output_shapes
::�

_output_shapes
: 
�
O
3__inference_average_pooling2d_1_layer_call_fn_55970

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4������������������������������������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *W
fRRP
N__inference_average_pooling2d_1_layer_call_and_return_conditional_losses_53032�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
�
C__inference_conv2d_6_layer_call_and_return_conditional_losses_55931

inputs8
conv2d_readvariableop_resource:<x-
biasadd_readvariableop_resource:x
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:<x*
dtype0r
Conv2D/CastCastConv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*&
_output_shapes
:<x�
Conv2DConv2DinputsConv2D/Cast:y:0*
T0*/
_output_shapes
:���������xxx*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:x*
dtype0h
BiasAdd/CastCastBiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:xo
BiasAddBiasAddConv2D:output:0BiasAdd/Cast:y:0*
T0*/
_output_shapes
:���������xxxH
mul/xConst*
_output_shapes
: *
dtype0*
value
B j�pf
mulMulmul/x:output:0BiasAdd:output:0*
T0*/
_output_shapes
:���������xxxI
Pow/yConst*
_output_shapes
: *
dtype0*
valueB	 j��f
PowPowBiasAdd:output:0Pow/y:output:0*
T0*/
_output_shapes
:���������xxxJ
mul_1/xConst*
_output_shapes
: *
dtype0*
value
B j�Sa
mul_1Mulmul_1/x:output:0Pow:z:0*
T0*/
_output_shapes
:���������xxxc
addAddV2BiasAdd:output:0	mul_1:z:0*
T0*/
_output_shapes
:���������xxxJ
mul_2/xConst*
_output_shapes
: *
dtype0*
value
B j�ta
mul_2Mulmul_2/x:output:0add:z:0*
T0*/
_output_shapes
:���������xxxQ
TanhTanh	mul_2:z:0*
T0*/
_output_shapes
:���������xxxJ
add_1/xConst*
_output_shapes
: *
dtype0*
value
B j�xd
add_1AddV2add_1/x:output:0Tanh:y:0*
T0*/
_output_shapes
:���������xxxZ
mul_3Mulmul:z:0	add_1:z:0*
T0*/
_output_shapes
:���������xxx`
IdentityIdentity	mul_3:z:0^NoOp*
T0*/
_output_shapes
:���������xxxw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������xx<: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������xx<
 
_user_specified_nameinputs
�
�
D__inference_conv2d_15_layer_call_and_return_conditional_losses_56385

inputs8
conv2d_readvariableop_resource:<<-
biasadd_readvariableop_resource:<
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:<<*
dtype0r
Conv2D/CastCastConv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*&
_output_shapes
:<<�
Conv2DConv2DinputsConv2D/Cast:y:0*
T0*1
_output_shapes
:�����������<*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:<*
dtype0h
BiasAdd/CastCastBiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:<q
BiasAddBiasAddConv2D:output:0BiasAdd/Cast:y:0*
T0*1
_output_shapes
:�����������<H
mul/xConst*
_output_shapes
: *
dtype0*
value
B j�ph
mulMulmul/x:output:0BiasAdd:output:0*
T0*1
_output_shapes
:�����������<I
Pow/yConst*
_output_shapes
: *
dtype0*
valueB	 j��h
PowPowBiasAdd:output:0Pow/y:output:0*
T0*1
_output_shapes
:�����������<J
mul_1/xConst*
_output_shapes
: *
dtype0*
value
B j�Sc
mul_1Mulmul_1/x:output:0Pow:z:0*
T0*1
_output_shapes
:�����������<e
addAddV2BiasAdd:output:0	mul_1:z:0*
T0*1
_output_shapes
:�����������<J
mul_2/xConst*
_output_shapes
: *
dtype0*
value
B j�tc
mul_2Mulmul_2/x:output:0add:z:0*
T0*1
_output_shapes
:�����������<S
TanhTanh	mul_2:z:0*
T0*1
_output_shapes
:�����������<J
add_1/xConst*
_output_shapes
: *
dtype0*
value
B j�xf
add_1AddV2add_1/x:output:0Tanh:y:0*
T0*1
_output_shapes
:�����������<\
mul_3Mulmul:z:0	add_1:z:0*
T0*1
_output_shapes
:�����������<b
IdentityIdentity	mul_3:z:0^NoOp*
T0*1
_output_shapes
:�����������<w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:�����������<: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:�����������<
 
_user_specified_nameinputs
�
�

%__inference_model_layer_call_fn_53802
input_1!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:#
	unknown_3:
	unknown_4:#
	unknown_5:
	unknown_6:#
	unknown_7:<
	unknown_8:<#
	unknown_9:<<

unknown_10:<$

unknown_11:<x

unknown_12:x$

unknown_13:xx

unknown_14:x%

unknown_15:x�

unknown_16:	�&

unknown_17:��

unknown_18:	�&

unknown_19:��

unknown_20:	�&

unknown_21:��

unknown_22:	�%

unknown_23:x�

unknown_24:x$

unknown_25:xx

unknown_26:x$

unknown_27:xx

unknown_28:x$

unknown_29:<x

unknown_30:<$

unknown_31:<<

unknown_32:<$

unknown_33:<<

unknown_34:<$

unknown_35:<

unknown_36:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36*2
Tin+
)2'*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������*H
_read_only_resource_inputs*
(&	
 !"#$%&*0
config_proto 

CPU

GPU2*0J 8� *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_53723y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:�����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*|
_input_shapesk
i:�����������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:�����������
!
_user_specified_name	input_1
�
�
C__inference_conv2d_1_layer_call_and_return_conditional_losses_55738

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0r
Conv2D/CastCastConv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*&
_output_shapes
:�
Conv2DConv2DinputsConv2D/Cast:y:0*
T0*1
_output_shapes
:�����������*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0h
BiasAdd/CastCastBiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:q
BiasAddBiasAddConv2D:output:0BiasAdd/Cast:y:0*
T0*1
_output_shapes
:�����������H
mul/xConst*
_output_shapes
: *
dtype0*
value
B j�ph
mulMulmul/x:output:0BiasAdd:output:0*
T0*1
_output_shapes
:�����������I
Pow/yConst*
_output_shapes
: *
dtype0*
valueB	 j��h
PowPowBiasAdd:output:0Pow/y:output:0*
T0*1
_output_shapes
:�����������J
mul_1/xConst*
_output_shapes
: *
dtype0*
value
B j�Sc
mul_1Mulmul_1/x:output:0Pow:z:0*
T0*1
_output_shapes
:�����������e
addAddV2BiasAdd:output:0	mul_1:z:0*
T0*1
_output_shapes
:�����������J
mul_2/xConst*
_output_shapes
: *
dtype0*
value
B j�tc
mul_2Mulmul_2/x:output:0add:z:0*
T0*1
_output_shapes
:�����������S
TanhTanh	mul_2:z:0*
T0*1
_output_shapes
:�����������J
add_1/xConst*
_output_shapes
: *
dtype0*
value
B j�xf
add_1AddV2add_1/x:output:0Tanh:y:0*
T0*1
_output_shapes
:�����������\
mul_3Mulmul:z:0	add_1:z:0*
T0*1
_output_shapes
:�����������b
IdentityIdentity	mul_3:z:0^NoOp*
T0*1
_output_shapes
:�����������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:�����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
�
C__inference_conv2d_7_layer_call_and_return_conditional_losses_55965

inputs8
conv2d_readvariableop_resource:xx-
biasadd_readvariableop_resource:x
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:xx*
dtype0r
Conv2D/CastCastConv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*&
_output_shapes
:xx�
Conv2DConv2DinputsConv2D/Cast:y:0*
T0*/
_output_shapes
:���������xxx*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:x*
dtype0h
BiasAdd/CastCastBiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:xo
BiasAddBiasAddConv2D:output:0BiasAdd/Cast:y:0*
T0*/
_output_shapes
:���������xxxH
mul/xConst*
_output_shapes
: *
dtype0*
value
B j�pf
mulMulmul/x:output:0BiasAdd:output:0*
T0*/
_output_shapes
:���������xxxI
Pow/yConst*
_output_shapes
: *
dtype0*
valueB	 j��f
PowPowBiasAdd:output:0Pow/y:output:0*
T0*/
_output_shapes
:���������xxxJ
mul_1/xConst*
_output_shapes
: *
dtype0*
value
B j�Sa
mul_1Mulmul_1/x:output:0Pow:z:0*
T0*/
_output_shapes
:���������xxxc
addAddV2BiasAdd:output:0	mul_1:z:0*
T0*/
_output_shapes
:���������xxxJ
mul_2/xConst*
_output_shapes
: *
dtype0*
value
B j�ta
mul_2Mulmul_2/x:output:0add:z:0*
T0*/
_output_shapes
:���������xxxQ
TanhTanh	mul_2:z:0*
T0*/
_output_shapes
:���������xxxJ
add_1/xConst*
_output_shapes
: *
dtype0*
value
B j�xd
add_1AddV2add_1/x:output:0Tanh:y:0*
T0*/
_output_shapes
:���������xxxZ
mul_3Mulmul:z:0	add_1:z:0*
T0*/
_output_shapes
:���������xxx`
IdentityIdentity	mul_3:z:0^NoOp*
T0*/
_output_shapes
:���������xxxw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������xxx: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������xxx
 
_user_specified_nameinputs
�,
�
K__inference_conv2d_transpose_layer_call_and_return_conditional_losses_56168

inputsC
(conv2d_transpose_readvariableop_resource:x�-
biasadd_readvariableop_resource:x
identity��BiasAdd/ReadVariableOp�conv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B :xy
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*'
_output_shapes
:x�*
dtype0�
conv2d_transpose/CastCast'conv2d_transpose/ReadVariableOp:value:0*

DstT0*

SrcT0*'
_output_shapes
:x��
conv2d_transposeConv2DBackpropInputstack:output:0conv2d_transpose/Cast:y:0inputs*
T0*A
_output_shapes/
-:+���������������������������x*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:x*
dtype0h
BiasAdd/CastCastBiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:x�
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/Cast:y:0*
T0*A
_output_shapes/
-:+���������������������������xJ
mul_2/xConst*
_output_shapes
: *
dtype0*
value
B j�p|
mul_2Mulmul_2/x:output:0BiasAdd:output:0*
T0*A
_output_shapes/
-:+���������������������������xI
Pow/yConst*
_output_shapes
: *
dtype0*
valueB	 j��x
PowPowBiasAdd:output:0Pow/y:output:0*
T0*A
_output_shapes/
-:+���������������������������xJ
mul_3/xConst*
_output_shapes
: *
dtype0*
value
B j�Ss
mul_3Mulmul_3/x:output:0Pow:z:0*
T0*A
_output_shapes/
-:+���������������������������xu
addAddV2BiasAdd:output:0	mul_3:z:0*
T0*A
_output_shapes/
-:+���������������������������xJ
mul_4/xConst*
_output_shapes
: *
dtype0*
value
B j�ts
mul_4Mulmul_4/x:output:0add:z:0*
T0*A
_output_shapes/
-:+���������������������������xc
TanhTanh	mul_4:z:0*
T0*A
_output_shapes/
-:+���������������������������xJ
add_1/xConst*
_output_shapes
: *
dtype0*
value
B j�xv
add_1AddV2add_1/x:output:0Tanh:y:0*
T0*A
_output_shapes/
-:+���������������������������xn
mul_5Mul	mul_2:z:0	add_1:z:0*
T0*A
_output_shapes/
-:+���������������������������xr
IdentityIdentity	mul_5:z:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������x�
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,����������������������������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
D__inference_conv2d_15_layer_call_and_return_conditional_losses_53689

inputs8
conv2d_readvariableop_resource:<<-
biasadd_readvariableop_resource:<
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:<<*
dtype0r
Conv2D/CastCastConv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*&
_output_shapes
:<<�
Conv2DConv2DinputsConv2D/Cast:y:0*
T0*1
_output_shapes
:�����������<*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:<*
dtype0h
BiasAdd/CastCastBiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:<q
BiasAddBiasAddConv2D:output:0BiasAdd/Cast:y:0*
T0*1
_output_shapes
:�����������<H
mul/xConst*
_output_shapes
: *
dtype0*
value
B j�ph
mulMulmul/x:output:0BiasAdd:output:0*
T0*1
_output_shapes
:�����������<I
Pow/yConst*
_output_shapes
: *
dtype0*
valueB	 j��h
PowPowBiasAdd:output:0Pow/y:output:0*
T0*1
_output_shapes
:�����������<J
mul_1/xConst*
_output_shapes
: *
dtype0*
value
B j�Sc
mul_1Mulmul_1/x:output:0Pow:z:0*
T0*1
_output_shapes
:�����������<e
addAddV2BiasAdd:output:0	mul_1:z:0*
T0*1
_output_shapes
:�����������<J
mul_2/xConst*
_output_shapes
: *
dtype0*
value
B j�tc
mul_2Mulmul_2/x:output:0add:z:0*
T0*1
_output_shapes
:�����������<S
TanhTanh	mul_2:z:0*
T0*1
_output_shapes
:�����������<J
add_1/xConst*
_output_shapes
: *
dtype0*
value
B j�xf
add_1AddV2add_1/x:output:0Tanh:y:0*
T0*1
_output_shapes
:�����������<\
mul_3Mulmul:z:0	add_1:z:0*
T0*1
_output_shapes
:�����������<b
IdentityIdentity	mul_3:z:0^NoOp*
T0*1
_output_shapes
:�����������<w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:�����������<: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:�����������<
 
_user_specified_nameinputs
�
�
D__inference_conv2d_16_layer_call_and_return_conditional_losses_56406

inputs8
conv2d_readvariableop_resource:<-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:<*
dtype0r
Conv2D/CastCastConv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*&
_output_shapes
:<�
Conv2DConv2DinputsConv2D/Cast:y:0*
T0*1
_output_shapes
:�����������*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0h
BiasAdd/CastCastBiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:q
BiasAddBiasAddConv2D:output:0BiasAdd/Cast:y:0*
T0*1
_output_shapes
:�����������i
IdentityIdentityBiasAdd:output:0^NoOp*
T0*1
_output_shapes
:�����������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:�����������<: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:�����������<
 
_user_specified_nameinputs
�,
�
K__inference_conv2d_transpose_layer_call_and_return_conditional_losses_53087

inputsC
(conv2d_transpose_readvariableop_resource:x�-
biasadd_readvariableop_resource:x
identity��BiasAdd/ReadVariableOp�conv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B :xy
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*'
_output_shapes
:x�*
dtype0�
conv2d_transpose/CastCast'conv2d_transpose/ReadVariableOp:value:0*

DstT0*

SrcT0*'
_output_shapes
:x��
conv2d_transposeConv2DBackpropInputstack:output:0conv2d_transpose/Cast:y:0inputs*
T0*A
_output_shapes/
-:+���������������������������x*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:x*
dtype0h
BiasAdd/CastCastBiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:x�
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/Cast:y:0*
T0*A
_output_shapes/
-:+���������������������������xJ
mul_2/xConst*
_output_shapes
: *
dtype0*
value
B j�p|
mul_2Mulmul_2/x:output:0BiasAdd:output:0*
T0*A
_output_shapes/
-:+���������������������������xI
Pow/yConst*
_output_shapes
: *
dtype0*
valueB	 j��x
PowPowBiasAdd:output:0Pow/y:output:0*
T0*A
_output_shapes/
-:+���������������������������xJ
mul_3/xConst*
_output_shapes
: *
dtype0*
value
B j�Ss
mul_3Mulmul_3/x:output:0Pow:z:0*
T0*A
_output_shapes/
-:+���������������������������xu
addAddV2BiasAdd:output:0	mul_3:z:0*
T0*A
_output_shapes/
-:+���������������������������xJ
mul_4/xConst*
_output_shapes
: *
dtype0*
value
B j�ts
mul_4Mulmul_4/x:output:0add:z:0*
T0*A
_output_shapes/
-:+���������������������������xc
TanhTanh	mul_4:z:0*
T0*A
_output_shapes/
-:+���������������������������xJ
add_1/xConst*
_output_shapes
: *
dtype0*
value
B j�xv
add_1AddV2add_1/x:output:0Tanh:y:0*
T0*A
_output_shapes/
-:+���������������������������xn
mul_5Mul	mul_2:z:0	add_1:z:0*
T0*A
_output_shapes/
-:+���������������������������xr
IdentityIdentity	mul_5:z:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������x�
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,����������������������������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�,
�
M__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_53146

inputsB
(conv2d_transpose_readvariableop_resource:<x-
biasadd_readvariableop_resource:<
identity��BiasAdd/ReadVariableOp�conv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B :<y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:<x*
dtype0�
conv2d_transpose/CastCast'conv2d_transpose/ReadVariableOp:value:0*

DstT0*

SrcT0*&
_output_shapes
:<x�
conv2d_transposeConv2DBackpropInputstack:output:0conv2d_transpose/Cast:y:0inputs*
T0*A
_output_shapes/
-:+���������������������������<*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:<*
dtype0h
BiasAdd/CastCastBiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:<�
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/Cast:y:0*
T0*A
_output_shapes/
-:+���������������������������<J
mul_2/xConst*
_output_shapes
: *
dtype0*
value
B j�p|
mul_2Mulmul_2/x:output:0BiasAdd:output:0*
T0*A
_output_shapes/
-:+���������������������������<I
Pow/yConst*
_output_shapes
: *
dtype0*
valueB	 j��x
PowPowBiasAdd:output:0Pow/y:output:0*
T0*A
_output_shapes/
-:+���������������������������<J
mul_3/xConst*
_output_shapes
: *
dtype0*
value
B j�Ss
mul_3Mulmul_3/x:output:0Pow:z:0*
T0*A
_output_shapes/
-:+���������������������������<u
addAddV2BiasAdd:output:0	mul_3:z:0*
T0*A
_output_shapes/
-:+���������������������������<J
mul_4/xConst*
_output_shapes
: *
dtype0*
value
B j�ts
mul_4Mulmul_4/x:output:0add:z:0*
T0*A
_output_shapes/
-:+���������������������������<c
TanhTanh	mul_4:z:0*
T0*A
_output_shapes/
-:+���������������������������<J
add_1/xConst*
_output_shapes
: *
dtype0*
value
B j�xv
add_1AddV2add_1/x:output:0Tanh:y:0*
T0*A
_output_shapes/
-:+���������������������������<n
mul_5Mul	mul_2:z:0	add_1:z:0*
T0*A
_output_shapes/
-:+���������������������������<r
IdentityIdentity	mul_5:z:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������<�
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+���������������������������x: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+���������������������������x
 
_user_specified_nameinputs
�
�
D__inference_conv2d_11_layer_call_and_return_conditional_losses_56111

inputs:
conv2d_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0t
Conv2D/CastCastConv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*(
_output_shapes
:���
Conv2DConv2DinputsConv2D/Cast:y:0*
T0*0
_output_shapes
:���������<<�*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0i
BiasAdd/CastCastBiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes	
:�p
BiasAddBiasAddConv2D:output:0BiasAdd/Cast:y:0*
T0*0
_output_shapes
:���������<<�H
mul/xConst*
_output_shapes
: *
dtype0*
value
B j�pg
mulMulmul/x:output:0BiasAdd:output:0*
T0*0
_output_shapes
:���������<<�I
Pow/yConst*
_output_shapes
: *
dtype0*
valueB	 j��g
PowPowBiasAdd:output:0Pow/y:output:0*
T0*0
_output_shapes
:���������<<�J
mul_1/xConst*
_output_shapes
: *
dtype0*
value
B j�Sb
mul_1Mulmul_1/x:output:0Pow:z:0*
T0*0
_output_shapes
:���������<<�d
addAddV2BiasAdd:output:0	mul_1:z:0*
T0*0
_output_shapes
:���������<<�J
mul_2/xConst*
_output_shapes
: *
dtype0*
value
B j�tb
mul_2Mulmul_2/x:output:0add:z:0*
T0*0
_output_shapes
:���������<<�R
TanhTanh	mul_2:z:0*
T0*0
_output_shapes
:���������<<�J
add_1/xConst*
_output_shapes
: *
dtype0*
value
B j�xe
add_1AddV2add_1/x:output:0Tanh:y:0*
T0*0
_output_shapes
:���������<<�[
mul_3Mulmul:z:0	add_1:z:0*
T0*0
_output_shapes
:���������<<�a
IdentityIdentity	mul_3:z:0^NoOp*
T0*0
_output_shapes
:���������<<�w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������<<�: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:���������<<�
 
_user_specified_nameinputs
�
�
C__inference_conv2d_3_layer_call_and_return_conditional_losses_53279

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0r
Conv2D/CastCastConv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*&
_output_shapes
:�
Conv2DConv2DinputsConv2D/Cast:y:0*
T0*1
_output_shapes
:�����������*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0h
BiasAdd/CastCastBiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:q
BiasAddBiasAddConv2D:output:0BiasAdd/Cast:y:0*
T0*1
_output_shapes
:�����������H
mul/xConst*
_output_shapes
: *
dtype0*
value
B j�ph
mulMulmul/x:output:0BiasAdd:output:0*
T0*1
_output_shapes
:�����������I
Pow/yConst*
_output_shapes
: *
dtype0*
valueB	 j��h
PowPowBiasAdd:output:0Pow/y:output:0*
T0*1
_output_shapes
:�����������J
mul_1/xConst*
_output_shapes
: *
dtype0*
value
B j�Sc
mul_1Mulmul_1/x:output:0Pow:z:0*
T0*1
_output_shapes
:�����������e
addAddV2BiasAdd:output:0	mul_1:z:0*
T0*1
_output_shapes
:�����������J
mul_2/xConst*
_output_shapes
: *
dtype0*
value
B j�tc
mul_2Mulmul_2/x:output:0add:z:0*
T0*1
_output_shapes
:�����������S
TanhTanh	mul_2:z:0*
T0*1
_output_shapes
:�����������J
add_1/xConst*
_output_shapes
: *
dtype0*
value
B j�xf
add_1AddV2add_1/x:output:0Tanh:y:0*
T0*1
_output_shapes
:�����������\
mul_3Mulmul:z:0	add_1:z:0*
T0*1
_output_shapes
:�����������b
IdentityIdentity	mul_3:z:0^NoOp*
T0*1
_output_shapes
:�����������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:�����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
�
D__inference_conv2d_10_layer_call_and_return_conditional_losses_53508

inputs:
conv2d_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0t
Conv2D/CastCastConv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*(
_output_shapes
:���
Conv2DConv2DinputsConv2D/Cast:y:0*
T0*0
_output_shapes
:���������<<�*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0i
BiasAdd/CastCastBiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes	
:�p
BiasAddBiasAddConv2D:output:0BiasAdd/Cast:y:0*
T0*0
_output_shapes
:���������<<�H
mul/xConst*
_output_shapes
: *
dtype0*
value
B j�pg
mulMulmul/x:output:0BiasAdd:output:0*
T0*0
_output_shapes
:���������<<�I
Pow/yConst*
_output_shapes
: *
dtype0*
valueB	 j��g
PowPowBiasAdd:output:0Pow/y:output:0*
T0*0
_output_shapes
:���������<<�J
mul_1/xConst*
_output_shapes
: *
dtype0*
value
B j�Sb
mul_1Mulmul_1/x:output:0Pow:z:0*
T0*0
_output_shapes
:���������<<�d
addAddV2BiasAdd:output:0	mul_1:z:0*
T0*0
_output_shapes
:���������<<�J
mul_2/xConst*
_output_shapes
: *
dtype0*
value
B j�tb
mul_2Mulmul_2/x:output:0add:z:0*
T0*0
_output_shapes
:���������<<�R
TanhTanh	mul_2:z:0*
T0*0
_output_shapes
:���������<<�J
add_1/xConst*
_output_shapes
: *
dtype0*
value
B j�xe
add_1AddV2add_1/x:output:0Tanh:y:0*
T0*0
_output_shapes
:���������<<�[
mul_3Mulmul:z:0	add_1:z:0*
T0*0
_output_shapes
:���������<<�a
IdentityIdentity	mul_3:z:0^NoOp*
T0*0
_output_shapes
:���������<<�w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������<<�: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:���������<<�
 
_user_specified_nameinputs
�
h
L__inference_average_pooling2d_layer_call_and_return_conditional_losses_55897

inputs
identity�
AvgPoolAvgPoolinputs*
T0*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingSAME*
strides
{
IdentityIdentityAvgPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
�
)__inference_conv2d_15_layer_call_fn_56360

inputs!
unknown:<<
	unknown_0:<
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������<*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv2d_15_layer_call_and_return_conditional_losses_53689y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:�����������<`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:�����������<: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:�����������<
 
_user_specified_nameinputs
�w
�
@__inference_model_layer_call_and_return_conditional_losses_54191

inputs&
conv2d_54087:
conv2d_54089:(
conv2d_1_54092:
conv2d_1_54094:(
conv2d_2_54097:
conv2d_2_54099:(
conv2d_3_54102:
conv2d_3_54104:(
conv2d_4_54109:<
conv2d_4_54111:<(
conv2d_5_54114:<<
conv2d_5_54116:<(
conv2d_6_54120:<x
conv2d_6_54122:x(
conv2d_7_54125:xx
conv2d_7_54127:x)
conv2d_8_54131:x�
conv2d_8_54133:	�*
conv2d_9_54136:��
conv2d_9_54138:	�+
conv2d_10_54141:��
conv2d_10_54143:	�+
conv2d_11_54146:��
conv2d_11_54148:	�1
conv2d_transpose_54151:x�$
conv2d_transpose_54153:x)
conv2d_12_54157:xx
conv2d_12_54159:x)
conv2d_13_54162:xx
conv2d_13_54164:x2
conv2d_transpose_1_54167:<x&
conv2d_transpose_1_54169:<)
conv2d_14_54173:<<
conv2d_14_54175:<)
conv2d_15_54178:<<
conv2d_15_54180:<)
conv2d_16_54183:<
conv2d_16_54185:
identity��conv2d/StatefulPartitionedCall� conv2d_1/StatefulPartitionedCall�!conv2d_10/StatefulPartitionedCall�!conv2d_11/StatefulPartitionedCall�!conv2d_12/StatefulPartitionedCall�!conv2d_13/StatefulPartitionedCall�!conv2d_14/StatefulPartitionedCall�!conv2d_15/StatefulPartitionedCall�!conv2d_16/StatefulPartitionedCall� conv2d_2/StatefulPartitionedCall� conv2d_3/StatefulPartitionedCall� conv2d_4/StatefulPartitionedCall� conv2d_5/StatefulPartitionedCall� conv2d_6/StatefulPartitionedCall� conv2d_7/StatefulPartitionedCall� conv2d_8/StatefulPartitionedCall� conv2d_9/StatefulPartitionedCall�(conv2d_transpose/StatefulPartitionedCall�*conv2d_transpose_1/StatefulPartitionedCallf
conv2d/CastCastinputs*

DstT0*

SrcT0*1
_output_shapes
:������������
conv2d/StatefulPartitionedCallStatefulPartitionedCallconv2d/Cast:y:0conv2d_54087conv2d_54089*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_53186�
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0conv2d_1_54092conv2d_1_54094*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_53217�
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0conv2d_2_54097conv2d_2_54099*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_conv2d_2_layer_call_and_return_conditional_losses_53248�
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0conv2d_3_54102conv2d_3_54104*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_conv2d_3_layer_call_and_return_conditional_losses_53279k
concatenate/CastCastinputs*

DstT0*

SrcT0*1
_output_shapes
:������������
concatenate/PartitionedCallPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0concatenate/Cast:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_concatenate_layer_call_and_return_conditional_losses_53293�
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0conv2d_4_54109conv2d_4_54111*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������<*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_conv2d_4_layer_call_and_return_conditional_losses_53320�
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0conv2d_5_54114conv2d_5_54116*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������<*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_conv2d_5_layer_call_and_return_conditional_losses_53351�
!average_pooling2d/PartitionedCallPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������xx<* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_average_pooling2d_layer_call_and_return_conditional_losses_53020�
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCall*average_pooling2d/PartitionedCall:output:0conv2d_6_54120conv2d_6_54122*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������xxx*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_conv2d_6_layer_call_and_return_conditional_losses_53383�
 conv2d_7/StatefulPartitionedCallStatefulPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0conv2d_7_54125conv2d_7_54127*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������xxx*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_conv2d_7_layer_call_and_return_conditional_losses_53414�
#average_pooling2d_1/PartitionedCallPartitionedCall)conv2d_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������<<x* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *W
fRRP
N__inference_average_pooling2d_1_layer_call_and_return_conditional_losses_53032�
 conv2d_8/StatefulPartitionedCallStatefulPartitionedCall,average_pooling2d_1/PartitionedCall:output:0conv2d_8_54131conv2d_8_54133*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������<<�*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_conv2d_8_layer_call_and_return_conditional_losses_53446�
 conv2d_9/StatefulPartitionedCallStatefulPartitionedCall)conv2d_8/StatefulPartitionedCall:output:0conv2d_9_54136conv2d_9_54138*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������<<�*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_conv2d_9_layer_call_and_return_conditional_losses_53477�
!conv2d_10/StatefulPartitionedCallStatefulPartitionedCall)conv2d_9/StatefulPartitionedCall:output:0conv2d_10_54141conv2d_10_54143*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������<<�*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv2d_10_layer_call_and_return_conditional_losses_53508�
!conv2d_11/StatefulPartitionedCallStatefulPartitionedCall*conv2d_10/StatefulPartitionedCall:output:0conv2d_11_54146conv2d_11_54148*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������<<�*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv2d_11_layer_call_and_return_conditional_losses_53539�
(conv2d_transpose/StatefulPartitionedCallStatefulPartitionedCall*conv2d_11/StatefulPartitionedCall:output:0conv2d_transpose_54151conv2d_transpose_54153*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������xxx*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *T
fORM
K__inference_conv2d_transpose_layer_call_and_return_conditional_losses_53087�
add/PartitionedCallPartitionedCall1conv2d_transpose/StatefulPartitionedCall:output:0)conv2d_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������xxx* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *G
fBR@
>__inference_add_layer_call_and_return_conditional_losses_53556�
!conv2d_12/StatefulPartitionedCallStatefulPartitionedCalladd/PartitionedCall:output:0conv2d_12_54157conv2d_12_54159*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������xxx*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv2d_12_layer_call_and_return_conditional_losses_53583�
!conv2d_13/StatefulPartitionedCallStatefulPartitionedCall*conv2d_12/StatefulPartitionedCall:output:0conv2d_13_54162conv2d_13_54164*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������xxx*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv2d_13_layer_call_and_return_conditional_losses_53614�
*conv2d_transpose_1/StatefulPartitionedCallStatefulPartitionedCall*conv2d_13/StatefulPartitionedCall:output:0conv2d_transpose_1_54167conv2d_transpose_1_54169*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������<*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *V
fQRO
M__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_53146�
add_1/PartitionedCallPartitionedCall3conv2d_transpose_1/StatefulPartitionedCall:output:0)conv2d_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������<* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *I
fDRB
@__inference_add_1_layer_call_and_return_conditional_losses_53631�
!conv2d_14/StatefulPartitionedCallStatefulPartitionedCalladd_1/PartitionedCall:output:0conv2d_14_54173conv2d_14_54175*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������<*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv2d_14_layer_call_and_return_conditional_losses_53658�
!conv2d_15/StatefulPartitionedCallStatefulPartitionedCall*conv2d_14/StatefulPartitionedCall:output:0conv2d_15_54178conv2d_15_54180*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������<*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv2d_15_layer_call_and_return_conditional_losses_53689�
!conv2d_16/StatefulPartitionedCallStatefulPartitionedCall*conv2d_15/StatefulPartitionedCall:output:0conv2d_16_54183conv2d_16_54185*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv2d_16_layer_call_and_return_conditional_losses_53707�

add_2/CastCast*conv2d_16/StatefulPartitionedCall:output:0*

DstT0*

SrcT0*1
_output_shapes
:������������
add_2/PartitionedCallPartitionedCalladd_2/Cast:y:0inputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *I
fDRB
@__inference_add_2_layer_call_and_return_conditional_losses_53720w
IdentityIdentityadd_2/PartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:������������
NoOpNoOp^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall"^conv2d_10/StatefulPartitionedCall"^conv2d_11/StatefulPartitionedCall"^conv2d_12/StatefulPartitionedCall"^conv2d_13/StatefulPartitionedCall"^conv2d_14/StatefulPartitionedCall"^conv2d_15/StatefulPartitionedCall"^conv2d_16/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall!^conv2d_5/StatefulPartitionedCall!^conv2d_6/StatefulPartitionedCall!^conv2d_7/StatefulPartitionedCall!^conv2d_8/StatefulPartitionedCall!^conv2d_9/StatefulPartitionedCall)^conv2d_transpose/StatefulPartitionedCall+^conv2d_transpose_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*|
_input_shapesk
i:�����������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2F
!conv2d_10/StatefulPartitionedCall!conv2d_10/StatefulPartitionedCall2F
!conv2d_11/StatefulPartitionedCall!conv2d_11/StatefulPartitionedCall2F
!conv2d_12/StatefulPartitionedCall!conv2d_12/StatefulPartitionedCall2F
!conv2d_13/StatefulPartitionedCall!conv2d_13/StatefulPartitionedCall2F
!conv2d_14/StatefulPartitionedCall!conv2d_14/StatefulPartitionedCall2F
!conv2d_15/StatefulPartitionedCall!conv2d_15/StatefulPartitionedCall2F
!conv2d_16/StatefulPartitionedCall!conv2d_16/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall2D
 conv2d_7/StatefulPartitionedCall conv2d_7/StatefulPartitionedCall2D
 conv2d_8/StatefulPartitionedCall conv2d_8/StatefulPartitionedCall2D
 conv2d_9/StatefulPartitionedCall conv2d_9/StatefulPartitionedCall2T
(conv2d_transpose/StatefulPartitionedCall(conv2d_transpose/StatefulPartitionedCall2X
*conv2d_transpose_1/StatefulPartitionedCall*conv2d_transpose_1/StatefulPartitionedCall:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
�
(__inference_conv2d_1_layer_call_fn_55713

inputs!
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_53217y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:�����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:�����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
�
D__inference_conv2d_14_layer_call_and_return_conditional_losses_56351

inputs8
conv2d_readvariableop_resource:<<-
biasadd_readvariableop_resource:<
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:<<*
dtype0r
Conv2D/CastCastConv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*&
_output_shapes
:<<�
Conv2DConv2DinputsConv2D/Cast:y:0*
T0*1
_output_shapes
:�����������<*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:<*
dtype0h
BiasAdd/CastCastBiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:<q
BiasAddBiasAddConv2D:output:0BiasAdd/Cast:y:0*
T0*1
_output_shapes
:�����������<H
mul/xConst*
_output_shapes
: *
dtype0*
value
B j�ph
mulMulmul/x:output:0BiasAdd:output:0*
T0*1
_output_shapes
:�����������<I
Pow/yConst*
_output_shapes
: *
dtype0*
valueB	 j��h
PowPowBiasAdd:output:0Pow/y:output:0*
T0*1
_output_shapes
:�����������<J
mul_1/xConst*
_output_shapes
: *
dtype0*
value
B j�Sc
mul_1Mulmul_1/x:output:0Pow:z:0*
T0*1
_output_shapes
:�����������<e
addAddV2BiasAdd:output:0	mul_1:z:0*
T0*1
_output_shapes
:�����������<J
mul_2/xConst*
_output_shapes
: *
dtype0*
value
B j�tc
mul_2Mulmul_2/x:output:0add:z:0*
T0*1
_output_shapes
:�����������<S
TanhTanh	mul_2:z:0*
T0*1
_output_shapes
:�����������<J
add_1/xConst*
_output_shapes
: *
dtype0*
value
B j�xf
add_1AddV2add_1/x:output:0Tanh:y:0*
T0*1
_output_shapes
:�����������<\
mul_3Mulmul:z:0	add_1:z:0*
T0*1
_output_shapes
:�����������<b
IdentityIdentity	mul_3:z:0^NoOp*
T0*1
_output_shapes
:�����������<w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:�����������<: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:�����������<
 
_user_specified_nameinputs
�
h
>__inference_add_layer_call_and_return_conditional_losses_53556

inputs
inputs_1
identityX
addAddV2inputsinputs_1*
T0*/
_output_shapes
:���������xxxW
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:���������xxx"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:���������xxx:���������xxx:W S
/
_output_shapes
:���������xxx
 
_user_specified_nameinputs:WS
/
_output_shapes
:���������xxx
 
_user_specified_nameinputs
�
�
D__inference_conv2d_14_layer_call_and_return_conditional_losses_53658

inputs8
conv2d_readvariableop_resource:<<-
biasadd_readvariableop_resource:<
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:<<*
dtype0r
Conv2D/CastCastConv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*&
_output_shapes
:<<�
Conv2DConv2DinputsConv2D/Cast:y:0*
T0*1
_output_shapes
:�����������<*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:<*
dtype0h
BiasAdd/CastCastBiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:<q
BiasAddBiasAddConv2D:output:0BiasAdd/Cast:y:0*
T0*1
_output_shapes
:�����������<H
mul/xConst*
_output_shapes
: *
dtype0*
value
B j�ph
mulMulmul/x:output:0BiasAdd:output:0*
T0*1
_output_shapes
:�����������<I
Pow/yConst*
_output_shapes
: *
dtype0*
valueB	 j��h
PowPowBiasAdd:output:0Pow/y:output:0*
T0*1
_output_shapes
:�����������<J
mul_1/xConst*
_output_shapes
: *
dtype0*
value
B j�Sc
mul_1Mulmul_1/x:output:0Pow:z:0*
T0*1
_output_shapes
:�����������<e
addAddV2BiasAdd:output:0	mul_1:z:0*
T0*1
_output_shapes
:�����������<J
mul_2/xConst*
_output_shapes
: *
dtype0*
value
B j�tc
mul_2Mulmul_2/x:output:0add:z:0*
T0*1
_output_shapes
:�����������<S
TanhTanh	mul_2:z:0*
T0*1
_output_shapes
:�����������<J
add_1/xConst*
_output_shapes
: *
dtype0*
value
B j�xf
add_1AddV2add_1/x:output:0Tanh:y:0*
T0*1
_output_shapes
:�����������<\
mul_3Mulmul:z:0	add_1:z:0*
T0*1
_output_shapes
:�����������<b
IdentityIdentity	mul_3:z:0^NoOp*
T0*1
_output_shapes
:�����������<w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:�����������<: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:�����������<
 
_user_specified_nameinputs
�
�
(__inference_conv2d_5_layer_call_fn_55862

inputs!
unknown:<<
	unknown_0:<
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������<*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_conv2d_5_layer_call_and_return_conditional_losses_53351y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:�����������<`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:�����������<: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:�����������<
 
_user_specified_nameinputs
�
M
1__inference_average_pooling2d_layer_call_fn_55892

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4������������������������������������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_average_pooling2d_layer_call_and_return_conditional_losses_53020�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
�
C__inference_conv2d_5_layer_call_and_return_conditional_losses_53351

inputs8
conv2d_readvariableop_resource:<<-
biasadd_readvariableop_resource:<
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:<<*
dtype0r
Conv2D/CastCastConv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*&
_output_shapes
:<<�
Conv2DConv2DinputsConv2D/Cast:y:0*
T0*1
_output_shapes
:�����������<*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:<*
dtype0h
BiasAdd/CastCastBiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:<q
BiasAddBiasAddConv2D:output:0BiasAdd/Cast:y:0*
T0*1
_output_shapes
:�����������<H
mul/xConst*
_output_shapes
: *
dtype0*
value
B j�ph
mulMulmul/x:output:0BiasAdd:output:0*
T0*1
_output_shapes
:�����������<I
Pow/yConst*
_output_shapes
: *
dtype0*
valueB	 j��h
PowPowBiasAdd:output:0Pow/y:output:0*
T0*1
_output_shapes
:�����������<J
mul_1/xConst*
_output_shapes
: *
dtype0*
value
B j�Sc
mul_1Mulmul_1/x:output:0Pow:z:0*
T0*1
_output_shapes
:�����������<e
addAddV2BiasAdd:output:0	mul_1:z:0*
T0*1
_output_shapes
:�����������<J
mul_2/xConst*
_output_shapes
: *
dtype0*
value
B j�tc
mul_2Mulmul_2/x:output:0add:z:0*
T0*1
_output_shapes
:�����������<S
TanhTanh	mul_2:z:0*
T0*1
_output_shapes
:�����������<J
add_1/xConst*
_output_shapes
: *
dtype0*
value
B j�xf
add_1AddV2add_1/x:output:0Tanh:y:0*
T0*1
_output_shapes
:�����������<\
mul_3Mulmul:z:0	add_1:z:0*
T0*1
_output_shapes
:�����������<b
IdentityIdentity	mul_3:z:0^NoOp*
T0*1
_output_shapes
:�����������<w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:�����������<: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:�����������<
 
_user_specified_nameinputs
�
�
C__inference_conv2d_2_layer_call_and_return_conditional_losses_55772

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0r
Conv2D/CastCastConv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*&
_output_shapes
:�
Conv2DConv2DinputsConv2D/Cast:y:0*
T0*1
_output_shapes
:�����������*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0h
BiasAdd/CastCastBiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:q
BiasAddBiasAddConv2D:output:0BiasAdd/Cast:y:0*
T0*1
_output_shapes
:�����������H
mul/xConst*
_output_shapes
: *
dtype0*
value
B j�ph
mulMulmul/x:output:0BiasAdd:output:0*
T0*1
_output_shapes
:�����������I
Pow/yConst*
_output_shapes
: *
dtype0*
valueB	 j��h
PowPowBiasAdd:output:0Pow/y:output:0*
T0*1
_output_shapes
:�����������J
mul_1/xConst*
_output_shapes
: *
dtype0*
value
B j�Sc
mul_1Mulmul_1/x:output:0Pow:z:0*
T0*1
_output_shapes
:�����������e
addAddV2BiasAdd:output:0	mul_1:z:0*
T0*1
_output_shapes
:�����������J
mul_2/xConst*
_output_shapes
: *
dtype0*
value
B j�tc
mul_2Mulmul_2/x:output:0add:z:0*
T0*1
_output_shapes
:�����������S
TanhTanh	mul_2:z:0*
T0*1
_output_shapes
:�����������J
add_1/xConst*
_output_shapes
: *
dtype0*
value
B j�xf
add_1AddV2add_1/x:output:0Tanh:y:0*
T0*1
_output_shapes
:�����������\
mul_3Mulmul:z:0	add_1:z:0*
T0*1
_output_shapes
:�����������b
IdentityIdentity	mul_3:z:0^NoOp*
T0*1
_output_shapes
:�����������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:�����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
�
C__inference_conv2d_4_layer_call_and_return_conditional_losses_55853

inputs8
conv2d_readvariableop_resource:<-
biasadd_readvariableop_resource:<
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:<*
dtype0r
Conv2D/CastCastConv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*&
_output_shapes
:<�
Conv2DConv2DinputsConv2D/Cast:y:0*
T0*1
_output_shapes
:�����������<*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:<*
dtype0h
BiasAdd/CastCastBiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:<q
BiasAddBiasAddConv2D:output:0BiasAdd/Cast:y:0*
T0*1
_output_shapes
:�����������<H
mul/xConst*
_output_shapes
: *
dtype0*
value
B j�ph
mulMulmul/x:output:0BiasAdd:output:0*
T0*1
_output_shapes
:�����������<I
Pow/yConst*
_output_shapes
: *
dtype0*
valueB	 j��h
PowPowBiasAdd:output:0Pow/y:output:0*
T0*1
_output_shapes
:�����������<J
mul_1/xConst*
_output_shapes
: *
dtype0*
value
B j�Sc
mul_1Mulmul_1/x:output:0Pow:z:0*
T0*1
_output_shapes
:�����������<e
addAddV2BiasAdd:output:0	mul_1:z:0*
T0*1
_output_shapes
:�����������<J
mul_2/xConst*
_output_shapes
: *
dtype0*
value
B j�tc
mul_2Mulmul_2/x:output:0add:z:0*
T0*1
_output_shapes
:�����������<S
TanhTanh	mul_2:z:0*
T0*1
_output_shapes
:�����������<J
add_1/xConst*
_output_shapes
: *
dtype0*
value
B j�xf
add_1AddV2add_1/x:output:0Tanh:y:0*
T0*1
_output_shapes
:�����������<\
mul_3Mulmul:z:0	add_1:z:0*
T0*1
_output_shapes
:�����������<b
IdentityIdentity	mul_3:z:0^NoOp*
T0*1
_output_shapes
:�����������<w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:�����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
�
2__inference_conv2d_transpose_1_layer_call_fn_56257

inputs!
unknown:<x
	unknown_0:<
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������<*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *V
fQRO
M__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_53146�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������<`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+���������������������������x: : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������x
 
_user_specified_nameinputs
�,
�
M__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_56305

inputsB
(conv2d_transpose_readvariableop_resource:<x-
biasadd_readvariableop_resource:<
identity��BiasAdd/ReadVariableOp�conv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B :<y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:<x*
dtype0�
conv2d_transpose/CastCast'conv2d_transpose/ReadVariableOp:value:0*

DstT0*

SrcT0*&
_output_shapes
:<x�
conv2d_transposeConv2DBackpropInputstack:output:0conv2d_transpose/Cast:y:0inputs*
T0*A
_output_shapes/
-:+���������������������������<*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:<*
dtype0h
BiasAdd/CastCastBiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:<�
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/Cast:y:0*
T0*A
_output_shapes/
-:+���������������������������<J
mul_2/xConst*
_output_shapes
: *
dtype0*
value
B j�p|
mul_2Mulmul_2/x:output:0BiasAdd:output:0*
T0*A
_output_shapes/
-:+���������������������������<I
Pow/yConst*
_output_shapes
: *
dtype0*
valueB	 j��x
PowPowBiasAdd:output:0Pow/y:output:0*
T0*A
_output_shapes/
-:+���������������������������<J
mul_3/xConst*
_output_shapes
: *
dtype0*
value
B j�Ss
mul_3Mulmul_3/x:output:0Pow:z:0*
T0*A
_output_shapes/
-:+���������������������������<u
addAddV2BiasAdd:output:0	mul_3:z:0*
T0*A
_output_shapes/
-:+���������������������������<J
mul_4/xConst*
_output_shapes
: *
dtype0*
value
B j�ts
mul_4Mulmul_4/x:output:0add:z:0*
T0*A
_output_shapes/
-:+���������������������������<c
TanhTanh	mul_4:z:0*
T0*A
_output_shapes/
-:+���������������������������<J
add_1/xConst*
_output_shapes
: *
dtype0*
value
B j�xv
add_1AddV2add_1/x:output:0Tanh:y:0*
T0*A
_output_shapes/
-:+���������������������������<n
mul_5Mul	mul_2:z:0	add_1:z:0*
T0*A
_output_shapes/
-:+���������������������������<r
IdentityIdentity	mul_5:z:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������<�
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+���������������������������x: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+���������������������������x
 
_user_specified_nameinputs
�
�
C__inference_conv2d_8_layer_call_and_return_conditional_losses_53446

inputs9
conv2d_readvariableop_resource:x�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:x�*
dtype0s
Conv2D/CastCastConv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*'
_output_shapes
:x��
Conv2DConv2DinputsConv2D/Cast:y:0*
T0*0
_output_shapes
:���������<<�*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0i
BiasAdd/CastCastBiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes	
:�p
BiasAddBiasAddConv2D:output:0BiasAdd/Cast:y:0*
T0*0
_output_shapes
:���������<<�H
mul/xConst*
_output_shapes
: *
dtype0*
value
B j�pg
mulMulmul/x:output:0BiasAdd:output:0*
T0*0
_output_shapes
:���������<<�I
Pow/yConst*
_output_shapes
: *
dtype0*
valueB	 j��g
PowPowBiasAdd:output:0Pow/y:output:0*
T0*0
_output_shapes
:���������<<�J
mul_1/xConst*
_output_shapes
: *
dtype0*
value
B j�Sb
mul_1Mulmul_1/x:output:0Pow:z:0*
T0*0
_output_shapes
:���������<<�d
addAddV2BiasAdd:output:0	mul_1:z:0*
T0*0
_output_shapes
:���������<<�J
mul_2/xConst*
_output_shapes
: *
dtype0*
value
B j�tb
mul_2Mulmul_2/x:output:0add:z:0*
T0*0
_output_shapes
:���������<<�R
TanhTanh	mul_2:z:0*
T0*0
_output_shapes
:���������<<�J
add_1/xConst*
_output_shapes
: *
dtype0*
value
B j�xe
add_1AddV2add_1/x:output:0Tanh:y:0*
T0*0
_output_shapes
:���������<<�[
mul_3Mulmul:z:0	add_1:z:0*
T0*0
_output_shapes
:���������<<�a
IdentityIdentity	mul_3:z:0^NoOp*
T0*0
_output_shapes
:���������<<�w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������<<x: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������<<x
 
_user_specified_nameinputs
�
j
N__inference_average_pooling2d_1_layer_call_and_return_conditional_losses_55975

inputs
identity�
AvgPoolAvgPoolinputs*
T0*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingSAME*
strides
{
IdentityIdentityAvgPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
��
�
@__inference_model_layer_call_and_return_conditional_losses_55244

inputs?
%conv2d_conv2d_readvariableop_resource:4
&conv2d_biasadd_readvariableop_resource:A
'conv2d_1_conv2d_readvariableop_resource:6
(conv2d_1_biasadd_readvariableop_resource:A
'conv2d_2_conv2d_readvariableop_resource:6
(conv2d_2_biasadd_readvariableop_resource:A
'conv2d_3_conv2d_readvariableop_resource:6
(conv2d_3_biasadd_readvariableop_resource:A
'conv2d_4_conv2d_readvariableop_resource:<6
(conv2d_4_biasadd_readvariableop_resource:<A
'conv2d_5_conv2d_readvariableop_resource:<<6
(conv2d_5_biasadd_readvariableop_resource:<A
'conv2d_6_conv2d_readvariableop_resource:<x6
(conv2d_6_biasadd_readvariableop_resource:xA
'conv2d_7_conv2d_readvariableop_resource:xx6
(conv2d_7_biasadd_readvariableop_resource:xB
'conv2d_8_conv2d_readvariableop_resource:x�7
(conv2d_8_biasadd_readvariableop_resource:	�C
'conv2d_9_conv2d_readvariableop_resource:��7
(conv2d_9_biasadd_readvariableop_resource:	�D
(conv2d_10_conv2d_readvariableop_resource:��8
)conv2d_10_biasadd_readvariableop_resource:	�D
(conv2d_11_conv2d_readvariableop_resource:��8
)conv2d_11_biasadd_readvariableop_resource:	�T
9conv2d_transpose_conv2d_transpose_readvariableop_resource:x�>
0conv2d_transpose_biasadd_readvariableop_resource:xB
(conv2d_12_conv2d_readvariableop_resource:xx7
)conv2d_12_biasadd_readvariableop_resource:xB
(conv2d_13_conv2d_readvariableop_resource:xx7
)conv2d_13_biasadd_readvariableop_resource:xU
;conv2d_transpose_1_conv2d_transpose_readvariableop_resource:<x@
2conv2d_transpose_1_biasadd_readvariableop_resource:<B
(conv2d_14_conv2d_readvariableop_resource:<<7
)conv2d_14_biasadd_readvariableop_resource:<B
(conv2d_15_conv2d_readvariableop_resource:<<7
)conv2d_15_biasadd_readvariableop_resource:<B
(conv2d_16_conv2d_readvariableop_resource:<7
)conv2d_16_biasadd_readvariableop_resource:
identity��conv2d/BiasAdd/ReadVariableOp�conv2d/Conv2D/ReadVariableOp�conv2d_1/BiasAdd/ReadVariableOp�conv2d_1/Conv2D/ReadVariableOp� conv2d_10/BiasAdd/ReadVariableOp�conv2d_10/Conv2D/ReadVariableOp� conv2d_11/BiasAdd/ReadVariableOp�conv2d_11/Conv2D/ReadVariableOp� conv2d_12/BiasAdd/ReadVariableOp�conv2d_12/Conv2D/ReadVariableOp� conv2d_13/BiasAdd/ReadVariableOp�conv2d_13/Conv2D/ReadVariableOp� conv2d_14/BiasAdd/ReadVariableOp�conv2d_14/Conv2D/ReadVariableOp� conv2d_15/BiasAdd/ReadVariableOp�conv2d_15/Conv2D/ReadVariableOp� conv2d_16/BiasAdd/ReadVariableOp�conv2d_16/Conv2D/ReadVariableOp�conv2d_2/BiasAdd/ReadVariableOp�conv2d_2/Conv2D/ReadVariableOp�conv2d_3/BiasAdd/ReadVariableOp�conv2d_3/Conv2D/ReadVariableOp�conv2d_4/BiasAdd/ReadVariableOp�conv2d_4/Conv2D/ReadVariableOp�conv2d_5/BiasAdd/ReadVariableOp�conv2d_5/Conv2D/ReadVariableOp�conv2d_6/BiasAdd/ReadVariableOp�conv2d_6/Conv2D/ReadVariableOp�conv2d_7/BiasAdd/ReadVariableOp�conv2d_7/Conv2D/ReadVariableOp�conv2d_8/BiasAdd/ReadVariableOp�conv2d_8/Conv2D/ReadVariableOp�conv2d_9/BiasAdd/ReadVariableOp�conv2d_9/Conv2D/ReadVariableOp�'conv2d_transpose/BiasAdd/ReadVariableOp�0conv2d_transpose/conv2d_transpose/ReadVariableOp�)conv2d_transpose_1/BiasAdd/ReadVariableOp�2conv2d_transpose_1/conv2d_transpose/ReadVariableOpf
conv2d/CastCastinputs*

DstT0*

SrcT0*1
_output_shapes
:������������
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
conv2d/Conv2D/CastCast$conv2d/Conv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*&
_output_shapes
:�
conv2d/Conv2DConv2Dconv2d/Cast:y:0conv2d/Conv2D/Cast:y:0*
T0*1
_output_shapes
:�����������*
paddingSAME*
strides
�
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
conv2d/BiasAdd/CastCast%conv2d/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:�
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0conv2d/BiasAdd/Cast:y:0*
T0*1
_output_shapes
:�����������O
conv2d/mul/xConst*
_output_shapes
: *
dtype0*
value
B j�p}

conv2d/mulMulconv2d/mul/x:output:0conv2d/BiasAdd:output:0*
T0*1
_output_shapes
:�����������P
conv2d/Pow/yConst*
_output_shapes
: *
dtype0*
valueB	 j��}

conv2d/PowPowconv2d/BiasAdd:output:0conv2d/Pow/y:output:0*
T0*1
_output_shapes
:�����������Q
conv2d/mul_1/xConst*
_output_shapes
: *
dtype0*
value
B j�Sx
conv2d/mul_1Mulconv2d/mul_1/x:output:0conv2d/Pow:z:0*
T0*1
_output_shapes
:�����������z

conv2d/addAddV2conv2d/BiasAdd:output:0conv2d/mul_1:z:0*
T0*1
_output_shapes
:�����������Q
conv2d/mul_2/xConst*
_output_shapes
: *
dtype0*
value
B j�tx
conv2d/mul_2Mulconv2d/mul_2/x:output:0conv2d/add:z:0*
T0*1
_output_shapes
:�����������a
conv2d/TanhTanhconv2d/mul_2:z:0*
T0*1
_output_shapes
:�����������Q
conv2d/add_1/xConst*
_output_shapes
: *
dtype0*
value
B j�x{
conv2d/add_1AddV2conv2d/add_1/x:output:0conv2d/Tanh:y:0*
T0*1
_output_shapes
:�����������q
conv2d/mul_3Mulconv2d/mul:z:0conv2d/add_1:z:0*
T0*1
_output_shapes
:������������
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
conv2d_1/Conv2D/CastCast&conv2d_1/Conv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*&
_output_shapes
:�
conv2d_1/Conv2DConv2Dconv2d/mul_3:z:0conv2d_1/Conv2D/Cast:y:0*
T0*1
_output_shapes
:�����������*
paddingSAME*
strides
�
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0z
conv2d_1/BiasAdd/CastCast'conv2d_1/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:�
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0conv2d_1/BiasAdd/Cast:y:0*
T0*1
_output_shapes
:�����������Q
conv2d_1/mul/xConst*
_output_shapes
: *
dtype0*
value
B j�p�
conv2d_1/mulMulconv2d_1/mul/x:output:0conv2d_1/BiasAdd:output:0*
T0*1
_output_shapes
:�����������R
conv2d_1/Pow/yConst*
_output_shapes
: *
dtype0*
valueB	 j���
conv2d_1/PowPowconv2d_1/BiasAdd:output:0conv2d_1/Pow/y:output:0*
T0*1
_output_shapes
:�����������S
conv2d_1/mul_1/xConst*
_output_shapes
: *
dtype0*
value
B j�S~
conv2d_1/mul_1Mulconv2d_1/mul_1/x:output:0conv2d_1/Pow:z:0*
T0*1
_output_shapes
:������������
conv2d_1/addAddV2conv2d_1/BiasAdd:output:0conv2d_1/mul_1:z:0*
T0*1
_output_shapes
:�����������S
conv2d_1/mul_2/xConst*
_output_shapes
: *
dtype0*
value
B j�t~
conv2d_1/mul_2Mulconv2d_1/mul_2/x:output:0conv2d_1/add:z:0*
T0*1
_output_shapes
:�����������e
conv2d_1/TanhTanhconv2d_1/mul_2:z:0*
T0*1
_output_shapes
:�����������S
conv2d_1/add_1/xConst*
_output_shapes
: *
dtype0*
value
B j�x�
conv2d_1/add_1AddV2conv2d_1/add_1/x:output:0conv2d_1/Tanh:y:0*
T0*1
_output_shapes
:�����������w
conv2d_1/mul_3Mulconv2d_1/mul:z:0conv2d_1/add_1:z:0*
T0*1
_output_shapes
:������������
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
conv2d_2/Conv2D/CastCast&conv2d_2/Conv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*&
_output_shapes
:�
conv2d_2/Conv2DConv2Dconv2d_1/mul_3:z:0conv2d_2/Conv2D/Cast:y:0*
T0*1
_output_shapes
:�����������*
paddingSAME*
strides
�
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0z
conv2d_2/BiasAdd/CastCast'conv2d_2/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:�
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0conv2d_2/BiasAdd/Cast:y:0*
T0*1
_output_shapes
:�����������Q
conv2d_2/mul/xConst*
_output_shapes
: *
dtype0*
value
B j�p�
conv2d_2/mulMulconv2d_2/mul/x:output:0conv2d_2/BiasAdd:output:0*
T0*1
_output_shapes
:�����������R
conv2d_2/Pow/yConst*
_output_shapes
: *
dtype0*
valueB	 j���
conv2d_2/PowPowconv2d_2/BiasAdd:output:0conv2d_2/Pow/y:output:0*
T0*1
_output_shapes
:�����������S
conv2d_2/mul_1/xConst*
_output_shapes
: *
dtype0*
value
B j�S~
conv2d_2/mul_1Mulconv2d_2/mul_1/x:output:0conv2d_2/Pow:z:0*
T0*1
_output_shapes
:������������
conv2d_2/addAddV2conv2d_2/BiasAdd:output:0conv2d_2/mul_1:z:0*
T0*1
_output_shapes
:�����������S
conv2d_2/mul_2/xConst*
_output_shapes
: *
dtype0*
value
B j�t~
conv2d_2/mul_2Mulconv2d_2/mul_2/x:output:0conv2d_2/add:z:0*
T0*1
_output_shapes
:�����������e
conv2d_2/TanhTanhconv2d_2/mul_2:z:0*
T0*1
_output_shapes
:�����������S
conv2d_2/add_1/xConst*
_output_shapes
: *
dtype0*
value
B j�x�
conv2d_2/add_1AddV2conv2d_2/add_1/x:output:0conv2d_2/Tanh:y:0*
T0*1
_output_shapes
:�����������w
conv2d_2/mul_3Mulconv2d_2/mul:z:0conv2d_2/add_1:z:0*
T0*1
_output_shapes
:������������
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
conv2d_3/Conv2D/CastCast&conv2d_3/Conv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*&
_output_shapes
:�
conv2d_3/Conv2DConv2Dconv2d_2/mul_3:z:0conv2d_3/Conv2D/Cast:y:0*
T0*1
_output_shapes
:�����������*
paddingSAME*
strides
�
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0z
conv2d_3/BiasAdd/CastCast'conv2d_3/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:�
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0conv2d_3/BiasAdd/Cast:y:0*
T0*1
_output_shapes
:�����������Q
conv2d_3/mul/xConst*
_output_shapes
: *
dtype0*
value
B j�p�
conv2d_3/mulMulconv2d_3/mul/x:output:0conv2d_3/BiasAdd:output:0*
T0*1
_output_shapes
:�����������R
conv2d_3/Pow/yConst*
_output_shapes
: *
dtype0*
valueB	 j���
conv2d_3/PowPowconv2d_3/BiasAdd:output:0conv2d_3/Pow/y:output:0*
T0*1
_output_shapes
:�����������S
conv2d_3/mul_1/xConst*
_output_shapes
: *
dtype0*
value
B j�S~
conv2d_3/mul_1Mulconv2d_3/mul_1/x:output:0conv2d_3/Pow:z:0*
T0*1
_output_shapes
:������������
conv2d_3/addAddV2conv2d_3/BiasAdd:output:0conv2d_3/mul_1:z:0*
T0*1
_output_shapes
:�����������S
conv2d_3/mul_2/xConst*
_output_shapes
: *
dtype0*
value
B j�t~
conv2d_3/mul_2Mulconv2d_3/mul_2/x:output:0conv2d_3/add:z:0*
T0*1
_output_shapes
:�����������e
conv2d_3/TanhTanhconv2d_3/mul_2:z:0*
T0*1
_output_shapes
:�����������S
conv2d_3/add_1/xConst*
_output_shapes
: *
dtype0*
value
B j�x�
conv2d_3/add_1AddV2conv2d_3/add_1/x:output:0conv2d_3/Tanh:y:0*
T0*1
_output_shapes
:�����������w
conv2d_3/mul_3Mulconv2d_3/mul:z:0conv2d_3/add_1:z:0*
T0*1
_output_shapes
:�����������k
concatenate/CastCastinputs*

DstT0*

SrcT0*1
_output_shapes
:�����������Y
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concatenate/concatConcatV2conv2d_3/mul_3:z:0concatenate/Cast:y:0 concatenate/concat/axis:output:0*
N*
T0*1
_output_shapes
:������������
conv2d_4/Conv2D/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
:<*
dtype0�
conv2d_4/Conv2D/CastCast&conv2d_4/Conv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*&
_output_shapes
:<�
conv2d_4/Conv2DConv2Dconcatenate/concat:output:0conv2d_4/Conv2D/Cast:y:0*
T0*1
_output_shapes
:�����������<*
paddingSAME*
strides
�
conv2d_4/BiasAdd/ReadVariableOpReadVariableOp(conv2d_4_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype0z
conv2d_4/BiasAdd/CastCast'conv2d_4/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:<�
conv2d_4/BiasAddBiasAddconv2d_4/Conv2D:output:0conv2d_4/BiasAdd/Cast:y:0*
T0*1
_output_shapes
:�����������<Q
conv2d_4/mul/xConst*
_output_shapes
: *
dtype0*
value
B j�p�
conv2d_4/mulMulconv2d_4/mul/x:output:0conv2d_4/BiasAdd:output:0*
T0*1
_output_shapes
:�����������<R
conv2d_4/Pow/yConst*
_output_shapes
: *
dtype0*
valueB	 j���
conv2d_4/PowPowconv2d_4/BiasAdd:output:0conv2d_4/Pow/y:output:0*
T0*1
_output_shapes
:�����������<S
conv2d_4/mul_1/xConst*
_output_shapes
: *
dtype0*
value
B j�S~
conv2d_4/mul_1Mulconv2d_4/mul_1/x:output:0conv2d_4/Pow:z:0*
T0*1
_output_shapes
:�����������<�
conv2d_4/addAddV2conv2d_4/BiasAdd:output:0conv2d_4/mul_1:z:0*
T0*1
_output_shapes
:�����������<S
conv2d_4/mul_2/xConst*
_output_shapes
: *
dtype0*
value
B j�t~
conv2d_4/mul_2Mulconv2d_4/mul_2/x:output:0conv2d_4/add:z:0*
T0*1
_output_shapes
:�����������<e
conv2d_4/TanhTanhconv2d_4/mul_2:z:0*
T0*1
_output_shapes
:�����������<S
conv2d_4/add_1/xConst*
_output_shapes
: *
dtype0*
value
B j�x�
conv2d_4/add_1AddV2conv2d_4/add_1/x:output:0conv2d_4/Tanh:y:0*
T0*1
_output_shapes
:�����������<w
conv2d_4/mul_3Mulconv2d_4/mul:z:0conv2d_4/add_1:z:0*
T0*1
_output_shapes
:�����������<�
conv2d_5/Conv2D/ReadVariableOpReadVariableOp'conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
:<<*
dtype0�
conv2d_5/Conv2D/CastCast&conv2d_5/Conv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*&
_output_shapes
:<<�
conv2d_5/Conv2DConv2Dconv2d_4/mul_3:z:0conv2d_5/Conv2D/Cast:y:0*
T0*1
_output_shapes
:�����������<*
paddingSAME*
strides
�
conv2d_5/BiasAdd/ReadVariableOpReadVariableOp(conv2d_5_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype0z
conv2d_5/BiasAdd/CastCast'conv2d_5/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:<�
conv2d_5/BiasAddBiasAddconv2d_5/Conv2D:output:0conv2d_5/BiasAdd/Cast:y:0*
T0*1
_output_shapes
:�����������<Q
conv2d_5/mul/xConst*
_output_shapes
: *
dtype0*
value
B j�p�
conv2d_5/mulMulconv2d_5/mul/x:output:0conv2d_5/BiasAdd:output:0*
T0*1
_output_shapes
:�����������<R
conv2d_5/Pow/yConst*
_output_shapes
: *
dtype0*
valueB	 j���
conv2d_5/PowPowconv2d_5/BiasAdd:output:0conv2d_5/Pow/y:output:0*
T0*1
_output_shapes
:�����������<S
conv2d_5/mul_1/xConst*
_output_shapes
: *
dtype0*
value
B j�S~
conv2d_5/mul_1Mulconv2d_5/mul_1/x:output:0conv2d_5/Pow:z:0*
T0*1
_output_shapes
:�����������<�
conv2d_5/addAddV2conv2d_5/BiasAdd:output:0conv2d_5/mul_1:z:0*
T0*1
_output_shapes
:�����������<S
conv2d_5/mul_2/xConst*
_output_shapes
: *
dtype0*
value
B j�t~
conv2d_5/mul_2Mulconv2d_5/mul_2/x:output:0conv2d_5/add:z:0*
T0*1
_output_shapes
:�����������<e
conv2d_5/TanhTanhconv2d_5/mul_2:z:0*
T0*1
_output_shapes
:�����������<S
conv2d_5/add_1/xConst*
_output_shapes
: *
dtype0*
value
B j�x�
conv2d_5/add_1AddV2conv2d_5/add_1/x:output:0conv2d_5/Tanh:y:0*
T0*1
_output_shapes
:�����������<w
conv2d_5/mul_3Mulconv2d_5/mul:z:0conv2d_5/add_1:z:0*
T0*1
_output_shapes
:�����������<�
average_pooling2d/AvgPoolAvgPoolconv2d_5/mul_3:z:0*
T0*/
_output_shapes
:���������xx<*
ksize
*
paddingSAME*
strides
�
conv2d_6/Conv2D/ReadVariableOpReadVariableOp'conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
:<x*
dtype0�
conv2d_6/Conv2D/CastCast&conv2d_6/Conv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*&
_output_shapes
:<x�
conv2d_6/Conv2DConv2D"average_pooling2d/AvgPool:output:0conv2d_6/Conv2D/Cast:y:0*
T0*/
_output_shapes
:���������xxx*
paddingSAME*
strides
�
conv2d_6/BiasAdd/ReadVariableOpReadVariableOp(conv2d_6_biasadd_readvariableop_resource*
_output_shapes
:x*
dtype0z
conv2d_6/BiasAdd/CastCast'conv2d_6/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:x�
conv2d_6/BiasAddBiasAddconv2d_6/Conv2D:output:0conv2d_6/BiasAdd/Cast:y:0*
T0*/
_output_shapes
:���������xxxQ
conv2d_6/mul/xConst*
_output_shapes
: *
dtype0*
value
B j�p�
conv2d_6/mulMulconv2d_6/mul/x:output:0conv2d_6/BiasAdd:output:0*
T0*/
_output_shapes
:���������xxxR
conv2d_6/Pow/yConst*
_output_shapes
: *
dtype0*
valueB	 j���
conv2d_6/PowPowconv2d_6/BiasAdd:output:0conv2d_6/Pow/y:output:0*
T0*/
_output_shapes
:���������xxxS
conv2d_6/mul_1/xConst*
_output_shapes
: *
dtype0*
value
B j�S|
conv2d_6/mul_1Mulconv2d_6/mul_1/x:output:0conv2d_6/Pow:z:0*
T0*/
_output_shapes
:���������xxx~
conv2d_6/addAddV2conv2d_6/BiasAdd:output:0conv2d_6/mul_1:z:0*
T0*/
_output_shapes
:���������xxxS
conv2d_6/mul_2/xConst*
_output_shapes
: *
dtype0*
value
B j�t|
conv2d_6/mul_2Mulconv2d_6/mul_2/x:output:0conv2d_6/add:z:0*
T0*/
_output_shapes
:���������xxxc
conv2d_6/TanhTanhconv2d_6/mul_2:z:0*
T0*/
_output_shapes
:���������xxxS
conv2d_6/add_1/xConst*
_output_shapes
: *
dtype0*
value
B j�x
conv2d_6/add_1AddV2conv2d_6/add_1/x:output:0conv2d_6/Tanh:y:0*
T0*/
_output_shapes
:���������xxxu
conv2d_6/mul_3Mulconv2d_6/mul:z:0conv2d_6/add_1:z:0*
T0*/
_output_shapes
:���������xxx�
conv2d_7/Conv2D/ReadVariableOpReadVariableOp'conv2d_7_conv2d_readvariableop_resource*&
_output_shapes
:xx*
dtype0�
conv2d_7/Conv2D/CastCast&conv2d_7/Conv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*&
_output_shapes
:xx�
conv2d_7/Conv2DConv2Dconv2d_6/mul_3:z:0conv2d_7/Conv2D/Cast:y:0*
T0*/
_output_shapes
:���������xxx*
paddingSAME*
strides
�
conv2d_7/BiasAdd/ReadVariableOpReadVariableOp(conv2d_7_biasadd_readvariableop_resource*
_output_shapes
:x*
dtype0z
conv2d_7/BiasAdd/CastCast'conv2d_7/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:x�
conv2d_7/BiasAddBiasAddconv2d_7/Conv2D:output:0conv2d_7/BiasAdd/Cast:y:0*
T0*/
_output_shapes
:���������xxxQ
conv2d_7/mul/xConst*
_output_shapes
: *
dtype0*
value
B j�p�
conv2d_7/mulMulconv2d_7/mul/x:output:0conv2d_7/BiasAdd:output:0*
T0*/
_output_shapes
:���������xxxR
conv2d_7/Pow/yConst*
_output_shapes
: *
dtype0*
valueB	 j���
conv2d_7/PowPowconv2d_7/BiasAdd:output:0conv2d_7/Pow/y:output:0*
T0*/
_output_shapes
:���������xxxS
conv2d_7/mul_1/xConst*
_output_shapes
: *
dtype0*
value
B j�S|
conv2d_7/mul_1Mulconv2d_7/mul_1/x:output:0conv2d_7/Pow:z:0*
T0*/
_output_shapes
:���������xxx~
conv2d_7/addAddV2conv2d_7/BiasAdd:output:0conv2d_7/mul_1:z:0*
T0*/
_output_shapes
:���������xxxS
conv2d_7/mul_2/xConst*
_output_shapes
: *
dtype0*
value
B j�t|
conv2d_7/mul_2Mulconv2d_7/mul_2/x:output:0conv2d_7/add:z:0*
T0*/
_output_shapes
:���������xxxc
conv2d_7/TanhTanhconv2d_7/mul_2:z:0*
T0*/
_output_shapes
:���������xxxS
conv2d_7/add_1/xConst*
_output_shapes
: *
dtype0*
value
B j�x
conv2d_7/add_1AddV2conv2d_7/add_1/x:output:0conv2d_7/Tanh:y:0*
T0*/
_output_shapes
:���������xxxu
conv2d_7/mul_3Mulconv2d_7/mul:z:0conv2d_7/add_1:z:0*
T0*/
_output_shapes
:���������xxx�
average_pooling2d_1/AvgPoolAvgPoolconv2d_7/mul_3:z:0*
T0*/
_output_shapes
:���������<<x*
ksize
*
paddingSAME*
strides
�
conv2d_8/Conv2D/ReadVariableOpReadVariableOp'conv2d_8_conv2d_readvariableop_resource*'
_output_shapes
:x�*
dtype0�
conv2d_8/Conv2D/CastCast&conv2d_8/Conv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*'
_output_shapes
:x��
conv2d_8/Conv2DConv2D$average_pooling2d_1/AvgPool:output:0conv2d_8/Conv2D/Cast:y:0*
T0*0
_output_shapes
:���������<<�*
paddingSAME*
strides
�
conv2d_8/BiasAdd/ReadVariableOpReadVariableOp(conv2d_8_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0{
conv2d_8/BiasAdd/CastCast'conv2d_8/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes	
:��
conv2d_8/BiasAddBiasAddconv2d_8/Conv2D:output:0conv2d_8/BiasAdd/Cast:y:0*
T0*0
_output_shapes
:���������<<�Q
conv2d_8/mul/xConst*
_output_shapes
: *
dtype0*
value
B j�p�
conv2d_8/mulMulconv2d_8/mul/x:output:0conv2d_8/BiasAdd:output:0*
T0*0
_output_shapes
:���������<<�R
conv2d_8/Pow/yConst*
_output_shapes
: *
dtype0*
valueB	 j���
conv2d_8/PowPowconv2d_8/BiasAdd:output:0conv2d_8/Pow/y:output:0*
T0*0
_output_shapes
:���������<<�S
conv2d_8/mul_1/xConst*
_output_shapes
: *
dtype0*
value
B j�S}
conv2d_8/mul_1Mulconv2d_8/mul_1/x:output:0conv2d_8/Pow:z:0*
T0*0
_output_shapes
:���������<<�
conv2d_8/addAddV2conv2d_8/BiasAdd:output:0conv2d_8/mul_1:z:0*
T0*0
_output_shapes
:���������<<�S
conv2d_8/mul_2/xConst*
_output_shapes
: *
dtype0*
value
B j�t}
conv2d_8/mul_2Mulconv2d_8/mul_2/x:output:0conv2d_8/add:z:0*
T0*0
_output_shapes
:���������<<�d
conv2d_8/TanhTanhconv2d_8/mul_2:z:0*
T0*0
_output_shapes
:���������<<�S
conv2d_8/add_1/xConst*
_output_shapes
: *
dtype0*
value
B j�x�
conv2d_8/add_1AddV2conv2d_8/add_1/x:output:0conv2d_8/Tanh:y:0*
T0*0
_output_shapes
:���������<<�v
conv2d_8/mul_3Mulconv2d_8/mul:z:0conv2d_8/add_1:z:0*
T0*0
_output_shapes
:���������<<��
conv2d_9/Conv2D/ReadVariableOpReadVariableOp'conv2d_9_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
conv2d_9/Conv2D/CastCast&conv2d_9/Conv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*(
_output_shapes
:���
conv2d_9/Conv2DConv2Dconv2d_8/mul_3:z:0conv2d_9/Conv2D/Cast:y:0*
T0*0
_output_shapes
:���������<<�*
paddingSAME*
strides
�
conv2d_9/BiasAdd/ReadVariableOpReadVariableOp(conv2d_9_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0{
conv2d_9/BiasAdd/CastCast'conv2d_9/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes	
:��
conv2d_9/BiasAddBiasAddconv2d_9/Conv2D:output:0conv2d_9/BiasAdd/Cast:y:0*
T0*0
_output_shapes
:���������<<�Q
conv2d_9/mul/xConst*
_output_shapes
: *
dtype0*
value
B j�p�
conv2d_9/mulMulconv2d_9/mul/x:output:0conv2d_9/BiasAdd:output:0*
T0*0
_output_shapes
:���������<<�R
conv2d_9/Pow/yConst*
_output_shapes
: *
dtype0*
valueB	 j���
conv2d_9/PowPowconv2d_9/BiasAdd:output:0conv2d_9/Pow/y:output:0*
T0*0
_output_shapes
:���������<<�S
conv2d_9/mul_1/xConst*
_output_shapes
: *
dtype0*
value
B j�S}
conv2d_9/mul_1Mulconv2d_9/mul_1/x:output:0conv2d_9/Pow:z:0*
T0*0
_output_shapes
:���������<<�
conv2d_9/addAddV2conv2d_9/BiasAdd:output:0conv2d_9/mul_1:z:0*
T0*0
_output_shapes
:���������<<�S
conv2d_9/mul_2/xConst*
_output_shapes
: *
dtype0*
value
B j�t}
conv2d_9/mul_2Mulconv2d_9/mul_2/x:output:0conv2d_9/add:z:0*
T0*0
_output_shapes
:���������<<�d
conv2d_9/TanhTanhconv2d_9/mul_2:z:0*
T0*0
_output_shapes
:���������<<�S
conv2d_9/add_1/xConst*
_output_shapes
: *
dtype0*
value
B j�x�
conv2d_9/add_1AddV2conv2d_9/add_1/x:output:0conv2d_9/Tanh:y:0*
T0*0
_output_shapes
:���������<<�v
conv2d_9/mul_3Mulconv2d_9/mul:z:0conv2d_9/add_1:z:0*
T0*0
_output_shapes
:���������<<��
conv2d_10/Conv2D/ReadVariableOpReadVariableOp(conv2d_10_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
conv2d_10/Conv2D/CastCast'conv2d_10/Conv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*(
_output_shapes
:���
conv2d_10/Conv2DConv2Dconv2d_9/mul_3:z:0conv2d_10/Conv2D/Cast:y:0*
T0*0
_output_shapes
:���������<<�*
paddingSAME*
strides
�
 conv2d_10/BiasAdd/ReadVariableOpReadVariableOp)conv2d_10_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0}
conv2d_10/BiasAdd/CastCast(conv2d_10/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes	
:��
conv2d_10/BiasAddBiasAddconv2d_10/Conv2D:output:0conv2d_10/BiasAdd/Cast:y:0*
T0*0
_output_shapes
:���������<<�R
conv2d_10/mul/xConst*
_output_shapes
: *
dtype0*
value
B j�p�
conv2d_10/mulMulconv2d_10/mul/x:output:0conv2d_10/BiasAdd:output:0*
T0*0
_output_shapes
:���������<<�S
conv2d_10/Pow/yConst*
_output_shapes
: *
dtype0*
valueB	 j���
conv2d_10/PowPowconv2d_10/BiasAdd:output:0conv2d_10/Pow/y:output:0*
T0*0
_output_shapes
:���������<<�T
conv2d_10/mul_1/xConst*
_output_shapes
: *
dtype0*
value
B j�S�
conv2d_10/mul_1Mulconv2d_10/mul_1/x:output:0conv2d_10/Pow:z:0*
T0*0
_output_shapes
:���������<<��
conv2d_10/addAddV2conv2d_10/BiasAdd:output:0conv2d_10/mul_1:z:0*
T0*0
_output_shapes
:���������<<�T
conv2d_10/mul_2/xConst*
_output_shapes
: *
dtype0*
value
B j�t�
conv2d_10/mul_2Mulconv2d_10/mul_2/x:output:0conv2d_10/add:z:0*
T0*0
_output_shapes
:���������<<�f
conv2d_10/TanhTanhconv2d_10/mul_2:z:0*
T0*0
_output_shapes
:���������<<�T
conv2d_10/add_1/xConst*
_output_shapes
: *
dtype0*
value
B j�x�
conv2d_10/add_1AddV2conv2d_10/add_1/x:output:0conv2d_10/Tanh:y:0*
T0*0
_output_shapes
:���������<<�y
conv2d_10/mul_3Mulconv2d_10/mul:z:0conv2d_10/add_1:z:0*
T0*0
_output_shapes
:���������<<��
conv2d_11/Conv2D/ReadVariableOpReadVariableOp(conv2d_11_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
conv2d_11/Conv2D/CastCast'conv2d_11/Conv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*(
_output_shapes
:���
conv2d_11/Conv2DConv2Dconv2d_10/mul_3:z:0conv2d_11/Conv2D/Cast:y:0*
T0*0
_output_shapes
:���������<<�*
paddingSAME*
strides
�
 conv2d_11/BiasAdd/ReadVariableOpReadVariableOp)conv2d_11_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0}
conv2d_11/BiasAdd/CastCast(conv2d_11/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes	
:��
conv2d_11/BiasAddBiasAddconv2d_11/Conv2D:output:0conv2d_11/BiasAdd/Cast:y:0*
T0*0
_output_shapes
:���������<<�R
conv2d_11/mul/xConst*
_output_shapes
: *
dtype0*
value
B j�p�
conv2d_11/mulMulconv2d_11/mul/x:output:0conv2d_11/BiasAdd:output:0*
T0*0
_output_shapes
:���������<<�S
conv2d_11/Pow/yConst*
_output_shapes
: *
dtype0*
valueB	 j���
conv2d_11/PowPowconv2d_11/BiasAdd:output:0conv2d_11/Pow/y:output:0*
T0*0
_output_shapes
:���������<<�T
conv2d_11/mul_1/xConst*
_output_shapes
: *
dtype0*
value
B j�S�
conv2d_11/mul_1Mulconv2d_11/mul_1/x:output:0conv2d_11/Pow:z:0*
T0*0
_output_shapes
:���������<<��
conv2d_11/addAddV2conv2d_11/BiasAdd:output:0conv2d_11/mul_1:z:0*
T0*0
_output_shapes
:���������<<�T
conv2d_11/mul_2/xConst*
_output_shapes
: *
dtype0*
value
B j�t�
conv2d_11/mul_2Mulconv2d_11/mul_2/x:output:0conv2d_11/add:z:0*
T0*0
_output_shapes
:���������<<�f
conv2d_11/TanhTanhconv2d_11/mul_2:z:0*
T0*0
_output_shapes
:���������<<�T
conv2d_11/add_1/xConst*
_output_shapes
: *
dtype0*
value
B j�x�
conv2d_11/add_1AddV2conv2d_11/add_1/x:output:0conv2d_11/Tanh:y:0*
T0*0
_output_shapes
:���������<<�y
conv2d_11/mul_3Mulconv2d_11/mul:z:0conv2d_11/add_1:z:0*
T0*0
_output_shapes
:���������<<�Y
conv2d_transpose/ShapeShapeconv2d_11/mul_3:z:0*
T0*
_output_shapes
:n
$conv2d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: p
&conv2d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&conv2d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
conv2d_transpose/strided_sliceStridedSliceconv2d_transpose/Shape:output:0-conv2d_transpose/strided_slice/stack:output:0/conv2d_transpose/strided_slice/stack_1:output:0/conv2d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskZ
conv2d_transpose/stack/1Const*
_output_shapes
: *
dtype0*
value	B :xZ
conv2d_transpose/stack/2Const*
_output_shapes
: *
dtype0*
value	B :xZ
conv2d_transpose/stack/3Const*
_output_shapes
: *
dtype0*
value	B :x�
conv2d_transpose/stackPack'conv2d_transpose/strided_slice:output:0!conv2d_transpose/stack/1:output:0!conv2d_transpose/stack/2:output:0!conv2d_transpose/stack/3:output:0*
N*
T0*
_output_shapes
:p
&conv2d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv2d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv2d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
 conv2d_transpose/strided_slice_1StridedSliceconv2d_transpose/stack:output:0/conv2d_transpose/strided_slice_1/stack:output:01conv2d_transpose/strided_slice_1/stack_1:output:01conv2d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
0conv2d_transpose/conv2d_transpose/ReadVariableOpReadVariableOp9conv2d_transpose_conv2d_transpose_readvariableop_resource*'
_output_shapes
:x�*
dtype0�
&conv2d_transpose/conv2d_transpose/CastCast8conv2d_transpose/conv2d_transpose/ReadVariableOp:value:0*

DstT0*

SrcT0*'
_output_shapes
:x��
!conv2d_transpose/conv2d_transposeConv2DBackpropInputconv2d_transpose/stack:output:0*conv2d_transpose/conv2d_transpose/Cast:y:0conv2d_11/mul_3:z:0*
T0*/
_output_shapes
:���������xxx*
paddingSAME*
strides
�
'conv2d_transpose/BiasAdd/ReadVariableOpReadVariableOp0conv2d_transpose_biasadd_readvariableop_resource*
_output_shapes
:x*
dtype0�
conv2d_transpose/BiasAdd/CastCast/conv2d_transpose/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:x�
conv2d_transpose/BiasAddBiasAdd*conv2d_transpose/conv2d_transpose:output:0!conv2d_transpose/BiasAdd/Cast:y:0*
T0*/
_output_shapes
:���������xxxY
conv2d_transpose/mul/xConst*
_output_shapes
: *
dtype0*
value
B j�p�
conv2d_transpose/mulMulconv2d_transpose/mul/x:output:0!conv2d_transpose/BiasAdd:output:0*
T0*/
_output_shapes
:���������xxxZ
conv2d_transpose/Pow/yConst*
_output_shapes
: *
dtype0*
valueB	 j���
conv2d_transpose/PowPow!conv2d_transpose/BiasAdd:output:0conv2d_transpose/Pow/y:output:0*
T0*/
_output_shapes
:���������xxx[
conv2d_transpose/mul_1/xConst*
_output_shapes
: *
dtype0*
value
B j�S�
conv2d_transpose/mul_1Mul!conv2d_transpose/mul_1/x:output:0conv2d_transpose/Pow:z:0*
T0*/
_output_shapes
:���������xxx�
conv2d_transpose/addAddV2!conv2d_transpose/BiasAdd:output:0conv2d_transpose/mul_1:z:0*
T0*/
_output_shapes
:���������xxx[
conv2d_transpose/mul_2/xConst*
_output_shapes
: *
dtype0*
value
B j�t�
conv2d_transpose/mul_2Mul!conv2d_transpose/mul_2/x:output:0conv2d_transpose/add:z:0*
T0*/
_output_shapes
:���������xxxs
conv2d_transpose/TanhTanhconv2d_transpose/mul_2:z:0*
T0*/
_output_shapes
:���������xxx[
conv2d_transpose/add_1/xConst*
_output_shapes
: *
dtype0*
value
B j�x�
conv2d_transpose/add_1AddV2!conv2d_transpose/add_1/x:output:0conv2d_transpose/Tanh:y:0*
T0*/
_output_shapes
:���������xxx�
conv2d_transpose/mul_3Mulconv2d_transpose/mul:z:0conv2d_transpose/add_1:z:0*
T0*/
_output_shapes
:���������xxxz
add/addAddV2conv2d_transpose/mul_3:z:0conv2d_7/mul_3:z:0*
T0*/
_output_shapes
:���������xxx�
conv2d_12/Conv2D/ReadVariableOpReadVariableOp(conv2d_12_conv2d_readvariableop_resource*&
_output_shapes
:xx*
dtype0�
conv2d_12/Conv2D/CastCast'conv2d_12/Conv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*&
_output_shapes
:xx�
conv2d_12/Conv2DConv2Dadd/add:z:0conv2d_12/Conv2D/Cast:y:0*
T0*/
_output_shapes
:���������xxx*
paddingSAME*
strides
�
 conv2d_12/BiasAdd/ReadVariableOpReadVariableOp)conv2d_12_biasadd_readvariableop_resource*
_output_shapes
:x*
dtype0|
conv2d_12/BiasAdd/CastCast(conv2d_12/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:x�
conv2d_12/BiasAddBiasAddconv2d_12/Conv2D:output:0conv2d_12/BiasAdd/Cast:y:0*
T0*/
_output_shapes
:���������xxxR
conv2d_12/mul/xConst*
_output_shapes
: *
dtype0*
value
B j�p�
conv2d_12/mulMulconv2d_12/mul/x:output:0conv2d_12/BiasAdd:output:0*
T0*/
_output_shapes
:���������xxxS
conv2d_12/Pow/yConst*
_output_shapes
: *
dtype0*
valueB	 j���
conv2d_12/PowPowconv2d_12/BiasAdd:output:0conv2d_12/Pow/y:output:0*
T0*/
_output_shapes
:���������xxxT
conv2d_12/mul_1/xConst*
_output_shapes
: *
dtype0*
value
B j�S
conv2d_12/mul_1Mulconv2d_12/mul_1/x:output:0conv2d_12/Pow:z:0*
T0*/
_output_shapes
:���������xxx�
conv2d_12/addAddV2conv2d_12/BiasAdd:output:0conv2d_12/mul_1:z:0*
T0*/
_output_shapes
:���������xxxT
conv2d_12/mul_2/xConst*
_output_shapes
: *
dtype0*
value
B j�t
conv2d_12/mul_2Mulconv2d_12/mul_2/x:output:0conv2d_12/add:z:0*
T0*/
_output_shapes
:���������xxxe
conv2d_12/TanhTanhconv2d_12/mul_2:z:0*
T0*/
_output_shapes
:���������xxxT
conv2d_12/add_1/xConst*
_output_shapes
: *
dtype0*
value
B j�x�
conv2d_12/add_1AddV2conv2d_12/add_1/x:output:0conv2d_12/Tanh:y:0*
T0*/
_output_shapes
:���������xxxx
conv2d_12/mul_3Mulconv2d_12/mul:z:0conv2d_12/add_1:z:0*
T0*/
_output_shapes
:���������xxx�
conv2d_13/Conv2D/ReadVariableOpReadVariableOp(conv2d_13_conv2d_readvariableop_resource*&
_output_shapes
:xx*
dtype0�
conv2d_13/Conv2D/CastCast'conv2d_13/Conv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*&
_output_shapes
:xx�
conv2d_13/Conv2DConv2Dconv2d_12/mul_3:z:0conv2d_13/Conv2D/Cast:y:0*
T0*/
_output_shapes
:���������xxx*
paddingSAME*
strides
�
 conv2d_13/BiasAdd/ReadVariableOpReadVariableOp)conv2d_13_biasadd_readvariableop_resource*
_output_shapes
:x*
dtype0|
conv2d_13/BiasAdd/CastCast(conv2d_13/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:x�
conv2d_13/BiasAddBiasAddconv2d_13/Conv2D:output:0conv2d_13/BiasAdd/Cast:y:0*
T0*/
_output_shapes
:���������xxxR
conv2d_13/mul/xConst*
_output_shapes
: *
dtype0*
value
B j�p�
conv2d_13/mulMulconv2d_13/mul/x:output:0conv2d_13/BiasAdd:output:0*
T0*/
_output_shapes
:���������xxxS
conv2d_13/Pow/yConst*
_output_shapes
: *
dtype0*
valueB	 j���
conv2d_13/PowPowconv2d_13/BiasAdd:output:0conv2d_13/Pow/y:output:0*
T0*/
_output_shapes
:���������xxxT
conv2d_13/mul_1/xConst*
_output_shapes
: *
dtype0*
value
B j�S
conv2d_13/mul_1Mulconv2d_13/mul_1/x:output:0conv2d_13/Pow:z:0*
T0*/
_output_shapes
:���������xxx�
conv2d_13/addAddV2conv2d_13/BiasAdd:output:0conv2d_13/mul_1:z:0*
T0*/
_output_shapes
:���������xxxT
conv2d_13/mul_2/xConst*
_output_shapes
: *
dtype0*
value
B j�t
conv2d_13/mul_2Mulconv2d_13/mul_2/x:output:0conv2d_13/add:z:0*
T0*/
_output_shapes
:���������xxxe
conv2d_13/TanhTanhconv2d_13/mul_2:z:0*
T0*/
_output_shapes
:���������xxxT
conv2d_13/add_1/xConst*
_output_shapes
: *
dtype0*
value
B j�x�
conv2d_13/add_1AddV2conv2d_13/add_1/x:output:0conv2d_13/Tanh:y:0*
T0*/
_output_shapes
:���������xxxx
conv2d_13/mul_3Mulconv2d_13/mul:z:0conv2d_13/add_1:z:0*
T0*/
_output_shapes
:���������xxx[
conv2d_transpose_1/ShapeShapeconv2d_13/mul_3:z:0*
T0*
_output_shapes
:p
&conv2d_transpose_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv2d_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv2d_transpose_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
 conv2d_transpose_1/strided_sliceStridedSlice!conv2d_transpose_1/Shape:output:0/conv2d_transpose_1/strided_slice/stack:output:01conv2d_transpose_1/strided_slice/stack_1:output:01conv2d_transpose_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
conv2d_transpose_1/stack/1Const*
_output_shapes
: *
dtype0*
value
B :�]
conv2d_transpose_1/stack/2Const*
_output_shapes
: *
dtype0*
value
B :�\
conv2d_transpose_1/stack/3Const*
_output_shapes
: *
dtype0*
value	B :<�
conv2d_transpose_1/stackPack)conv2d_transpose_1/strided_slice:output:0#conv2d_transpose_1/stack/1:output:0#conv2d_transpose_1/stack/2:output:0#conv2d_transpose_1/stack/3:output:0*
N*
T0*
_output_shapes
:r
(conv2d_transpose_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*conv2d_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*conv2d_transpose_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
"conv2d_transpose_1/strided_slice_1StridedSlice!conv2d_transpose_1/stack:output:01conv2d_transpose_1/strided_slice_1/stack:output:03conv2d_transpose_1/strided_slice_1/stack_1:output:03conv2d_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
2conv2d_transpose_1/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_1_conv2d_transpose_readvariableop_resource*&
_output_shapes
:<x*
dtype0�
(conv2d_transpose_1/conv2d_transpose/CastCast:conv2d_transpose_1/conv2d_transpose/ReadVariableOp:value:0*

DstT0*

SrcT0*&
_output_shapes
:<x�
#conv2d_transpose_1/conv2d_transposeConv2DBackpropInput!conv2d_transpose_1/stack:output:0,conv2d_transpose_1/conv2d_transpose/Cast:y:0conv2d_13/mul_3:z:0*
T0*1
_output_shapes
:�����������<*
paddingSAME*
strides
�
)conv2d_transpose_1/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_1_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype0�
conv2d_transpose_1/BiasAdd/CastCast1conv2d_transpose_1/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:<�
conv2d_transpose_1/BiasAddBiasAdd,conv2d_transpose_1/conv2d_transpose:output:0#conv2d_transpose_1/BiasAdd/Cast:y:0*
T0*1
_output_shapes
:�����������<[
conv2d_transpose_1/mul/xConst*
_output_shapes
: *
dtype0*
value
B j�p�
conv2d_transpose_1/mulMul!conv2d_transpose_1/mul/x:output:0#conv2d_transpose_1/BiasAdd:output:0*
T0*1
_output_shapes
:�����������<\
conv2d_transpose_1/Pow/yConst*
_output_shapes
: *
dtype0*
valueB	 j���
conv2d_transpose_1/PowPow#conv2d_transpose_1/BiasAdd:output:0!conv2d_transpose_1/Pow/y:output:0*
T0*1
_output_shapes
:�����������<]
conv2d_transpose_1/mul_1/xConst*
_output_shapes
: *
dtype0*
value
B j�S�
conv2d_transpose_1/mul_1Mul#conv2d_transpose_1/mul_1/x:output:0conv2d_transpose_1/Pow:z:0*
T0*1
_output_shapes
:�����������<�
conv2d_transpose_1/addAddV2#conv2d_transpose_1/BiasAdd:output:0conv2d_transpose_1/mul_1:z:0*
T0*1
_output_shapes
:�����������<]
conv2d_transpose_1/mul_2/xConst*
_output_shapes
: *
dtype0*
value
B j�t�
conv2d_transpose_1/mul_2Mul#conv2d_transpose_1/mul_2/x:output:0conv2d_transpose_1/add:z:0*
T0*1
_output_shapes
:�����������<y
conv2d_transpose_1/TanhTanhconv2d_transpose_1/mul_2:z:0*
T0*1
_output_shapes
:�����������<]
conv2d_transpose_1/add_1/xConst*
_output_shapes
: *
dtype0*
value
B j�x�
conv2d_transpose_1/add_1AddV2#conv2d_transpose_1/add_1/x:output:0conv2d_transpose_1/Tanh:y:0*
T0*1
_output_shapes
:�����������<�
conv2d_transpose_1/mul_3Mulconv2d_transpose_1/mul:z:0conv2d_transpose_1/add_1:z:0*
T0*1
_output_shapes
:�����������<�
	add_1/addAddV2conv2d_transpose_1/mul_3:z:0conv2d_5/mul_3:z:0*
T0*1
_output_shapes
:�����������<�
conv2d_14/Conv2D/ReadVariableOpReadVariableOp(conv2d_14_conv2d_readvariableop_resource*&
_output_shapes
:<<*
dtype0�
conv2d_14/Conv2D/CastCast'conv2d_14/Conv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*&
_output_shapes
:<<�
conv2d_14/Conv2DConv2Dadd_1/add:z:0conv2d_14/Conv2D/Cast:y:0*
T0*1
_output_shapes
:�����������<*
paddingSAME*
strides
�
 conv2d_14/BiasAdd/ReadVariableOpReadVariableOp)conv2d_14_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype0|
conv2d_14/BiasAdd/CastCast(conv2d_14/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:<�
conv2d_14/BiasAddBiasAddconv2d_14/Conv2D:output:0conv2d_14/BiasAdd/Cast:y:0*
T0*1
_output_shapes
:�����������<R
conv2d_14/mul/xConst*
_output_shapes
: *
dtype0*
value
B j�p�
conv2d_14/mulMulconv2d_14/mul/x:output:0conv2d_14/BiasAdd:output:0*
T0*1
_output_shapes
:�����������<S
conv2d_14/Pow/yConst*
_output_shapes
: *
dtype0*
valueB	 j���
conv2d_14/PowPowconv2d_14/BiasAdd:output:0conv2d_14/Pow/y:output:0*
T0*1
_output_shapes
:�����������<T
conv2d_14/mul_1/xConst*
_output_shapes
: *
dtype0*
value
B j�S�
conv2d_14/mul_1Mulconv2d_14/mul_1/x:output:0conv2d_14/Pow:z:0*
T0*1
_output_shapes
:�����������<�
conv2d_14/addAddV2conv2d_14/BiasAdd:output:0conv2d_14/mul_1:z:0*
T0*1
_output_shapes
:�����������<T
conv2d_14/mul_2/xConst*
_output_shapes
: *
dtype0*
value
B j�t�
conv2d_14/mul_2Mulconv2d_14/mul_2/x:output:0conv2d_14/add:z:0*
T0*1
_output_shapes
:�����������<g
conv2d_14/TanhTanhconv2d_14/mul_2:z:0*
T0*1
_output_shapes
:�����������<T
conv2d_14/add_1/xConst*
_output_shapes
: *
dtype0*
value
B j�x�
conv2d_14/add_1AddV2conv2d_14/add_1/x:output:0conv2d_14/Tanh:y:0*
T0*1
_output_shapes
:�����������<z
conv2d_14/mul_3Mulconv2d_14/mul:z:0conv2d_14/add_1:z:0*
T0*1
_output_shapes
:�����������<�
conv2d_15/Conv2D/ReadVariableOpReadVariableOp(conv2d_15_conv2d_readvariableop_resource*&
_output_shapes
:<<*
dtype0�
conv2d_15/Conv2D/CastCast'conv2d_15/Conv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*&
_output_shapes
:<<�
conv2d_15/Conv2DConv2Dconv2d_14/mul_3:z:0conv2d_15/Conv2D/Cast:y:0*
T0*1
_output_shapes
:�����������<*
paddingSAME*
strides
�
 conv2d_15/BiasAdd/ReadVariableOpReadVariableOp)conv2d_15_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype0|
conv2d_15/BiasAdd/CastCast(conv2d_15/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:<�
conv2d_15/BiasAddBiasAddconv2d_15/Conv2D:output:0conv2d_15/BiasAdd/Cast:y:0*
T0*1
_output_shapes
:�����������<R
conv2d_15/mul/xConst*
_output_shapes
: *
dtype0*
value
B j�p�
conv2d_15/mulMulconv2d_15/mul/x:output:0conv2d_15/BiasAdd:output:0*
T0*1
_output_shapes
:�����������<S
conv2d_15/Pow/yConst*
_output_shapes
: *
dtype0*
valueB	 j���
conv2d_15/PowPowconv2d_15/BiasAdd:output:0conv2d_15/Pow/y:output:0*
T0*1
_output_shapes
:�����������<T
conv2d_15/mul_1/xConst*
_output_shapes
: *
dtype0*
value
B j�S�
conv2d_15/mul_1Mulconv2d_15/mul_1/x:output:0conv2d_15/Pow:z:0*
T0*1
_output_shapes
:�����������<�
conv2d_15/addAddV2conv2d_15/BiasAdd:output:0conv2d_15/mul_1:z:0*
T0*1
_output_shapes
:�����������<T
conv2d_15/mul_2/xConst*
_output_shapes
: *
dtype0*
value
B j�t�
conv2d_15/mul_2Mulconv2d_15/mul_2/x:output:0conv2d_15/add:z:0*
T0*1
_output_shapes
:�����������<g
conv2d_15/TanhTanhconv2d_15/mul_2:z:0*
T0*1
_output_shapes
:�����������<T
conv2d_15/add_1/xConst*
_output_shapes
: *
dtype0*
value
B j�x�
conv2d_15/add_1AddV2conv2d_15/add_1/x:output:0conv2d_15/Tanh:y:0*
T0*1
_output_shapes
:�����������<z
conv2d_15/mul_3Mulconv2d_15/mul:z:0conv2d_15/add_1:z:0*
T0*1
_output_shapes
:�����������<�
conv2d_16/Conv2D/ReadVariableOpReadVariableOp(conv2d_16_conv2d_readvariableop_resource*&
_output_shapes
:<*
dtype0�
conv2d_16/Conv2D/CastCast'conv2d_16/Conv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*&
_output_shapes
:<�
conv2d_16/Conv2DConv2Dconv2d_15/mul_3:z:0conv2d_16/Conv2D/Cast:y:0*
T0*1
_output_shapes
:�����������*
paddingSAME*
strides
�
 conv2d_16/BiasAdd/ReadVariableOpReadVariableOp)conv2d_16_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0|
conv2d_16/BiasAdd/CastCast(conv2d_16/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:�
conv2d_16/BiasAddBiasAddconv2d_16/Conv2D:output:0conv2d_16/BiasAdd/Cast:y:0*
T0*1
_output_shapes
:�����������y

add_2/CastCastconv2d_16/BiasAdd:output:0*

DstT0*

SrcT0*1
_output_shapes
:�����������f
	add_2/addAddV2add_2/Cast:y:0inputs*
T0*1
_output_shapes
:�����������f
IdentityIdentityadd_2/add:z:0^NoOp*
T0*1
_output_shapes
:������������
NoOpNoOp^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp!^conv2d_10/BiasAdd/ReadVariableOp ^conv2d_10/Conv2D/ReadVariableOp!^conv2d_11/BiasAdd/ReadVariableOp ^conv2d_11/Conv2D/ReadVariableOp!^conv2d_12/BiasAdd/ReadVariableOp ^conv2d_12/Conv2D/ReadVariableOp!^conv2d_13/BiasAdd/ReadVariableOp ^conv2d_13/Conv2D/ReadVariableOp!^conv2d_14/BiasAdd/ReadVariableOp ^conv2d_14/Conv2D/ReadVariableOp!^conv2d_15/BiasAdd/ReadVariableOp ^conv2d_15/Conv2D/ReadVariableOp!^conv2d_16/BiasAdd/ReadVariableOp ^conv2d_16/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp ^conv2d_4/BiasAdd/ReadVariableOp^conv2d_4/Conv2D/ReadVariableOp ^conv2d_5/BiasAdd/ReadVariableOp^conv2d_5/Conv2D/ReadVariableOp ^conv2d_6/BiasAdd/ReadVariableOp^conv2d_6/Conv2D/ReadVariableOp ^conv2d_7/BiasAdd/ReadVariableOp^conv2d_7/Conv2D/ReadVariableOp ^conv2d_8/BiasAdd/ReadVariableOp^conv2d_8/Conv2D/ReadVariableOp ^conv2d_9/BiasAdd/ReadVariableOp^conv2d_9/Conv2D/ReadVariableOp(^conv2d_transpose/BiasAdd/ReadVariableOp1^conv2d_transpose/conv2d_transpose/ReadVariableOp*^conv2d_transpose_1/BiasAdd/ReadVariableOp3^conv2d_transpose_1/conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*|
_input_shapesk
i:�����������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2D
 conv2d_10/BiasAdd/ReadVariableOp conv2d_10/BiasAdd/ReadVariableOp2B
conv2d_10/Conv2D/ReadVariableOpconv2d_10/Conv2D/ReadVariableOp2D
 conv2d_11/BiasAdd/ReadVariableOp conv2d_11/BiasAdd/ReadVariableOp2B
conv2d_11/Conv2D/ReadVariableOpconv2d_11/Conv2D/ReadVariableOp2D
 conv2d_12/BiasAdd/ReadVariableOp conv2d_12/BiasAdd/ReadVariableOp2B
conv2d_12/Conv2D/ReadVariableOpconv2d_12/Conv2D/ReadVariableOp2D
 conv2d_13/BiasAdd/ReadVariableOp conv2d_13/BiasAdd/ReadVariableOp2B
conv2d_13/Conv2D/ReadVariableOpconv2d_13/Conv2D/ReadVariableOp2D
 conv2d_14/BiasAdd/ReadVariableOp conv2d_14/BiasAdd/ReadVariableOp2B
conv2d_14/Conv2D/ReadVariableOpconv2d_14/Conv2D/ReadVariableOp2D
 conv2d_15/BiasAdd/ReadVariableOp conv2d_15/BiasAdd/ReadVariableOp2B
conv2d_15/Conv2D/ReadVariableOpconv2d_15/Conv2D/ReadVariableOp2D
 conv2d_16/BiasAdd/ReadVariableOp conv2d_16/BiasAdd/ReadVariableOp2B
conv2d_16/Conv2D/ReadVariableOpconv2d_16/Conv2D/ReadVariableOp2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2B
conv2d_3/BiasAdd/ReadVariableOpconv2d_3/BiasAdd/ReadVariableOp2@
conv2d_3/Conv2D/ReadVariableOpconv2d_3/Conv2D/ReadVariableOp2B
conv2d_4/BiasAdd/ReadVariableOpconv2d_4/BiasAdd/ReadVariableOp2@
conv2d_4/Conv2D/ReadVariableOpconv2d_4/Conv2D/ReadVariableOp2B
conv2d_5/BiasAdd/ReadVariableOpconv2d_5/BiasAdd/ReadVariableOp2@
conv2d_5/Conv2D/ReadVariableOpconv2d_5/Conv2D/ReadVariableOp2B
conv2d_6/BiasAdd/ReadVariableOpconv2d_6/BiasAdd/ReadVariableOp2@
conv2d_6/Conv2D/ReadVariableOpconv2d_6/Conv2D/ReadVariableOp2B
conv2d_7/BiasAdd/ReadVariableOpconv2d_7/BiasAdd/ReadVariableOp2@
conv2d_7/Conv2D/ReadVariableOpconv2d_7/Conv2D/ReadVariableOp2B
conv2d_8/BiasAdd/ReadVariableOpconv2d_8/BiasAdd/ReadVariableOp2@
conv2d_8/Conv2D/ReadVariableOpconv2d_8/Conv2D/ReadVariableOp2B
conv2d_9/BiasAdd/ReadVariableOpconv2d_9/BiasAdd/ReadVariableOp2@
conv2d_9/Conv2D/ReadVariableOpconv2d_9/Conv2D/ReadVariableOp2R
'conv2d_transpose/BiasAdd/ReadVariableOp'conv2d_transpose/BiasAdd/ReadVariableOp2d
0conv2d_transpose/conv2d_transpose/ReadVariableOp0conv2d_transpose/conv2d_transpose/ReadVariableOp2V
)conv2d_transpose_1/BiasAdd/ReadVariableOp)conv2d_transpose_1/BiasAdd/ReadVariableOp2h
2conv2d_transpose_1/conv2d_transpose/ReadVariableOp2conv2d_transpose_1/conv2d_transpose/ReadVariableOp:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
�
D__inference_conv2d_12_layer_call_and_return_conditional_losses_56214

inputs8
conv2d_readvariableop_resource:xx-
biasadd_readvariableop_resource:x
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:xx*
dtype0r
Conv2D/CastCastConv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*&
_output_shapes
:xx�
Conv2DConv2DinputsConv2D/Cast:y:0*
T0*/
_output_shapes
:���������xxx*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:x*
dtype0h
BiasAdd/CastCastBiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:xo
BiasAddBiasAddConv2D:output:0BiasAdd/Cast:y:0*
T0*/
_output_shapes
:���������xxxH
mul/xConst*
_output_shapes
: *
dtype0*
value
B j�pf
mulMulmul/x:output:0BiasAdd:output:0*
T0*/
_output_shapes
:���������xxxI
Pow/yConst*
_output_shapes
: *
dtype0*
valueB	 j��f
PowPowBiasAdd:output:0Pow/y:output:0*
T0*/
_output_shapes
:���������xxxJ
mul_1/xConst*
_output_shapes
: *
dtype0*
value
B j�Sa
mul_1Mulmul_1/x:output:0Pow:z:0*
T0*/
_output_shapes
:���������xxxc
addAddV2BiasAdd:output:0	mul_1:z:0*
T0*/
_output_shapes
:���������xxxJ
mul_2/xConst*
_output_shapes
: *
dtype0*
value
B j�ta
mul_2Mulmul_2/x:output:0add:z:0*
T0*/
_output_shapes
:���������xxxQ
TanhTanh	mul_2:z:0*
T0*/
_output_shapes
:���������xxxJ
add_1/xConst*
_output_shapes
: *
dtype0*
value
B j�xd
add_1AddV2add_1/x:output:0Tanh:y:0*
T0*/
_output_shapes
:���������xxxZ
mul_3Mulmul:z:0	add_1:z:0*
T0*/
_output_shapes
:���������xxx`
IdentityIdentity	mul_3:z:0^NoOp*
T0*/
_output_shapes
:���������xxxw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������xxx: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������xxx
 
_user_specified_nameinputs
�
�
(__inference_conv2d_2_layer_call_fn_55747

inputs!
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_conv2d_2_layer_call_and_return_conditional_losses_53248y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:�����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:�����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
W
+__inference_concatenate_layer_call_fn_55812
inputs_0
inputs_1
identity�
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_concatenate_layer_call_and_return_conditional_losses_53293j
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:�����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*M
_input_shapes<
::�����������:�����������:[ W
1
_output_shapes
:�����������
"
_user_specified_name
inputs/0:[W
1
_output_shapes
:�����������
"
_user_specified_name
inputs/1"�	L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
E
input_1:
serving_default_input_1:0�����������C
add_2:
StatefulPartitionedCall:0�����������tensorflow/serving/predict:��
�
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer-5
layer_with_weights-4
layer-6
layer_with_weights-5
layer-7
	layer-8

layer_with_weights-6

layer-9
layer_with_weights-7
layer-10
layer-11
layer_with_weights-8
layer-12
layer_with_weights-9
layer-13
layer_with_weights-10
layer-14
layer_with_weights-11
layer-15
layer_with_weights-12
layer-16
layer-17
layer_with_weights-13
layer-18
layer_with_weights-14
layer-19
layer_with_weights-15
layer-20
layer-21
layer_with_weights-16
layer-22
layer_with_weights-17
layer-23
layer_with_weights-18
layer-24
layer-25
	variables
trainable_variables
regularization_losses
	keras_api
__call__
* &call_and_return_all_conditional_losses
!_default_save_signature
"	optimizer
#
signatures"
_tf_keras_network
"
_tf_keras_input_layer
�
$	variables
%trainable_variables
&regularization_losses
'	keras_api
(__call__
*)&call_and_return_all_conditional_losses

*kernel
+bias
 ,_jit_compiled_convolution_op"
_tf_keras_layer
�
-	variables
.trainable_variables
/regularization_losses
0	keras_api
1__call__
*2&call_and_return_all_conditional_losses

3kernel
4bias
 5_jit_compiled_convolution_op"
_tf_keras_layer
�
6	variables
7trainable_variables
8regularization_losses
9	keras_api
:__call__
*;&call_and_return_all_conditional_losses

<kernel
=bias
 >_jit_compiled_convolution_op"
_tf_keras_layer
�
?	variables
@trainable_variables
Aregularization_losses
B	keras_api
C__call__
*D&call_and_return_all_conditional_losses

Ekernel
Fbias
 G_jit_compiled_convolution_op"
_tf_keras_layer
�
H	variables
Itrainable_variables
Jregularization_losses
K	keras_api
L__call__
*M&call_and_return_all_conditional_losses"
_tf_keras_layer
�
N	variables
Otrainable_variables
Pregularization_losses
Q	keras_api
R__call__
*S&call_and_return_all_conditional_losses

Tkernel
Ubias
 V_jit_compiled_convolution_op"
_tf_keras_layer
�
W	variables
Xtrainable_variables
Yregularization_losses
Z	keras_api
[__call__
*\&call_and_return_all_conditional_losses

]kernel
^bias
 __jit_compiled_convolution_op"
_tf_keras_layer
�
`	variables
atrainable_variables
bregularization_losses
c	keras_api
d__call__
*e&call_and_return_all_conditional_losses"
_tf_keras_layer
�
f	variables
gtrainable_variables
hregularization_losses
i	keras_api
j__call__
*k&call_and_return_all_conditional_losses

lkernel
mbias
 n_jit_compiled_convolution_op"
_tf_keras_layer
�
o	variables
ptrainable_variables
qregularization_losses
r	keras_api
s__call__
*t&call_and_return_all_conditional_losses

ukernel
vbias
 w_jit_compiled_convolution_op"
_tf_keras_layer
�
x	variables
ytrainable_variables
zregularization_losses
{	keras_api
|__call__
*}&call_and_return_all_conditional_losses"
_tf_keras_layer
�
~	variables
trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
*0
+1
32
43
<4
=5
E6
F7
T8
U9
]10
^11
l12
m13
u14
v15
�16
�17
�18
�19
�20
�21
�22
�23
�24
�25
�26
�27
�28
�29
�30
�31
�32
�33
�34
�35
�36
�37"
trackable_list_wrapper
�
*0
+1
32
43
<4
=5
E6
F7
T8
U9
]10
^11
l12
m13
u14
v15
�16
�17
�18
�19
�20
�21
�22
�23
�24
�25
�26
�27
�28
�29
�30
�31
�32
�33
�34
�35
�36
�37"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
__call__
!_default_save_signature
* &call_and_return_all_conditional_losses
& "call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_1
�trace_2
�trace_32�
%__inference_model_layer_call_fn_53802
%__inference_model_layer_call_fn_54737
%__inference_model_layer_call_fn_54818
%__inference_model_layer_call_fn_54351�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1z�trace_2z�trace_3
�
�trace_0
�trace_1
�trace_2
�trace_32�
@__inference_model_layer_call_and_return_conditional_losses_55244
@__inference_model_layer_call_and_return_conditional_losses_55670
@__inference_model_layer_call_and_return_conditional_losses_54459
@__inference_model_layer_call_and_return_conditional_losses_54567�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1z�trace_2z�trace_3
�B�
 __inference__wrapped_model_53011input_1"�
���
FullArgSpec
args� 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�
�
loss_scale
�base_optimizer
	�iter
�beta_1
�beta_2

�decay
�learning_rate*m�+m�3m�4m�<m�=m�Em�Fm�Tm�Um�]m�^m�lm�mm�um�vm�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�*v�+v�3v�4v�<v�=v�Ev�Fv�Tv�Uv�]v�^v�lv�mv�uv�vv�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�*vhat�+vhat�3vhat�4vhat�<vhat�=vhat�Evhat�Fvhat�Tvhat�Uvhat�]vhat�^vhat�lvhat�mvhat�uvhat�vvhat��vhat��vhat��vhat��vhat��vhat��vhat��vhat��vhat��vhat��vhat��vhat��vhat��vhat��vhat��vhat��vhat��vhat��vhat��vhat��vhat��vhat��vhat�"
	optimizer
-
�serving_default"
signature_map
.
*0
+1"
trackable_list_wrapper
.
*0
+1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
$	variables
%trainable_variables
&regularization_losses
(__call__
*)&call_and_return_all_conditional_losses
&)"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
&__inference_conv2d_layer_call_fn_55679�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
A__inference_conv2d_layer_call_and_return_conditional_losses_55704�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
':%2conv2d/kernel
:2conv2d/bias
�2��
���
FullArgSpec'
args�
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
.
30
41"
trackable_list_wrapper
.
30
41"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
-	variables
.trainable_variables
/regularization_losses
1__call__
*2&call_and_return_all_conditional_losses
&2"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
(__inference_conv2d_1_layer_call_fn_55713�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
C__inference_conv2d_1_layer_call_and_return_conditional_losses_55738�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
):'2conv2d_1/kernel
:2conv2d_1/bias
�2��
���
FullArgSpec'
args�
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
.
<0
=1"
trackable_list_wrapper
.
<0
=1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
6	variables
7trainable_variables
8regularization_losses
:__call__
*;&call_and_return_all_conditional_losses
&;"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
(__inference_conv2d_2_layer_call_fn_55747�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
C__inference_conv2d_2_layer_call_and_return_conditional_losses_55772�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
):'2conv2d_2/kernel
:2conv2d_2/bias
�2��
���
FullArgSpec'
args�
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
.
E0
F1"
trackable_list_wrapper
.
E0
F1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
?	variables
@trainable_variables
Aregularization_losses
C__call__
*D&call_and_return_all_conditional_losses
&D"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
(__inference_conv2d_3_layer_call_fn_55781�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
C__inference_conv2d_3_layer_call_and_return_conditional_losses_55806�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
):'2conv2d_3/kernel
:2conv2d_3/bias
�2��
���
FullArgSpec'
args�
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
H	variables
Itrainable_variables
Jregularization_losses
L__call__
*M&call_and_return_all_conditional_losses
&M"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
+__inference_concatenate_layer_call_fn_55812�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
F__inference_concatenate_layer_call_and_return_conditional_losses_55819�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
.
T0
U1"
trackable_list_wrapper
.
T0
U1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
N	variables
Otrainable_variables
Pregularization_losses
R__call__
*S&call_and_return_all_conditional_losses
&S"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
(__inference_conv2d_4_layer_call_fn_55828�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
C__inference_conv2d_4_layer_call_and_return_conditional_losses_55853�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
):'<2conv2d_4/kernel
:<2conv2d_4/bias
�2��
���
FullArgSpec'
args�
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
.
]0
^1"
trackable_list_wrapper
.
]0
^1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
W	variables
Xtrainable_variables
Yregularization_losses
[__call__
*\&call_and_return_all_conditional_losses
&\"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
(__inference_conv2d_5_layer_call_fn_55862�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
C__inference_conv2d_5_layer_call_and_return_conditional_losses_55887�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
):'<<2conv2d_5/kernel
:<2conv2d_5/bias
�2��
���
FullArgSpec'
args�
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
`	variables
atrainable_variables
bregularization_losses
d__call__
*e&call_and_return_all_conditional_losses
&e"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
1__inference_average_pooling2d_layer_call_fn_55892�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
L__inference_average_pooling2d_layer_call_and_return_conditional_losses_55897�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
.
l0
m1"
trackable_list_wrapper
.
l0
m1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
f	variables
gtrainable_variables
hregularization_losses
j__call__
*k&call_and_return_all_conditional_losses
&k"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
(__inference_conv2d_6_layer_call_fn_55906�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
C__inference_conv2d_6_layer_call_and_return_conditional_losses_55931�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
):'<x2conv2d_6/kernel
:x2conv2d_6/bias
�2��
���
FullArgSpec'
args�
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
.
u0
v1"
trackable_list_wrapper
.
u0
v1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
o	variables
ptrainable_variables
qregularization_losses
s__call__
*t&call_and_return_all_conditional_losses
&t"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
(__inference_conv2d_7_layer_call_fn_55940�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
C__inference_conv2d_7_layer_call_and_return_conditional_losses_55965�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
):'xx2conv2d_7/kernel
:x2conv2d_7/bias
�2��
���
FullArgSpec'
args�
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
x	variables
ytrainable_variables
zregularization_losses
|__call__
*}&call_and_return_all_conditional_losses
&}"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
3__inference_average_pooling2d_1_layer_call_fn_55970�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
N__inference_average_pooling2d_1_layer_call_and_return_conditional_losses_55975�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
~	variables
trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
(__inference_conv2d_8_layer_call_fn_55984�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
C__inference_conv2d_8_layer_call_and_return_conditional_losses_56009�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
*:(x�2conv2d_8/kernel
:�2conv2d_8/bias
�2��
���
FullArgSpec'
args�
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
(__inference_conv2d_9_layer_call_fn_56018�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
C__inference_conv2d_9_layer_call_and_return_conditional_losses_56043�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
+:)��2conv2d_9/kernel
:�2conv2d_9/bias
�2��
���
FullArgSpec'
args�
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
)__inference_conv2d_10_layer_call_fn_56052�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
D__inference_conv2d_10_layer_call_and_return_conditional_losses_56077�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
,:*��2conv2d_10/kernel
:�2conv2d_10/bias
�2��
���
FullArgSpec'
args�
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
)__inference_conv2d_11_layer_call_fn_56086�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
D__inference_conv2d_11_layer_call_and_return_conditional_losses_56111�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
,:*��2conv2d_11/kernel
:�2conv2d_11/bias
�2��
���
FullArgSpec'
args�
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
0__inference_conv2d_transpose_layer_call_fn_56120�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
K__inference_conv2d_transpose_layer_call_and_return_conditional_losses_56168�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
2:0x�2conv2d_transpose/kernel
#:!x2conv2d_transpose/bias
�2��
���
FullArgSpec'
args�
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
#__inference_add_layer_call_fn_56174�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
>__inference_add_layer_call_and_return_conditional_losses_56180�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
)__inference_conv2d_12_layer_call_fn_56189�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
D__inference_conv2d_12_layer_call_and_return_conditional_losses_56214�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
*:(xx2conv2d_12/kernel
:x2conv2d_12/bias
�2��
���
FullArgSpec'
args�
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
)__inference_conv2d_13_layer_call_fn_56223�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
D__inference_conv2d_13_layer_call_and_return_conditional_losses_56248�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
*:(xx2conv2d_13/kernel
:x2conv2d_13/bias
�2��
���
FullArgSpec'
args�
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
2__inference_conv2d_transpose_1_layer_call_fn_56257�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
M__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_56305�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
3:1<x2conv2d_transpose_1/kernel
%:#<2conv2d_transpose_1/bias
�2��
���
FullArgSpec'
args�
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
%__inference_add_1_layer_call_fn_56311�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
@__inference_add_1_layer_call_and_return_conditional_losses_56317�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
)__inference_conv2d_14_layer_call_fn_56326�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
D__inference_conv2d_14_layer_call_and_return_conditional_losses_56351�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
*:(<<2conv2d_14/kernel
:<2conv2d_14/bias
�2��
���
FullArgSpec'
args�
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
)__inference_conv2d_15_layer_call_fn_56360�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
D__inference_conv2d_15_layer_call_and_return_conditional_losses_56385�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
*:(<<2conv2d_15/kernel
:<2conv2d_15/bias
�2��
���
FullArgSpec'
args�
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
)__inference_conv2d_16_layer_call_fn_56394�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
D__inference_conv2d_16_layer_call_and_return_conditional_losses_56406�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
*:(<2conv2d_16/kernel
:2conv2d_16/bias
�2��
���
FullArgSpec'
args�
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
%__inference_add_2_layer_call_fn_56412�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
@__inference_add_2_layer_call_and_return_conditional_losses_56418�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
 "
trackable_list_wrapper
�
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25"
trackable_list_wrapper
8
�0
�1
�2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
%__inference_model_layer_call_fn_53802input_1"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
%__inference_model_layer_call_fn_54737inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
%__inference_model_layer_call_fn_54818inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
%__inference_model_layer_call_fn_54351input_1"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
@__inference_model_layer_call_and_return_conditional_losses_55244inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
@__inference_model_layer_call_and_return_conditional_losses_55670inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
@__inference_model_layer_call_and_return_conditional_losses_54459input_1"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
@__inference_model_layer_call_and_return_conditional_losses_54567input_1"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
H
�current_loss_scale
�
good_steps"
_generic_user_object
"
_generic_user_object
:	 (2cond_1/Adam/iter
: (2cond_1/Adam/beta_1
: (2cond_1/Adam/beta_2
: (2cond_1/Adam/decay
#:! (2cond_1/Adam/learning_rate
�B�
#__inference_signature_wrapper_54656input_1"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
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
�B�
&__inference_conv2d_layer_call_fn_55679inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
A__inference_conv2d_layer_call_and_return_conditional_losses_55704inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
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
�B�
(__inference_conv2d_1_layer_call_fn_55713inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_conv2d_1_layer_call_and_return_conditional_losses_55738inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
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
�B�
(__inference_conv2d_2_layer_call_fn_55747inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_conv2d_2_layer_call_and_return_conditional_losses_55772inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
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
�B�
(__inference_conv2d_3_layer_call_fn_55781inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_conv2d_3_layer_call_and_return_conditional_losses_55806inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
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
�B�
+__inference_concatenate_layer_call_fn_55812inputs/0inputs/1"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_concatenate_layer_call_and_return_conditional_losses_55819inputs/0inputs/1"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
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
�B�
(__inference_conv2d_4_layer_call_fn_55828inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_conv2d_4_layer_call_and_return_conditional_losses_55853inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
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
�B�
(__inference_conv2d_5_layer_call_fn_55862inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_conv2d_5_layer_call_and_return_conditional_losses_55887inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
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
�B�
1__inference_average_pooling2d_layer_call_fn_55892inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
L__inference_average_pooling2d_layer_call_and_return_conditional_losses_55897inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
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
�B�
(__inference_conv2d_6_layer_call_fn_55906inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_conv2d_6_layer_call_and_return_conditional_losses_55931inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
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
�B�
(__inference_conv2d_7_layer_call_fn_55940inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_conv2d_7_layer_call_and_return_conditional_losses_55965inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
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
�B�
3__inference_average_pooling2d_1_layer_call_fn_55970inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
N__inference_average_pooling2d_1_layer_call_and_return_conditional_losses_55975inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
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
�B�
(__inference_conv2d_8_layer_call_fn_55984inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_conv2d_8_layer_call_and_return_conditional_losses_56009inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
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
�B�
(__inference_conv2d_9_layer_call_fn_56018inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_conv2d_9_layer_call_and_return_conditional_losses_56043inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
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
�B�
)__inference_conv2d_10_layer_call_fn_56052inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_conv2d_10_layer_call_and_return_conditional_losses_56077inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
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
�B�
)__inference_conv2d_11_layer_call_fn_56086inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_conv2d_11_layer_call_and_return_conditional_losses_56111inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
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
�B�
0__inference_conv2d_transpose_layer_call_fn_56120inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
K__inference_conv2d_transpose_layer_call_and_return_conditional_losses_56168inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
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
�B�
#__inference_add_layer_call_fn_56174inputs/0inputs/1"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
>__inference_add_layer_call_and_return_conditional_losses_56180inputs/0inputs/1"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
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
�B�
)__inference_conv2d_12_layer_call_fn_56189inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_conv2d_12_layer_call_and_return_conditional_losses_56214inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
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
�B�
)__inference_conv2d_13_layer_call_fn_56223inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_conv2d_13_layer_call_and_return_conditional_losses_56248inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
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
�B�
2__inference_conv2d_transpose_1_layer_call_fn_56257inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
M__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_56305inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
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
�B�
%__inference_add_1_layer_call_fn_56311inputs/0inputs/1"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
@__inference_add_1_layer_call_and_return_conditional_losses_56317inputs/0inputs/1"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
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
�B�
)__inference_conv2d_14_layer_call_fn_56326inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_conv2d_14_layer_call_and_return_conditional_losses_56351inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
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
�B�
)__inference_conv2d_15_layer_call_fn_56360inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_conv2d_15_layer_call_and_return_conditional_losses_56385inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
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
�B�
)__inference_conv2d_16_layer_call_fn_56394inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_conv2d_16_layer_call_and_return_conditional_losses_56406inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
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
�B�
%__inference_add_2_layer_call_fn_56412inputs/0inputs/1"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
@__inference_add_2_layer_call_and_return_conditional_losses_56418inputs/0inputs/1"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
R
�	variables
�	keras_api

�total

�count"
_tf_keras_metric
c
�	variables
�	keras_api

�total

�count
�
_fn_kwargs"
_tf_keras_metric
c
�	variables
�	keras_api

�total

�count
�
_fn_kwargs"
_tf_keras_metric
: 2current_loss_scale
:	 2
good_steps
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
3:12cond_1/Adam/conv2d/kernel/m
%:#2cond_1/Adam/conv2d/bias/m
5:32cond_1/Adam/conv2d_1/kernel/m
':%2cond_1/Adam/conv2d_1/bias/m
5:32cond_1/Adam/conv2d_2/kernel/m
':%2cond_1/Adam/conv2d_2/bias/m
5:32cond_1/Adam/conv2d_3/kernel/m
':%2cond_1/Adam/conv2d_3/bias/m
5:3<2cond_1/Adam/conv2d_4/kernel/m
':%<2cond_1/Adam/conv2d_4/bias/m
5:3<<2cond_1/Adam/conv2d_5/kernel/m
':%<2cond_1/Adam/conv2d_5/bias/m
5:3<x2cond_1/Adam/conv2d_6/kernel/m
':%x2cond_1/Adam/conv2d_6/bias/m
5:3xx2cond_1/Adam/conv2d_7/kernel/m
':%x2cond_1/Adam/conv2d_7/bias/m
6:4x�2cond_1/Adam/conv2d_8/kernel/m
(:&�2cond_1/Adam/conv2d_8/bias/m
7:5��2cond_1/Adam/conv2d_9/kernel/m
(:&�2cond_1/Adam/conv2d_9/bias/m
8:6��2cond_1/Adam/conv2d_10/kernel/m
):'�2cond_1/Adam/conv2d_10/bias/m
8:6��2cond_1/Adam/conv2d_11/kernel/m
):'�2cond_1/Adam/conv2d_11/bias/m
>:<x�2%cond_1/Adam/conv2d_transpose/kernel/m
/:-x2#cond_1/Adam/conv2d_transpose/bias/m
6:4xx2cond_1/Adam/conv2d_12/kernel/m
(:&x2cond_1/Adam/conv2d_12/bias/m
6:4xx2cond_1/Adam/conv2d_13/kernel/m
(:&x2cond_1/Adam/conv2d_13/bias/m
?:=<x2'cond_1/Adam/conv2d_transpose_1/kernel/m
1:/<2%cond_1/Adam/conv2d_transpose_1/bias/m
6:4<<2cond_1/Adam/conv2d_14/kernel/m
(:&<2cond_1/Adam/conv2d_14/bias/m
6:4<<2cond_1/Adam/conv2d_15/kernel/m
(:&<2cond_1/Adam/conv2d_15/bias/m
6:4<2cond_1/Adam/conv2d_16/kernel/m
(:&2cond_1/Adam/conv2d_16/bias/m
3:12cond_1/Adam/conv2d/kernel/v
%:#2cond_1/Adam/conv2d/bias/v
5:32cond_1/Adam/conv2d_1/kernel/v
':%2cond_1/Adam/conv2d_1/bias/v
5:32cond_1/Adam/conv2d_2/kernel/v
':%2cond_1/Adam/conv2d_2/bias/v
5:32cond_1/Adam/conv2d_3/kernel/v
':%2cond_1/Adam/conv2d_3/bias/v
5:3<2cond_1/Adam/conv2d_4/kernel/v
':%<2cond_1/Adam/conv2d_4/bias/v
5:3<<2cond_1/Adam/conv2d_5/kernel/v
':%<2cond_1/Adam/conv2d_5/bias/v
5:3<x2cond_1/Adam/conv2d_6/kernel/v
':%x2cond_1/Adam/conv2d_6/bias/v
5:3xx2cond_1/Adam/conv2d_7/kernel/v
':%x2cond_1/Adam/conv2d_7/bias/v
6:4x�2cond_1/Adam/conv2d_8/kernel/v
(:&�2cond_1/Adam/conv2d_8/bias/v
7:5��2cond_1/Adam/conv2d_9/kernel/v
(:&�2cond_1/Adam/conv2d_9/bias/v
8:6��2cond_1/Adam/conv2d_10/kernel/v
):'�2cond_1/Adam/conv2d_10/bias/v
8:6��2cond_1/Adam/conv2d_11/kernel/v
):'�2cond_1/Adam/conv2d_11/bias/v
>:<x�2%cond_1/Adam/conv2d_transpose/kernel/v
/:-x2#cond_1/Adam/conv2d_transpose/bias/v
6:4xx2cond_1/Adam/conv2d_12/kernel/v
(:&x2cond_1/Adam/conv2d_12/bias/v
6:4xx2cond_1/Adam/conv2d_13/kernel/v
(:&x2cond_1/Adam/conv2d_13/bias/v
?:=<x2'cond_1/Adam/conv2d_transpose_1/kernel/v
1:/<2%cond_1/Adam/conv2d_transpose_1/bias/v
6:4<<2cond_1/Adam/conv2d_14/kernel/v
(:&<2cond_1/Adam/conv2d_14/bias/v
6:4<<2cond_1/Adam/conv2d_15/kernel/v
(:&<2cond_1/Adam/conv2d_15/bias/v
6:4<2cond_1/Adam/conv2d_16/kernel/v
(:&2cond_1/Adam/conv2d_16/bias/v
6:42cond_1/Adam/conv2d/kernel/vhat
(:&2cond_1/Adam/conv2d/bias/vhat
8:62 cond_1/Adam/conv2d_1/kernel/vhat
*:(2cond_1/Adam/conv2d_1/bias/vhat
8:62 cond_1/Adam/conv2d_2/kernel/vhat
*:(2cond_1/Adam/conv2d_2/bias/vhat
8:62 cond_1/Adam/conv2d_3/kernel/vhat
*:(2cond_1/Adam/conv2d_3/bias/vhat
8:6<2 cond_1/Adam/conv2d_4/kernel/vhat
*:(<2cond_1/Adam/conv2d_4/bias/vhat
8:6<<2 cond_1/Adam/conv2d_5/kernel/vhat
*:(<2cond_1/Adam/conv2d_5/bias/vhat
8:6<x2 cond_1/Adam/conv2d_6/kernel/vhat
*:(x2cond_1/Adam/conv2d_6/bias/vhat
8:6xx2 cond_1/Adam/conv2d_7/kernel/vhat
*:(x2cond_1/Adam/conv2d_7/bias/vhat
9:7x�2 cond_1/Adam/conv2d_8/kernel/vhat
+:)�2cond_1/Adam/conv2d_8/bias/vhat
::8��2 cond_1/Adam/conv2d_9/kernel/vhat
+:)�2cond_1/Adam/conv2d_9/bias/vhat
;:9��2!cond_1/Adam/conv2d_10/kernel/vhat
,:*�2cond_1/Adam/conv2d_10/bias/vhat
;:9��2!cond_1/Adam/conv2d_11/kernel/vhat
,:*�2cond_1/Adam/conv2d_11/bias/vhat
A:?x�2(cond_1/Adam/conv2d_transpose/kernel/vhat
2:0x2&cond_1/Adam/conv2d_transpose/bias/vhat
9:7xx2!cond_1/Adam/conv2d_12/kernel/vhat
+:)x2cond_1/Adam/conv2d_12/bias/vhat
9:7xx2!cond_1/Adam/conv2d_13/kernel/vhat
+:)x2cond_1/Adam/conv2d_13/bias/vhat
B:@<x2*cond_1/Adam/conv2d_transpose_1/kernel/vhat
4:2<2(cond_1/Adam/conv2d_transpose_1/bias/vhat
9:7<<2!cond_1/Adam/conv2d_14/kernel/vhat
+:)<2cond_1/Adam/conv2d_14/bias/vhat
9:7<<2!cond_1/Adam/conv2d_15/kernel/vhat
+:)<2cond_1/Adam/conv2d_15/bias/vhat
9:7<2!cond_1/Adam/conv2d_16/kernel/vhat
+:)2cond_1/Adam/conv2d_16/bias/vhat�
 __inference__wrapped_model_53011�<*+34<=EFTU]^lmuv����������������������:�7
0�-
+�(
input_1�����������
� "7�4
2
add_2)�&
add_2������������
@__inference_add_1_layer_call_and_return_conditional_losses_56317�n�k
d�a
_�\
,�)
inputs/0�����������<
,�)
inputs/1�����������<
� "/�,
%�"
0�����������<
� �
%__inference_add_1_layer_call_fn_56311�n�k
d�a
_�\
,�)
inputs/0�����������<
,�)
inputs/1�����������<
� ""������������<�
@__inference_add_2_layer_call_and_return_conditional_losses_56418�n�k
d�a
_�\
,�)
inputs/0�����������
,�)
inputs/1�����������
� "/�,
%�"
0�����������
� �
%__inference_add_2_layer_call_fn_56412�n�k
d�a
_�\
,�)
inputs/0�����������
,�)
inputs/1�����������
� ""�������������
>__inference_add_layer_call_and_return_conditional_losses_56180�j�g
`�]
[�X
*�'
inputs/0���������xxx
*�'
inputs/1���������xxx
� "-�*
#� 
0���������xxx
� �
#__inference_add_layer_call_fn_56174�j�g
`�]
[�X
*�'
inputs/0���������xxx
*�'
inputs/1���������xxx
� " ����������xxx�
N__inference_average_pooling2d_1_layer_call_and_return_conditional_losses_55975�R�O
H�E
C�@
inputs4������������������������������������
� "H�E
>�;
04������������������������������������
� �
3__inference_average_pooling2d_1_layer_call_fn_55970�R�O
H�E
C�@
inputs4������������������������������������
� ";�84�������������������������������������
L__inference_average_pooling2d_layer_call_and_return_conditional_losses_55897�R�O
H�E
C�@
inputs4������������������������������������
� "H�E
>�;
04������������������������������������
� �
1__inference_average_pooling2d_layer_call_fn_55892�R�O
H�E
C�@
inputs4������������������������������������
� ";�84�������������������������������������
F__inference_concatenate_layer_call_and_return_conditional_losses_55819�n�k
d�a
_�\
,�)
inputs/0�����������
,�)
inputs/1�����������
� "/�,
%�"
0�����������
� �
+__inference_concatenate_layer_call_fn_55812�n�k
d�a
_�\
,�)
inputs/0�����������
,�)
inputs/1�����������
� ""�������������
D__inference_conv2d_10_layer_call_and_return_conditional_losses_56077p��8�5
.�+
)�&
inputs���������<<�
� ".�+
$�!
0���������<<�
� �
)__inference_conv2d_10_layer_call_fn_56052c��8�5
.�+
)�&
inputs���������<<�
� "!����������<<��
D__inference_conv2d_11_layer_call_and_return_conditional_losses_56111p��8�5
.�+
)�&
inputs���������<<�
� ".�+
$�!
0���������<<�
� �
)__inference_conv2d_11_layer_call_fn_56086c��8�5
.�+
)�&
inputs���������<<�
� "!����������<<��
D__inference_conv2d_12_layer_call_and_return_conditional_losses_56214n��7�4
-�*
(�%
inputs���������xxx
� "-�*
#� 
0���������xxx
� �
)__inference_conv2d_12_layer_call_fn_56189a��7�4
-�*
(�%
inputs���������xxx
� " ����������xxx�
D__inference_conv2d_13_layer_call_and_return_conditional_losses_56248n��7�4
-�*
(�%
inputs���������xxx
� "-�*
#� 
0���������xxx
� �
)__inference_conv2d_13_layer_call_fn_56223a��7�4
-�*
(�%
inputs���������xxx
� " ����������xxx�
D__inference_conv2d_14_layer_call_and_return_conditional_losses_56351r��9�6
/�,
*�'
inputs�����������<
� "/�,
%�"
0�����������<
� �
)__inference_conv2d_14_layer_call_fn_56326e��9�6
/�,
*�'
inputs�����������<
� ""������������<�
D__inference_conv2d_15_layer_call_and_return_conditional_losses_56385r��9�6
/�,
*�'
inputs�����������<
� "/�,
%�"
0�����������<
� �
)__inference_conv2d_15_layer_call_fn_56360e��9�6
/�,
*�'
inputs�����������<
� ""������������<�
D__inference_conv2d_16_layer_call_and_return_conditional_losses_56406r��9�6
/�,
*�'
inputs�����������<
� "/�,
%�"
0�����������
� �
)__inference_conv2d_16_layer_call_fn_56394e��9�6
/�,
*�'
inputs�����������<
� ""�������������
C__inference_conv2d_1_layer_call_and_return_conditional_losses_55738p349�6
/�,
*�'
inputs�����������
� "/�,
%�"
0�����������
� �
(__inference_conv2d_1_layer_call_fn_55713c349�6
/�,
*�'
inputs�����������
� ""�������������
C__inference_conv2d_2_layer_call_and_return_conditional_losses_55772p<=9�6
/�,
*�'
inputs�����������
� "/�,
%�"
0�����������
� �
(__inference_conv2d_2_layer_call_fn_55747c<=9�6
/�,
*�'
inputs�����������
� ""�������������
C__inference_conv2d_3_layer_call_and_return_conditional_losses_55806pEF9�6
/�,
*�'
inputs�����������
� "/�,
%�"
0�����������
� �
(__inference_conv2d_3_layer_call_fn_55781cEF9�6
/�,
*�'
inputs�����������
� ""�������������
C__inference_conv2d_4_layer_call_and_return_conditional_losses_55853pTU9�6
/�,
*�'
inputs�����������
� "/�,
%�"
0�����������<
� �
(__inference_conv2d_4_layer_call_fn_55828cTU9�6
/�,
*�'
inputs�����������
� ""������������<�
C__inference_conv2d_5_layer_call_and_return_conditional_losses_55887p]^9�6
/�,
*�'
inputs�����������<
� "/�,
%�"
0�����������<
� �
(__inference_conv2d_5_layer_call_fn_55862c]^9�6
/�,
*�'
inputs�����������<
� ""������������<�
C__inference_conv2d_6_layer_call_and_return_conditional_losses_55931llm7�4
-�*
(�%
inputs���������xx<
� "-�*
#� 
0���������xxx
� �
(__inference_conv2d_6_layer_call_fn_55906_lm7�4
-�*
(�%
inputs���������xx<
� " ����������xxx�
C__inference_conv2d_7_layer_call_and_return_conditional_losses_55965luv7�4
-�*
(�%
inputs���������xxx
� "-�*
#� 
0���������xxx
� �
(__inference_conv2d_7_layer_call_fn_55940_uv7�4
-�*
(�%
inputs���������xxx
� " ����������xxx�
C__inference_conv2d_8_layer_call_and_return_conditional_losses_56009o��7�4
-�*
(�%
inputs���������<<x
� ".�+
$�!
0���������<<�
� �
(__inference_conv2d_8_layer_call_fn_55984b��7�4
-�*
(�%
inputs���������<<x
� "!����������<<��
C__inference_conv2d_9_layer_call_and_return_conditional_losses_56043p��8�5
.�+
)�&
inputs���������<<�
� ".�+
$�!
0���������<<�
� �
(__inference_conv2d_9_layer_call_fn_56018c��8�5
.�+
)�&
inputs���������<<�
� "!����������<<��
A__inference_conv2d_layer_call_and_return_conditional_losses_55704p*+9�6
/�,
*�'
inputs�����������
� "/�,
%�"
0�����������
� �
&__inference_conv2d_layer_call_fn_55679c*+9�6
/�,
*�'
inputs�����������
� ""�������������
M__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_56305���I�F
?�<
:�7
inputs+���������������������������x
� "?�<
5�2
0+���������������������������<
� �
2__inference_conv2d_transpose_1_layer_call_fn_56257���I�F
?�<
:�7
inputs+���������������������������x
� "2�/+���������������������������<�
K__inference_conv2d_transpose_layer_call_and_return_conditional_losses_56168���J�G
@�=
;�8
inputs,����������������������������
� "?�<
5�2
0+���������������������������x
� �
0__inference_conv2d_transpose_layer_call_fn_56120���J�G
@�=
;�8
inputs,����������������������������
� "2�/+���������������������������x�
@__inference_model_layer_call_and_return_conditional_losses_54459�<*+34<=EFTU]^lmuv����������������������B�?
8�5
+�(
input_1�����������
p 

 
� "/�,
%�"
0�����������
� �
@__inference_model_layer_call_and_return_conditional_losses_54567�<*+34<=EFTU]^lmuv����������������������B�?
8�5
+�(
input_1�����������
p

 
� "/�,
%�"
0�����������
� �
@__inference_model_layer_call_and_return_conditional_losses_55244�<*+34<=EFTU]^lmuv����������������������A�>
7�4
*�'
inputs�����������
p 

 
� "/�,
%�"
0�����������
� �
@__inference_model_layer_call_and_return_conditional_losses_55670�<*+34<=EFTU]^lmuv����������������������A�>
7�4
*�'
inputs�����������
p

 
� "/�,
%�"
0�����������
� �
%__inference_model_layer_call_fn_53802�<*+34<=EFTU]^lmuv����������������������B�?
8�5
+�(
input_1�����������
p 

 
� ""�������������
%__inference_model_layer_call_fn_54351�<*+34<=EFTU]^lmuv����������������������B�?
8�5
+�(
input_1�����������
p

 
� ""�������������
%__inference_model_layer_call_fn_54737�<*+34<=EFTU]^lmuv����������������������A�>
7�4
*�'
inputs�����������
p 

 
� ""�������������
%__inference_model_layer_call_fn_54818�<*+34<=EFTU]^lmuv����������������������A�>
7�4
*�'
inputs�����������
p

 
� ""�������������
#__inference_signature_wrapper_54656�<*+34<=EFTU]^lmuv����������������������E�B
� 
;�8
6
input_1+�(
input_1�����������"7�4
2
add_2)�&
add_2�����������