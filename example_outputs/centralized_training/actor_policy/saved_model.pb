��
��
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
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
A
BroadcastArgs
s0"T
s1"T
r0"T"
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
>
Maximum
x"T
y"T
z"T"
Ttype:
2	
�
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( �
>
Minimum
x"T
y"T
z"T"
Ttype:
2	
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
�
PartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
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
 �"serve*2.11.02v2.11.0-rc2-17-gd5b57ca93e58��
�
ActorNetwork/action/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*)
shared_nameActorNetwork/action/bias
�
,ActorNetwork/action/bias/Read/ReadVariableOpReadVariableOpActorNetwork/action/bias*
_output_shapes
:	*
dtype0
�
ActorNetwork/action/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d	*+
shared_nameActorNetwork/action/kernel
�
.ActorNetwork/action/kernel/Read/ReadVariableOpReadVariableOpActorNetwork/action/kernel*
_output_shapes

:d	*
dtype0
�
!ActorNetwork/input_mlp/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*2
shared_name#!ActorNetwork/input_mlp/dense/bias
�
5ActorNetwork/input_mlp/dense/bias/Read/ReadVariableOpReadVariableOp!ActorNetwork/input_mlp/dense/bias*
_output_shapes
:d*
dtype0
�
#ActorNetwork/input_mlp/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:dd*4
shared_name%#ActorNetwork/input_mlp/dense/kernel
�
7ActorNetwork/input_mlp/dense/kernel/Read/ReadVariableOpReadVariableOp#ActorNetwork/input_mlp/dense/kernel*
_output_shapes

:dd*
dtype0
�
#ActorNetwork/input_mlp/dense/bias_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:d*4
shared_name%#ActorNetwork/input_mlp/dense/bias_1
�
7ActorNetwork/input_mlp/dense/bias_1/Read/ReadVariableOpReadVariableOp#ActorNetwork/input_mlp/dense/bias_1*
_output_shapes
:d*
dtype0
�
%ActorNetwork/input_mlp/dense/kernel_1VarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*6
shared_name'%ActorNetwork/input_mlp/dense/kernel_1
�
9ActorNetwork/input_mlp/dense/kernel_1/Read/ReadVariableOpReadVariableOp%ActorNetwork/input_mlp/dense/kernel_1*
_output_shapes

:d*
dtype0
d
VariableVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name
Variable
]
Variable/Read/ReadVariableOpReadVariableOpVariable*
_output_shapes
: *
dtype0	
l
action_0_discountPlaceholder*#
_output_shapes
:���������*
dtype0*
shape:���������
w
action_0_observationPlaceholder*'
_output_shapes
:���������*
dtype0*
shape:���������
j
action_0_rewardPlaceholder*#
_output_shapes
:���������*
dtype0*
shape:���������
m
action_0_step_typePlaceholder*#
_output_shapes
:���������*
dtype0*
shape:���������
�
StatefulPartitionedCallStatefulPartitionedCallaction_0_discountaction_0_observationaction_0_rewardaction_0_step_type%ActorNetwork/input_mlp/dense/kernel_1#ActorNetwork/input_mlp/dense/bias_1#ActorNetwork/input_mlp/dense/kernel!ActorNetwork/input_mlp/dense/biasActorNetwork/action/kernelActorNetwork/action/bias*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������	*(
_read_only_resource_inputs

	*0
config_proto 

CPU

GPU2*0J 8� */
f*R(
&__inference_signature_wrapper_41993871
]
get_initial_state_batch_sizePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
PartitionedCallPartitionedCallget_initial_state_batch_size*
Tin
2*

Tout
 *
_collective_manager_ids
 * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� */
f*R(
&__inference_signature_wrapper_41993876
�
PartitionedCall_1PartitionedCall*	
Tin
 *

Tout
 *
_collective_manager_ids
 * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� */
f*R(
&__inference_signature_wrapper_41993888
�
StatefulPartitionedCall_1StatefulPartitionedCallVariable*
Tin
2*
Tout
2	*
_collective_manager_ids
 *
_output_shapes
: *#
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� */
f*R(
&__inference_signature_wrapper_41993884

NoOpNoOp
�
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�
value�B� B�
�

train_step
metadata
model_variables
_all_assets

action
distribution
get_initial_state
get_metadata
	get_train_step


signatures*
GA
VARIABLE_VALUEVariable%train_step/.ATTRIBUTES/VARIABLE_VALUE*
* 
.
0
1
2
3
4
5*

_actor_network*

trace_0
trace_1* 

trace_0* 

trace_0* 
* 
* 
K

action
get_initial_state
get_train_step
get_metadata* 
ke
VARIABLE_VALUE%ActorNetwork/input_mlp/dense/kernel_1,model_variables/0/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUE#ActorNetwork/input_mlp/dense/bias_1,model_variables/1/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUE#ActorNetwork/input_mlp/dense/kernel,model_variables/2/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUE!ActorNetwork/input_mlp/dense/bias,model_variables/3/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEActorNetwork/action/kernel,model_variables/4/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEActorNetwork/action/bias,model_variables/5/.ATTRIBUTES/VARIABLE_VALUE*
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
 _mlp_layers*
* 
* 
* 
* 
* 
* 
* 
* 
.
0
1
2
3
4
5*
.
0
1
2
3
4
5*
* 
�
!non_trainable_variables

"layers
#metrics
$layer_regularization_losses
%layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
 
&0
'1
(2
)3*
* 
 
&0
'1
(2
)3*
* 
* 
* 
�
*	variables
+trainable_variables
,regularization_losses
-	keras_api
.__call__
*/&call_and_return_all_conditional_losses* 
�
0	variables
1trainable_variables
2regularization_losses
3	keras_api
4__call__
*5&call_and_return_all_conditional_losses

kernel
bias*
�
6	variables
7trainable_variables
8regularization_losses
9	keras_api
:__call__
*;&call_and_return_all_conditional_losses

kernel
bias*
�
<	variables
=trainable_variables
>regularization_losses
?	keras_api
@__call__
*A&call_and_return_all_conditional_losses

kernel
bias*
* 
* 
* 
�
Bnon_trainable_variables

Clayers
Dmetrics
Elayer_regularization_losses
Flayer_metrics
*	variables
+trainable_variables
,regularization_losses
.__call__
*/&call_and_return_all_conditional_losses
&/"call_and_return_conditional_losses* 
* 
* 

0
1*

0
1*
* 
�
Gnon_trainable_variables

Hlayers
Imetrics
Jlayer_regularization_losses
Klayer_metrics
0	variables
1trainable_variables
2regularization_losses
4__call__
*5&call_and_return_all_conditional_losses
&5"call_and_return_conditional_losses*
* 
* 

0
1*

0
1*
* 
�
Lnon_trainable_variables

Mlayers
Nmetrics
Olayer_regularization_losses
Player_metrics
6	variables
7trainable_variables
8regularization_losses
:__call__
*;&call_and_return_all_conditional_losses
&;"call_and_return_conditional_losses*
* 
* 

0
1*

0
1*
* 
�
Qnon_trainable_variables

Rlayers
Smetrics
Tlayer_regularization_losses
Ulayer_metrics
<	variables
=trainable_variables
>regularization_losses
@__call__
*A&call_and_return_all_conditional_losses
&A"call_and_return_conditional_losses*
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
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameVariable/Read/ReadVariableOp9ActorNetwork/input_mlp/dense/kernel_1/Read/ReadVariableOp7ActorNetwork/input_mlp/dense/bias_1/Read/ReadVariableOp7ActorNetwork/input_mlp/dense/kernel/Read/ReadVariableOp5ActorNetwork/input_mlp/dense/bias/Read/ReadVariableOp.ActorNetwork/action/kernel/Read/ReadVariableOp,ActorNetwork/action/bias/Read/ReadVariableOpConst*
Tin
2		*
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
!__inference__traced_save_41993937
�
StatefulPartitionedCall_3StatefulPartitionedCallsaver_filenameVariable%ActorNetwork/input_mlp/dense/kernel_1#ActorNetwork/input_mlp/dense/bias_1#ActorNetwork/input_mlp/dense/kernel!ActorNetwork/input_mlp/dense/biasActorNetwork/action/kernelActorNetwork/action/bias*
Tin

2*
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
GPU2*0J 8� *-
f(R&
$__inference__traced_restore_41993968�
�O
�
(__inference_polymorphic_action_fn_294286
time_step_step_type
time_step_reward
time_step_discount
time_step_observationM
;actornetwork_input_mlp_dense_matmul_readvariableop_resource:dJ
<actornetwork_input_mlp_dense_biasadd_readvariableop_resource:dO
=actornetwork_input_mlp_dense_matmul_1_readvariableop_resource:ddL
>actornetwork_input_mlp_dense_biasadd_1_readvariableop_resource:dD
2actornetwork_action_matmul_readvariableop_resource:d	A
3actornetwork_action_biasadd_readvariableop_resource:	
identity��*ActorNetwork/action/BiasAdd/ReadVariableOp�)ActorNetwork/action/MatMul/ReadVariableOp�3ActorNetwork/input_mlp/dense/BiasAdd/ReadVariableOp�5ActorNetwork/input_mlp/dense/BiasAdd_1/ReadVariableOp�2ActorNetwork/input_mlp/dense/MatMul/ReadVariableOp�4ActorNetwork/input_mlp/dense/MatMul_1/ReadVariableOpm
ActorNetwork/flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"����   �
ActorNetwork/flatten_3/ReshapeReshapetime_step_observation%ActorNetwork/flatten_3/Const:output:0*
T0*'
_output_shapes
:����������
2ActorNetwork/input_mlp/dense/MatMul/ReadVariableOpReadVariableOp;actornetwork_input_mlp_dense_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0�
#ActorNetwork/input_mlp/dense/MatMulMatMul'ActorNetwork/flatten_3/Reshape:output:0:ActorNetwork/input_mlp/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d�
3ActorNetwork/input_mlp/dense/BiasAdd/ReadVariableOpReadVariableOp<actornetwork_input_mlp_dense_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0�
$ActorNetwork/input_mlp/dense/BiasAddBiasAdd-ActorNetwork/input_mlp/dense/MatMul:product:0;ActorNetwork/input_mlp/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d�
!ActorNetwork/input_mlp/dense/ReluRelu-ActorNetwork/input_mlp/dense/BiasAdd:output:0*
T0*'
_output_shapes
:���������d�
4ActorNetwork/input_mlp/dense/MatMul_1/ReadVariableOpReadVariableOp=actornetwork_input_mlp_dense_matmul_1_readvariableop_resource*
_output_shapes

:dd*
dtype0�
%ActorNetwork/input_mlp/dense/MatMul_1MatMul/ActorNetwork/input_mlp/dense/Relu:activations:0<ActorNetwork/input_mlp/dense/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d�
5ActorNetwork/input_mlp/dense/BiasAdd_1/ReadVariableOpReadVariableOp>actornetwork_input_mlp_dense_biasadd_1_readvariableop_resource*
_output_shapes
:d*
dtype0�
&ActorNetwork/input_mlp/dense/BiasAdd_1BiasAdd/ActorNetwork/input_mlp/dense/MatMul_1:product:0=ActorNetwork/input_mlp/dense/BiasAdd_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d�
#ActorNetwork/input_mlp/dense/Relu_1Relu/ActorNetwork/input_mlp/dense/BiasAdd_1:output:0*
T0*'
_output_shapes
:���������d�
)ActorNetwork/action/MatMul/ReadVariableOpReadVariableOp2actornetwork_action_matmul_readvariableop_resource*
_output_shapes

:d	*
dtype0�
ActorNetwork/action/MatMulMatMul1ActorNetwork/input_mlp/dense/Relu_1:activations:01ActorNetwork/action/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������	�
*ActorNetwork/action/BiasAdd/ReadVariableOpReadVariableOp3actornetwork_action_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype0�
ActorNetwork/action/BiasAddBiasAdd$ActorNetwork/action/MatMul:product:02ActorNetwork/action/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������	x
ActorNetwork/action/TanhTanh$ActorNetwork/action/BiasAdd:output:0*
T0*'
_output_shapes
:���������	k
ActorNetwork/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"����	   �
ActorNetwork/ReshapeReshapeActorNetwork/action/Tanh:y:0#ActorNetwork/Reshape/shape:output:0*
T0*'
_output_shapes
:���������	W
ActorNetwork/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
ActorNetwork/mulMulActorNetwork/mul/x:output:0ActorNetwork/Reshape:output:0*
T0*'
_output_shapes
:���������	W
ActorNetwork/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?~
ActorNetwork/addAddV2ActorNetwork/add/x:output:0ActorNetwork/mul:z:0*
T0*'
_output_shapes
:���������	W
Deterministic/atolConst*
_output_shapes
: *
dtype0*
valueB
 *    W
Deterministic/rtolConst*
_output_shapes
: *
dtype0*
valueB
 *    d
!Deterministic/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB ^
Deterministic/sample/ShapeShapeActorNetwork/add:z:0*
T0*
_output_shapes
:\
Deterministic/sample/ConstConst*
_output_shapes
: *
dtype0*
value	B : r
(Deterministic/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*Deterministic/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*Deterministic/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
"Deterministic/sample/strided_sliceStridedSlice#Deterministic/sample/Shape:output:01Deterministic/sample/strided_slice/stack:output:03Deterministic/sample/strided_slice/stack_1:output:03Deterministic/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskh
%Deterministic/sample/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB j
'Deterministic/sample/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB �
"Deterministic/sample/BroadcastArgsBroadcastArgs0Deterministic/sample/BroadcastArgs/s0_1:output:0+Deterministic/sample/strided_slice:output:0*
_output_shapes
:n
$Deterministic/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:g
$Deterministic/sample/concat/values_2Const*
_output_shapes
: *
dtype0*
valueB b
 Deterministic/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Deterministic/sample/concatConcatV2-Deterministic/sample/concat/values_0:output:0'Deterministic/sample/BroadcastArgs:r0:0-Deterministic/sample/concat/values_2:output:0)Deterministic/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:�
 Deterministic/sample/BroadcastToBroadcastToActorNetwork/add:z:0$Deterministic/sample/concat:output:0*
T0*+
_output_shapes
:���������	u
Deterministic/sample/Shape_1Shape)Deterministic/sample/BroadcastTo:output:0*
T0*
_output_shapes
:t
*Deterministic/sample/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:v
,Deterministic/sample/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: v
,Deterministic/sample/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
$Deterministic/sample/strided_slice_1StridedSlice%Deterministic/sample/Shape_1:output:03Deterministic/sample/strided_slice_1/stack:output:05Deterministic/sample/strided_slice_1/stack_1:output:05Deterministic/sample/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_maskd
"Deterministic/sample/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Deterministic/sample/concat_1ConcatV2*Deterministic/sample/sample_shape:output:0-Deterministic/sample/strided_slice_1:output:0+Deterministic/sample/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
Deterministic/sample/ReshapeReshape)Deterministic/sample/BroadcastTo:output:0&Deterministic/sample/concat_1:output:0*
T0*'
_output_shapes
:���������	\
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
clip_by_value/MinimumMinimum%Deterministic/sample/Reshape:output:0 clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:���������	T
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*'
_output_shapes
:���������	`
IdentityIdentityclip_by_value:z:0^NoOp*
T0*'
_output_shapes
:���������	�
NoOpNoOp+^ActorNetwork/action/BiasAdd/ReadVariableOp*^ActorNetwork/action/MatMul/ReadVariableOp4^ActorNetwork/input_mlp/dense/BiasAdd/ReadVariableOp6^ActorNetwork/input_mlp/dense/BiasAdd_1/ReadVariableOp3^ActorNetwork/input_mlp/dense/MatMul/ReadVariableOp5^ActorNetwork/input_mlp/dense/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*_
_input_shapesN
L:���������:���������:���������:���������: : : : : : 2X
*ActorNetwork/action/BiasAdd/ReadVariableOp*ActorNetwork/action/BiasAdd/ReadVariableOp2V
)ActorNetwork/action/MatMul/ReadVariableOp)ActorNetwork/action/MatMul/ReadVariableOp2j
3ActorNetwork/input_mlp/dense/BiasAdd/ReadVariableOp3ActorNetwork/input_mlp/dense/BiasAdd/ReadVariableOp2n
5ActorNetwork/input_mlp/dense/BiasAdd_1/ReadVariableOp5ActorNetwork/input_mlp/dense/BiasAdd_1/ReadVariableOp2h
2ActorNetwork/input_mlp/dense/MatMul/ReadVariableOp2ActorNetwork/input_mlp/dense/MatMul/ReadVariableOp2l
4ActorNetwork/input_mlp/dense/MatMul_1/ReadVariableOp4ActorNetwork/input_mlp/dense/MatMul_1/ReadVariableOp:X T
#
_output_shapes
:���������
-
_user_specified_nametime_step_step_type:UQ
#
_output_shapes
:���������
*
_user_specified_nametime_step_reward:WS
#
_output_shapes
:���������
,
_user_specified_nametime_step_discount:^Z
'
_output_shapes
:���������
/
_user_specified_nametime_step_observation
�#
�
$__inference__traced_restore_41993968
file_prefix#
assignvariableop_variable:	 J
8assignvariableop_1_actornetwork_input_mlp_dense_kernel_1:dD
6assignvariableop_2_actornetwork_input_mlp_dense_bias_1:dH
6assignvariableop_3_actornetwork_input_mlp_dense_kernel:ddB
4assignvariableop_4_actornetwork_input_mlp_dense_bias:d?
-assignvariableop_5_actornetwork_action_kernel:d	9
+assignvariableop_6_actornetwork_action_bias:	

identity_8��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_2�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B%train_step/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/0/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/1/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/2/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/3/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/4/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/5/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*#
valueBB B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*4
_output_shapes"
 ::::::::*
dtypes

2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOpAssignVariableOpassignvariableop_variableIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOp8assignvariableop_1_actornetwork_input_mlp_dense_kernel_1Identity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp6assignvariableop_2_actornetwork_input_mlp_dense_bias_1Identity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOp6assignvariableop_3_actornetwork_input_mlp_dense_kernelIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp4assignvariableop_4_actornetwork_input_mlp_dense_biasIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOp-assignvariableop_5_actornetwork_action_kernelIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOp+assignvariableop_6_actornetwork_action_biasIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 �

Identity_7Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^NoOp"/device:CPU:0*
T0*
_output_shapes
: U

Identity_8IdentityIdentity_7:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6*"
_acd_function_control_output(*
_output_shapes
 "!

identity_8Identity_8:output:0*#
_input_shapes
: : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_6:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�O
�
(__inference_polymorphic_action_fn_294085
	time_step
time_step_1
time_step_2
time_step_3M
;actornetwork_input_mlp_dense_matmul_readvariableop_resource:dJ
<actornetwork_input_mlp_dense_biasadd_readvariableop_resource:dO
=actornetwork_input_mlp_dense_matmul_1_readvariableop_resource:ddL
>actornetwork_input_mlp_dense_biasadd_1_readvariableop_resource:dD
2actornetwork_action_matmul_readvariableop_resource:d	A
3actornetwork_action_biasadd_readvariableop_resource:	
identity��*ActorNetwork/action/BiasAdd/ReadVariableOp�)ActorNetwork/action/MatMul/ReadVariableOp�3ActorNetwork/input_mlp/dense/BiasAdd/ReadVariableOp�5ActorNetwork/input_mlp/dense/BiasAdd_1/ReadVariableOp�2ActorNetwork/input_mlp/dense/MatMul/ReadVariableOp�4ActorNetwork/input_mlp/dense/MatMul_1/ReadVariableOpm
ActorNetwork/flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"����   �
ActorNetwork/flatten_3/ReshapeReshapetime_step_3%ActorNetwork/flatten_3/Const:output:0*
T0*'
_output_shapes
:����������
2ActorNetwork/input_mlp/dense/MatMul/ReadVariableOpReadVariableOp;actornetwork_input_mlp_dense_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0�
#ActorNetwork/input_mlp/dense/MatMulMatMul'ActorNetwork/flatten_3/Reshape:output:0:ActorNetwork/input_mlp/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d�
3ActorNetwork/input_mlp/dense/BiasAdd/ReadVariableOpReadVariableOp<actornetwork_input_mlp_dense_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0�
$ActorNetwork/input_mlp/dense/BiasAddBiasAdd-ActorNetwork/input_mlp/dense/MatMul:product:0;ActorNetwork/input_mlp/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d�
!ActorNetwork/input_mlp/dense/ReluRelu-ActorNetwork/input_mlp/dense/BiasAdd:output:0*
T0*'
_output_shapes
:���������d�
4ActorNetwork/input_mlp/dense/MatMul_1/ReadVariableOpReadVariableOp=actornetwork_input_mlp_dense_matmul_1_readvariableop_resource*
_output_shapes

:dd*
dtype0�
%ActorNetwork/input_mlp/dense/MatMul_1MatMul/ActorNetwork/input_mlp/dense/Relu:activations:0<ActorNetwork/input_mlp/dense/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d�
5ActorNetwork/input_mlp/dense/BiasAdd_1/ReadVariableOpReadVariableOp>actornetwork_input_mlp_dense_biasadd_1_readvariableop_resource*
_output_shapes
:d*
dtype0�
&ActorNetwork/input_mlp/dense/BiasAdd_1BiasAdd/ActorNetwork/input_mlp/dense/MatMul_1:product:0=ActorNetwork/input_mlp/dense/BiasAdd_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d�
#ActorNetwork/input_mlp/dense/Relu_1Relu/ActorNetwork/input_mlp/dense/BiasAdd_1:output:0*
T0*'
_output_shapes
:���������d�
)ActorNetwork/action/MatMul/ReadVariableOpReadVariableOp2actornetwork_action_matmul_readvariableop_resource*
_output_shapes

:d	*
dtype0�
ActorNetwork/action/MatMulMatMul1ActorNetwork/input_mlp/dense/Relu_1:activations:01ActorNetwork/action/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������	�
*ActorNetwork/action/BiasAdd/ReadVariableOpReadVariableOp3actornetwork_action_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype0�
ActorNetwork/action/BiasAddBiasAdd$ActorNetwork/action/MatMul:product:02ActorNetwork/action/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������	x
ActorNetwork/action/TanhTanh$ActorNetwork/action/BiasAdd:output:0*
T0*'
_output_shapes
:���������	k
ActorNetwork/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"����	   �
ActorNetwork/ReshapeReshapeActorNetwork/action/Tanh:y:0#ActorNetwork/Reshape/shape:output:0*
T0*'
_output_shapes
:���������	W
ActorNetwork/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
ActorNetwork/mulMulActorNetwork/mul/x:output:0ActorNetwork/Reshape:output:0*
T0*'
_output_shapes
:���������	W
ActorNetwork/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?~
ActorNetwork/addAddV2ActorNetwork/add/x:output:0ActorNetwork/mul:z:0*
T0*'
_output_shapes
:���������	W
Deterministic/atolConst*
_output_shapes
: *
dtype0*
valueB
 *    W
Deterministic/rtolConst*
_output_shapes
: *
dtype0*
valueB
 *    d
!Deterministic/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB ^
Deterministic/sample/ShapeShapeActorNetwork/add:z:0*
T0*
_output_shapes
:\
Deterministic/sample/ConstConst*
_output_shapes
: *
dtype0*
value	B : r
(Deterministic/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*Deterministic/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*Deterministic/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
"Deterministic/sample/strided_sliceStridedSlice#Deterministic/sample/Shape:output:01Deterministic/sample/strided_slice/stack:output:03Deterministic/sample/strided_slice/stack_1:output:03Deterministic/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskh
%Deterministic/sample/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB j
'Deterministic/sample/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB �
"Deterministic/sample/BroadcastArgsBroadcastArgs0Deterministic/sample/BroadcastArgs/s0_1:output:0+Deterministic/sample/strided_slice:output:0*
_output_shapes
:n
$Deterministic/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:g
$Deterministic/sample/concat/values_2Const*
_output_shapes
: *
dtype0*
valueB b
 Deterministic/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Deterministic/sample/concatConcatV2-Deterministic/sample/concat/values_0:output:0'Deterministic/sample/BroadcastArgs:r0:0-Deterministic/sample/concat/values_2:output:0)Deterministic/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:�
 Deterministic/sample/BroadcastToBroadcastToActorNetwork/add:z:0$Deterministic/sample/concat:output:0*
T0*+
_output_shapes
:���������	u
Deterministic/sample/Shape_1Shape)Deterministic/sample/BroadcastTo:output:0*
T0*
_output_shapes
:t
*Deterministic/sample/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:v
,Deterministic/sample/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: v
,Deterministic/sample/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
$Deterministic/sample/strided_slice_1StridedSlice%Deterministic/sample/Shape_1:output:03Deterministic/sample/strided_slice_1/stack:output:05Deterministic/sample/strided_slice_1/stack_1:output:05Deterministic/sample/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_maskd
"Deterministic/sample/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Deterministic/sample/concat_1ConcatV2*Deterministic/sample/sample_shape:output:0-Deterministic/sample/strided_slice_1:output:0+Deterministic/sample/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
Deterministic/sample/ReshapeReshape)Deterministic/sample/BroadcastTo:output:0&Deterministic/sample/concat_1:output:0*
T0*'
_output_shapes
:���������	\
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
clip_by_value/MinimumMinimum%Deterministic/sample/Reshape:output:0 clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:���������	T
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*'
_output_shapes
:���������	`
IdentityIdentityclip_by_value:z:0^NoOp*
T0*'
_output_shapes
:���������	�
NoOpNoOp+^ActorNetwork/action/BiasAdd/ReadVariableOp*^ActorNetwork/action/MatMul/ReadVariableOp4^ActorNetwork/input_mlp/dense/BiasAdd/ReadVariableOp6^ActorNetwork/input_mlp/dense/BiasAdd_1/ReadVariableOp3^ActorNetwork/input_mlp/dense/MatMul/ReadVariableOp5^ActorNetwork/input_mlp/dense/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*_
_input_shapesN
L:���������:���������:���������:���������: : : : : : 2X
*ActorNetwork/action/BiasAdd/ReadVariableOp*ActorNetwork/action/BiasAdd/ReadVariableOp2V
)ActorNetwork/action/MatMul/ReadVariableOp)ActorNetwork/action/MatMul/ReadVariableOp2j
3ActorNetwork/input_mlp/dense/BiasAdd/ReadVariableOp3ActorNetwork/input_mlp/dense/BiasAdd/ReadVariableOp2n
5ActorNetwork/input_mlp/dense/BiasAdd_1/ReadVariableOp5ActorNetwork/input_mlp/dense/BiasAdd_1/ReadVariableOp2h
2ActorNetwork/input_mlp/dense/MatMul/ReadVariableOp2ActorNetwork/input_mlp/dense/MatMul/ReadVariableOp2l
4ActorNetwork/input_mlp/dense/MatMul_1/ReadVariableOp4ActorNetwork/input_mlp/dense/MatMul_1/ReadVariableOp:N J
#
_output_shapes
:���������
#
_user_specified_name	time_step:NJ
#
_output_shapes
:���������
#
_user_specified_name	time_step:NJ
#
_output_shapes
:���������
#
_user_specified_name	time_step:RN
'
_output_shapes
:���������
#
_user_specified_name	time_step
�
�
*__inference_function_with_signature_294100
	step_type

reward
discount
observation
unknown:d
	unknown_0:d
	unknown_1:dd
	unknown_2:d
	unknown_3:d	
	unknown_4:	
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCall	step_typerewarddiscountobservationunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������	*(
_read_only_resource_inputs

	*0
config_proto 

CPU

GPU2*0J 8� *1
f,R*
(__inference_polymorphic_action_fn_294085o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������	`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*_
_input_shapesN
L:���������:���������:���������:���������: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
#
_output_shapes
:���������
%
_user_specified_name0/step_type:MI
#
_output_shapes
:���������
"
_user_specified_name
0/reward:OK
#
_output_shapes
:���������
$
_user_specified_name
0/discount:VR
'
_output_shapes
:���������
'
_user_specified_name0/observation
�
,
*__inference_function_with_signature_294152�
PartitionedCallPartitionedCall*	
Tin
 *

Tout
 *
_collective_manager_ids
 *
_output_shapes
 * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *!
fR
__inference_<lambda>_781*(
_construction_contextkEagerRuntime*
_input_shapes 
�
_
__inference_<lambda>_778!
readvariableop_resource:	 
identity	��ReadVariableOp^
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0	T
IdentityIdentityReadVariableOp:value:0^NoOp*
T0	*
_output_shapes
: W
NoOpNoOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2 
ReadVariableOpReadVariableOp
�
6
$__inference_get_initial_state_294128

batch_size*(
_construction_contextkEagerRuntime*
_input_shapes
: :B >

_output_shapes
: 
$
_user_specified_name
batch_size
Y

__inference_<lambda>_781*(
_construction_contextkEagerRuntime*
_input_shapes 
�
�
&__inference_signature_wrapper_41993871
discount
observation

reward
	step_type
unknown:d
	unknown_0:d
	unknown_1:dd
	unknown_2:d
	unknown_3:d	
	unknown_4:	
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCall	step_typerewarddiscountobservationunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������	*(
_read_only_resource_inputs

	*0
config_proto 

CPU

GPU2*0J 8� *3
f.R,
*__inference_function_with_signature_294100o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������	`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*_
_input_shapesN
L:���������:���������:���������:���������: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
#
_output_shapes
:���������
$
_user_specified_name
0/discount:VR
'
_output_shapes
:���������
'
_user_specified_name0/observation:MI
#
_output_shapes
:���������
"
_user_specified_name
0/reward:PL
#
_output_shapes
:���������
%
_user_specified_name0/step_type
�
6
$__inference_get_initial_state_294329

batch_size*(
_construction_contextkEagerRuntime*
_input_shapes
: :B >

_output_shapes
: 
$
_user_specified_name
batch_size
�
f
&__inference_signature_wrapper_41993884
unknown:	 
identity	��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallunknown*
Tin
2*
Tout
2	*
_collective_manager_ids
 *
_output_shapes
: *#
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *3
f.R,
*__inference_function_with_signature_294141^
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0	*
_output_shapes
: `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 22
StatefulPartitionedCallStatefulPartitionedCall
�
j
*__inference_function_with_signature_294141
unknown:	 
identity	��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallunknown*
Tin
2*
Tout
2	*
_collective_manager_ids
 *
_output_shapes
: *#
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *!
fR
__inference_<lambda>_778^
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0	*
_output_shapes
: `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 22
StatefulPartitionedCallStatefulPartitionedCall
�
<
*__inference_function_with_signature_294129

batch_size�
PartitionedCallPartitionedCall
batch_size*
Tin
2*

Tout
 *
_collective_manager_ids
 *
_output_shapes
 * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *-
f(R&
$__inference_get_initial_state_294128*(
_construction_contextkEagerRuntime*
_input_shapes
: :B >

_output_shapes
: 
$
_user_specified_name
batch_size
�
�
!__inference__traced_save_41993937
file_prefix'
#savev2_variable_read_readvariableop	D
@savev2_actornetwork_input_mlp_dense_kernel_1_read_readvariableopB
>savev2_actornetwork_input_mlp_dense_bias_1_read_readvariableopB
>savev2_actornetwork_input_mlp_dense_kernel_read_readvariableop@
<savev2_actornetwork_input_mlp_dense_bias_read_readvariableop9
5savev2_actornetwork_action_kernel_read_readvariableop7
3savev2_actornetwork_action_bias_read_readvariableop
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
: �
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B%train_step/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/0/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/1/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/2/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/3/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/4/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/5/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH}
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*#
valueBB B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0#savev2_variable_read_readvariableop@savev2_actornetwork_input_mlp_dense_kernel_1_read_readvariableop>savev2_actornetwork_input_mlp_dense_bias_1_read_readvariableop>savev2_actornetwork_input_mlp_dense_kernel_read_readvariableop<savev2_actornetwork_input_mlp_dense_bias_read_readvariableop5savev2_actornetwork_action_kernel_read_readvariableop3savev2_actornetwork_action_bias_read_readvariableopsavev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtypes

2	�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
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

identity_1Identity_1:output:0*I
_input_shapes8
6: : :d:d:dd:d:d	:	: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: :$ 

_output_shapes

:d: 

_output_shapes
:d:$ 

_output_shapes

:dd: 

_output_shapes
:d:$ 

_output_shapes

:d	: 

_output_shapes
:	:

_output_shapes
: 
�3
�
.__inference_polymorphic_distribution_fn_294326
	step_type

reward
discount
observationM
;actornetwork_input_mlp_dense_matmul_readvariableop_resource:dJ
<actornetwork_input_mlp_dense_biasadd_readvariableop_resource:dO
=actornetwork_input_mlp_dense_matmul_1_readvariableop_resource:ddL
>actornetwork_input_mlp_dense_biasadd_1_readvariableop_resource:dD
2actornetwork_action_matmul_readvariableop_resource:d	A
3actornetwork_action_biasadd_readvariableop_resource:	
identity

identity_1

identity_2��*ActorNetwork/action/BiasAdd/ReadVariableOp�)ActorNetwork/action/MatMul/ReadVariableOp�3ActorNetwork/input_mlp/dense/BiasAdd/ReadVariableOp�5ActorNetwork/input_mlp/dense/BiasAdd_1/ReadVariableOp�2ActorNetwork/input_mlp/dense/MatMul/ReadVariableOp�4ActorNetwork/input_mlp/dense/MatMul_1/ReadVariableOpm
ActorNetwork/flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"����   �
ActorNetwork/flatten_3/ReshapeReshapeobservation%ActorNetwork/flatten_3/Const:output:0*
T0*'
_output_shapes
:����������
2ActorNetwork/input_mlp/dense/MatMul/ReadVariableOpReadVariableOp;actornetwork_input_mlp_dense_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0�
#ActorNetwork/input_mlp/dense/MatMulMatMul'ActorNetwork/flatten_3/Reshape:output:0:ActorNetwork/input_mlp/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d�
3ActorNetwork/input_mlp/dense/BiasAdd/ReadVariableOpReadVariableOp<actornetwork_input_mlp_dense_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0�
$ActorNetwork/input_mlp/dense/BiasAddBiasAdd-ActorNetwork/input_mlp/dense/MatMul:product:0;ActorNetwork/input_mlp/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d�
!ActorNetwork/input_mlp/dense/ReluRelu-ActorNetwork/input_mlp/dense/BiasAdd:output:0*
T0*'
_output_shapes
:���������d�
4ActorNetwork/input_mlp/dense/MatMul_1/ReadVariableOpReadVariableOp=actornetwork_input_mlp_dense_matmul_1_readvariableop_resource*
_output_shapes

:dd*
dtype0�
%ActorNetwork/input_mlp/dense/MatMul_1MatMul/ActorNetwork/input_mlp/dense/Relu:activations:0<ActorNetwork/input_mlp/dense/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d�
5ActorNetwork/input_mlp/dense/BiasAdd_1/ReadVariableOpReadVariableOp>actornetwork_input_mlp_dense_biasadd_1_readvariableop_resource*
_output_shapes
:d*
dtype0�
&ActorNetwork/input_mlp/dense/BiasAdd_1BiasAdd/ActorNetwork/input_mlp/dense/MatMul_1:product:0=ActorNetwork/input_mlp/dense/BiasAdd_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d�
#ActorNetwork/input_mlp/dense/Relu_1Relu/ActorNetwork/input_mlp/dense/BiasAdd_1:output:0*
T0*'
_output_shapes
:���������d�
)ActorNetwork/action/MatMul/ReadVariableOpReadVariableOp2actornetwork_action_matmul_readvariableop_resource*
_output_shapes

:d	*
dtype0�
ActorNetwork/action/MatMulMatMul1ActorNetwork/input_mlp/dense/Relu_1:activations:01ActorNetwork/action/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������	�
*ActorNetwork/action/BiasAdd/ReadVariableOpReadVariableOp3actornetwork_action_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype0�
ActorNetwork/action/BiasAddBiasAdd$ActorNetwork/action/MatMul:product:02ActorNetwork/action/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������	x
ActorNetwork/action/TanhTanh$ActorNetwork/action/BiasAdd:output:0*
T0*'
_output_shapes
:���������	k
ActorNetwork/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"����	   �
ActorNetwork/ReshapeReshapeActorNetwork/action/Tanh:y:0#ActorNetwork/Reshape/shape:output:0*
T0*'
_output_shapes
:���������	W
ActorNetwork/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
ActorNetwork/mulMulActorNetwork/mul/x:output:0ActorNetwork/Reshape:output:0*
T0*'
_output_shapes
:���������	W
ActorNetwork/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?~
ActorNetwork/addAddV2ActorNetwork/add/x:output:0ActorNetwork/mul:z:0*
T0*'
_output_shapes
:���������	W
Deterministic/atolConst*
_output_shapes
: *
dtype0*
valueB
 *    W
Deterministic/rtolConst*
_output_shapes
: *
dtype0*
valueB
 *    Y
IdentityIdentityDeterministic/atol:output:0^NoOp*
T0*
_output_shapes
: e

Identity_1IdentityActorNetwork/add:z:0^NoOp*
T0*'
_output_shapes
:���������	[

Identity_2IdentityDeterministic/rtol:output:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOp+^ActorNetwork/action/BiasAdd/ReadVariableOp*^ActorNetwork/action/MatMul/ReadVariableOp4^ActorNetwork/input_mlp/dense/BiasAdd/ReadVariableOp6^ActorNetwork/input_mlp/dense/BiasAdd_1/ReadVariableOp3^ActorNetwork/input_mlp/dense/MatMul/ReadVariableOp5^ActorNetwork/input_mlp/dense/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*_
_input_shapesN
L:���������:���������:���������:���������: : : : : : 2X
*ActorNetwork/action/BiasAdd/ReadVariableOp*ActorNetwork/action/BiasAdd/ReadVariableOp2V
)ActorNetwork/action/MatMul/ReadVariableOp)ActorNetwork/action/MatMul/ReadVariableOp2j
3ActorNetwork/input_mlp/dense/BiasAdd/ReadVariableOp3ActorNetwork/input_mlp/dense/BiasAdd/ReadVariableOp2n
5ActorNetwork/input_mlp/dense/BiasAdd_1/ReadVariableOp5ActorNetwork/input_mlp/dense/BiasAdd_1/ReadVariableOp2h
2ActorNetwork/input_mlp/dense/MatMul/ReadVariableOp2ActorNetwork/input_mlp/dense/MatMul/ReadVariableOp2l
4ActorNetwork/input_mlp/dense/MatMul_1/ReadVariableOp4ActorNetwork/input_mlp/dense/MatMul_1/ReadVariableOp:N J
#
_output_shapes
:���������
#
_user_specified_name	step_type:KG
#
_output_shapes
:���������
 
_user_specified_namereward:MI
#
_output_shapes
:���������
"
_user_specified_name
discount:TP
'
_output_shapes
:���������
%
_user_specified_nameobservation
�
(
&__inference_signature_wrapper_41993888�
PartitionedCallPartitionedCall*	
Tin
 *

Tout
 *
_collective_manager_ids
 *
_output_shapes
 * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *3
f.R,
*__inference_function_with_signature_294152*(
_construction_contextkEagerRuntime*
_input_shapes 
�N
�
(__inference_polymorphic_action_fn_294221
	step_type

reward
discount
observationM
;actornetwork_input_mlp_dense_matmul_readvariableop_resource:dJ
<actornetwork_input_mlp_dense_biasadd_readvariableop_resource:dO
=actornetwork_input_mlp_dense_matmul_1_readvariableop_resource:ddL
>actornetwork_input_mlp_dense_biasadd_1_readvariableop_resource:dD
2actornetwork_action_matmul_readvariableop_resource:d	A
3actornetwork_action_biasadd_readvariableop_resource:	
identity��*ActorNetwork/action/BiasAdd/ReadVariableOp�)ActorNetwork/action/MatMul/ReadVariableOp�3ActorNetwork/input_mlp/dense/BiasAdd/ReadVariableOp�5ActorNetwork/input_mlp/dense/BiasAdd_1/ReadVariableOp�2ActorNetwork/input_mlp/dense/MatMul/ReadVariableOp�4ActorNetwork/input_mlp/dense/MatMul_1/ReadVariableOpm
ActorNetwork/flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"����   �
ActorNetwork/flatten_3/ReshapeReshapeobservation%ActorNetwork/flatten_3/Const:output:0*
T0*'
_output_shapes
:����������
2ActorNetwork/input_mlp/dense/MatMul/ReadVariableOpReadVariableOp;actornetwork_input_mlp_dense_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0�
#ActorNetwork/input_mlp/dense/MatMulMatMul'ActorNetwork/flatten_3/Reshape:output:0:ActorNetwork/input_mlp/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d�
3ActorNetwork/input_mlp/dense/BiasAdd/ReadVariableOpReadVariableOp<actornetwork_input_mlp_dense_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0�
$ActorNetwork/input_mlp/dense/BiasAddBiasAdd-ActorNetwork/input_mlp/dense/MatMul:product:0;ActorNetwork/input_mlp/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d�
!ActorNetwork/input_mlp/dense/ReluRelu-ActorNetwork/input_mlp/dense/BiasAdd:output:0*
T0*'
_output_shapes
:���������d�
4ActorNetwork/input_mlp/dense/MatMul_1/ReadVariableOpReadVariableOp=actornetwork_input_mlp_dense_matmul_1_readvariableop_resource*
_output_shapes

:dd*
dtype0�
%ActorNetwork/input_mlp/dense/MatMul_1MatMul/ActorNetwork/input_mlp/dense/Relu:activations:0<ActorNetwork/input_mlp/dense/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d�
5ActorNetwork/input_mlp/dense/BiasAdd_1/ReadVariableOpReadVariableOp>actornetwork_input_mlp_dense_biasadd_1_readvariableop_resource*
_output_shapes
:d*
dtype0�
&ActorNetwork/input_mlp/dense/BiasAdd_1BiasAdd/ActorNetwork/input_mlp/dense/MatMul_1:product:0=ActorNetwork/input_mlp/dense/BiasAdd_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d�
#ActorNetwork/input_mlp/dense/Relu_1Relu/ActorNetwork/input_mlp/dense/BiasAdd_1:output:0*
T0*'
_output_shapes
:���������d�
)ActorNetwork/action/MatMul/ReadVariableOpReadVariableOp2actornetwork_action_matmul_readvariableop_resource*
_output_shapes

:d	*
dtype0�
ActorNetwork/action/MatMulMatMul1ActorNetwork/input_mlp/dense/Relu_1:activations:01ActorNetwork/action/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������	�
*ActorNetwork/action/BiasAdd/ReadVariableOpReadVariableOp3actornetwork_action_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype0�
ActorNetwork/action/BiasAddBiasAdd$ActorNetwork/action/MatMul:product:02ActorNetwork/action/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������	x
ActorNetwork/action/TanhTanh$ActorNetwork/action/BiasAdd:output:0*
T0*'
_output_shapes
:���������	k
ActorNetwork/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"����	   �
ActorNetwork/ReshapeReshapeActorNetwork/action/Tanh:y:0#ActorNetwork/Reshape/shape:output:0*
T0*'
_output_shapes
:���������	W
ActorNetwork/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
ActorNetwork/mulMulActorNetwork/mul/x:output:0ActorNetwork/Reshape:output:0*
T0*'
_output_shapes
:���������	W
ActorNetwork/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?~
ActorNetwork/addAddV2ActorNetwork/add/x:output:0ActorNetwork/mul:z:0*
T0*'
_output_shapes
:���������	W
Deterministic/atolConst*
_output_shapes
: *
dtype0*
valueB
 *    W
Deterministic/rtolConst*
_output_shapes
: *
dtype0*
valueB
 *    d
!Deterministic/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB ^
Deterministic/sample/ShapeShapeActorNetwork/add:z:0*
T0*
_output_shapes
:\
Deterministic/sample/ConstConst*
_output_shapes
: *
dtype0*
value	B : r
(Deterministic/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*Deterministic/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*Deterministic/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
"Deterministic/sample/strided_sliceStridedSlice#Deterministic/sample/Shape:output:01Deterministic/sample/strided_slice/stack:output:03Deterministic/sample/strided_slice/stack_1:output:03Deterministic/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskh
%Deterministic/sample/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB j
'Deterministic/sample/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB �
"Deterministic/sample/BroadcastArgsBroadcastArgs0Deterministic/sample/BroadcastArgs/s0_1:output:0+Deterministic/sample/strided_slice:output:0*
_output_shapes
:n
$Deterministic/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:g
$Deterministic/sample/concat/values_2Const*
_output_shapes
: *
dtype0*
valueB b
 Deterministic/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Deterministic/sample/concatConcatV2-Deterministic/sample/concat/values_0:output:0'Deterministic/sample/BroadcastArgs:r0:0-Deterministic/sample/concat/values_2:output:0)Deterministic/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:�
 Deterministic/sample/BroadcastToBroadcastToActorNetwork/add:z:0$Deterministic/sample/concat:output:0*
T0*+
_output_shapes
:���������	u
Deterministic/sample/Shape_1Shape)Deterministic/sample/BroadcastTo:output:0*
T0*
_output_shapes
:t
*Deterministic/sample/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:v
,Deterministic/sample/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: v
,Deterministic/sample/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
$Deterministic/sample/strided_slice_1StridedSlice%Deterministic/sample/Shape_1:output:03Deterministic/sample/strided_slice_1/stack:output:05Deterministic/sample/strided_slice_1/stack_1:output:05Deterministic/sample/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_maskd
"Deterministic/sample/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Deterministic/sample/concat_1ConcatV2*Deterministic/sample/sample_shape:output:0-Deterministic/sample/strided_slice_1:output:0+Deterministic/sample/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
Deterministic/sample/ReshapeReshape)Deterministic/sample/BroadcastTo:output:0&Deterministic/sample/concat_1:output:0*
T0*'
_output_shapes
:���������	\
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
clip_by_value/MinimumMinimum%Deterministic/sample/Reshape:output:0 clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:���������	T
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*'
_output_shapes
:���������	`
IdentityIdentityclip_by_value:z:0^NoOp*
T0*'
_output_shapes
:���������	�
NoOpNoOp+^ActorNetwork/action/BiasAdd/ReadVariableOp*^ActorNetwork/action/MatMul/ReadVariableOp4^ActorNetwork/input_mlp/dense/BiasAdd/ReadVariableOp6^ActorNetwork/input_mlp/dense/BiasAdd_1/ReadVariableOp3^ActorNetwork/input_mlp/dense/MatMul/ReadVariableOp5^ActorNetwork/input_mlp/dense/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*_
_input_shapesN
L:���������:���������:���������:���������: : : : : : 2X
*ActorNetwork/action/BiasAdd/ReadVariableOp*ActorNetwork/action/BiasAdd/ReadVariableOp2V
)ActorNetwork/action/MatMul/ReadVariableOp)ActorNetwork/action/MatMul/ReadVariableOp2j
3ActorNetwork/input_mlp/dense/BiasAdd/ReadVariableOp3ActorNetwork/input_mlp/dense/BiasAdd/ReadVariableOp2n
5ActorNetwork/input_mlp/dense/BiasAdd_1/ReadVariableOp5ActorNetwork/input_mlp/dense/BiasAdd_1/ReadVariableOp2h
2ActorNetwork/input_mlp/dense/MatMul/ReadVariableOp2ActorNetwork/input_mlp/dense/MatMul/ReadVariableOp2l
4ActorNetwork/input_mlp/dense/MatMul_1/ReadVariableOp4ActorNetwork/input_mlp/dense/MatMul_1/ReadVariableOp:N J
#
_output_shapes
:���������
#
_user_specified_name	step_type:KG
#
_output_shapes
:���������
 
_user_specified_namereward:MI
#
_output_shapes
:���������
"
_user_specified_name
discount:TP
'
_output_shapes
:���������
%
_user_specified_nameobservation
�
8
&__inference_signature_wrapper_41993876

batch_size�
PartitionedCallPartitionedCall
batch_size*
Tin
2*

Tout
 *
_collective_manager_ids
 *
_output_shapes
 * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *3
f.R,
*__inference_function_with_signature_294129*(
_construction_contextkEagerRuntime*
_input_shapes
: :B >

_output_shapes
: 
$
_user_specified_name
batch_size"�
L
saver_filename:0StatefulPartitionedCall_2:0StatefulPartitionedCall_38"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
action�
4

0/discount&
action_0_discount:0���������
>
0/observation-
action_0_observation:0���������
0
0/reward$
action_0_reward:0���������
6
0/step_type'
action_0_step_type:0���������:
action0
StatefulPartitionedCall:0���������	tensorflow/serving/predict*e
get_initial_stateP
2

batch_size$
get_initial_state_batch_size:0 tensorflow/serving/predict*,
get_metadatatensorflow/serving/predict*Z
get_train_stepH*
int64!
StatefulPartitionedCall_1:0	 tensorflow/serving/predict:�]
�

train_step
metadata
model_variables
_all_assets

action
distribution
get_initial_state
get_metadata
	get_train_step


signatures"
_generic_user_object
:	 (2Variable
 "
trackable_dict_wrapper
K
0
1
2
3
4
5"
trackable_tuple_wrapper
4
_actor_network"
trackable_dict_wrapper
�
trace_0
trace_12�
(__inference_polymorphic_action_fn_294221
(__inference_polymorphic_action_fn_294286�
���
FullArgSpec(
args �
j	time_step
jpolicy_state
varargs
 
varkw
 
defaults�
� 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 ztrace_0ztrace_1
�
trace_02�
.__inference_polymorphic_distribution_fn_294326�
���
FullArgSpec(
args �
j	time_step
jpolicy_state
varargs
 
varkw
 
defaults�
� 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 ztrace_0
�
trace_02�
$__inference_get_initial_state_294329�
���
FullArgSpec!
args�
jself
j
batch_size
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
 ztrace_0
�B�
__inference_<lambda>_781"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
__inference_<lambda>_778"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
`

action
get_initial_state
get_train_step
get_metadata"
signature_map
5:3d2#ActorNetwork/input_mlp/dense/kernel
/:-d2!ActorNetwork/input_mlp/dense/bias
5:3dd2#ActorNetwork/input_mlp/dense/kernel
/:-d2!ActorNetwork/input_mlp/dense/bias
,:*d	2ActorNetwork/action/kernel
&:$	2ActorNetwork/action/bias
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
 _mlp_layers"
_tf_keras_layer
�B�
(__inference_polymorphic_action_fn_294221	step_typerewarddiscountobservation"�
���
FullArgSpec(
args �
j	time_step
jpolicy_state
varargs
 
varkw
 
defaults�
� 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
(__inference_polymorphic_action_fn_294286time_step_step_typetime_step_rewardtime_step_discounttime_step_observation"�
���
FullArgSpec(
args �
j	time_step
jpolicy_state
varargs
 
varkw
 
defaults�
� 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
.__inference_polymorphic_distribution_fn_294326	step_typerewarddiscountobservation"�
���
FullArgSpec(
args �
j	time_step
jpolicy_state
varargs
 
varkw
 
defaults�
� 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
$__inference_get_initial_state_294329
batch_size"�
���
FullArgSpec!
args�
jself
j
batch_size
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
&__inference_signature_wrapper_41993871
0/discount0/observation0/reward0/step_type"�
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
�B�
&__inference_signature_wrapper_41993876
batch_size"�
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
�B�
&__inference_signature_wrapper_41993884"�
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
�B�
&__inference_signature_wrapper_41993888"�
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
J
0
1
2
3
4
5"
trackable_list_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
 "
trackable_list_wrapper
�
!non_trainable_variables

"layers
#metrics
$layer_regularization_losses
%layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpecM
argsE�B
jself
jobservations
j	step_type
jnetwork_state

jtraining
varargs
 
varkw
 
defaults�
� 
� 
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpecM
argsE�B
jself
jobservations
j	step_type
jnetwork_state

jtraining
varargs
 
varkw
 
defaults�
� 
� 
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
<
&0
'1
(2
)3"
trackable_list_wrapper
 "
trackable_list_wrapper
<
&0
'1
(2
)3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�
*	variables
+trainable_variables
,regularization_losses
-	keras_api
.__call__
*/&call_and_return_all_conditional_losses"
_tf_keras_layer
�
0	variables
1trainable_variables
2regularization_losses
3	keras_api
4__call__
*5&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
�
6	variables
7trainable_variables
8regularization_losses
9	keras_api
:__call__
*;&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
�
<	variables
=trainable_variables
>regularization_losses
?	keras_api
@__call__
*A&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
Bnon_trainable_variables

Clayers
Dmetrics
Elayer_regularization_losses
Flayer_metrics
*	variables
+trainable_variables
,regularization_losses
.__call__
*/&call_and_return_all_conditional_losses
&/"call_and_return_conditional_losses"
_generic_user_object
�2��
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
�2��
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
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
Gnon_trainable_variables

Hlayers
Imetrics
Jlayer_regularization_losses
Klayer_metrics
0	variables
1trainable_variables
2regularization_losses
4__call__
*5&call_and_return_all_conditional_losses
&5"call_and_return_conditional_losses"
_generic_user_object
�2��
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
�2��
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
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
Lnon_trainable_variables

Mlayers
Nmetrics
Olayer_regularization_losses
Player_metrics
6	variables
7trainable_variables
8regularization_losses
:__call__
*;&call_and_return_all_conditional_losses
&;"call_and_return_conditional_losses"
_generic_user_object
�2��
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
�2��
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
�
Qnon_trainable_variables

Rlayers
Smetrics
Tlayer_regularization_losses
Ulayer_metrics
<	variables
=trainable_variables
>regularization_losses
@__call__
*A&call_and_return_all_conditional_losses
&A"call_and_return_conditional_losses"
_generic_user_object
�2��
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
�2��
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
trackable_dict_wrapper@
__inference_<lambda>_778$�

� 
� "�
unknown 	0
__inference_<lambda>_781�

� 
� "� Q
$__inference_get_initial_state_294329)"�
�
�

batch_size 
� "� �
(__inference_polymorphic_action_fn_294221����
���
���
TimeStep,
	step_type�
	step_type���������&
reward�
reward���������*
discount�
discount���������4
observation%�"
observation���������
� 
� "V�S

PolicyStep*
action �
action���������	
state� 
info� �
(__inference_polymorphic_action_fn_294286����
���
���
TimeStep6
	step_type)�&
time_step_step_type���������0
reward&�#
time_step_reward���������4
discount(�%
time_step_discount���������>
observation/�,
time_step_observation���������
� 
� "V�S

PolicyStep*
action �
action���������	
state� 
info� �
.__inference_polymorphic_distribution_fn_294326����
���
���
TimeStep,
	step_type�
	step_type���������&
reward�
reward���������*
discount�
discount���������4
observation%�"
observation���������
� 
� "���

PolicyStep�
action������
`
F�C

atol� 

loc����������	

rtol� 
L�I

allow_nan_statsp

namejDeterministic_1_1

validate_argsp 
�
j
parameters
� 
�
jname+tfp.distributions.Deterministic_ACTTypeSpec 
state� 
info� �
&__inference_signature_wrapper_41993871����
� 
���
5

0/discount'�$
tensor_0_discount���������
?
0/observation.�+
tensor_0_observation���������
1
0/reward%�"
tensor_0_reward���������
7
0/step_type(�%
tensor_0_step_type���������"/�,
*
action �
action���������	a
&__inference_signature_wrapper_4199387670�-
� 
&�#
!

batch_size�

batch_size "� Z
&__inference_signature_wrapper_419938840�

� 
� "�

int64�
int64 	>
&__inference_signature_wrapper_41993888�

� 
� "� 